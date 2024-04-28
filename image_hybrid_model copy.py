from collections import defaultdict
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import concurrent.futures
import cv2
from sklearn.metrics import ConfusionMatrixDisplay
import supervision as sv
from mmdet.apis import init_detector, inference_detector
from ensemble_boxes import WeightedBoxFusion
from ultralytics import YOLO
import torch
import os
import json
from multiprocessing import Process
from tqdm import tqdm
# Define device for model loading (cuda if available, cpu otherwise)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define paths to model files and dataset location
FILE_PATH = "Models"
CONFIG_PATH = f"{FILE_PATH}/custom_30.py"
WEIGHTS_PATH = f"{FILE_PATH}/epoch_30.pth"
YOLO_MODEL_PATH = f"{FILE_PATH}/best-35.pt"
dataset_location = "AquaGuardian-V1-1"
NUM_CLASSES = 3

# Load the RTMDet model using mmdet
model_rtmdet = init_detector(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)

# Load the YOLO model
model_yolo = YOLO(f'{YOLO_MODEL_PATH}')
#count = 1

final_boxes = []
final_confidences = []
final_classes = []

def get_predictions(model, image_path):
    """
  This function takes a model and an image path and returns the predictions for that image.

  Args:
      model: The model to use for prediction (either RTMDet or YOLO)
      image_path: The path to the image file

  Returns:
      A tuple containing the bounding boxes, confidence scores, and class labels for the detected objects.
  """
    boxes_confidences = []
    boxes_xyxy = []
    boxes_classes = []
    image = cv2.imread(image_path)

    # Extracting RTMDet model and YOLO model confidence, class, bounding box
    if model == model_rtmdet:
        result = inference_detector(model_rtmdet, image)
        detections = sv.Detections.from_mmdetection(result)
        detections = detections[detections.confidence > 0.5].with_nms()
        return detections.xyxy, detections.confidence, detections.class_id

    elif model == model_yolo:
        results = model.predict(image_path, imgsz=[640, 640], conf=0.5)
        for r in results:
          for box in r.boxes:
              xyxy = [box.xyxy[0][0].item(),box.xyxy[0][1].item(),
              box.xyxy[0][2].item(),box.xyxy[0][3].item()]
              boxes_xyxy.append(xyxy)
              boxes_confidences.append(box.conf.item())
              boxes_classes.append(box.cls.item())
        boxes_classes = [int(a) for a in boxes_classes]
        return boxes_xyxy, boxes_confidences, boxes_classes

# Collect both models confidence, class, bounding box and creating lists
def ensemble_predictions(predictions_model_1, predictions_model_2):
    boxes_model_1, scores_model_1, labels_model_1 = predictions_model_1
    boxes_model_2, scores_model_2, labels_model_2 = predictions_model_2
    all_boxes= [boxes_model_1, boxes_model_2]

    # Call the ensemble method (Weighted Boxes Fusion) with the combined predictions
    fusion_instance = WeightedBoxFusion(iou_threshold=0.8, confidence_type='avg') # iou_threshold can changed to 0.0 - 0.9

    ensemble_boxes, ensemble_scores, ensemble_labels = fusion_instance.fuse_weighted_bounding_boxes_data(
        all_boxes,  # List of boxes from both models
        [scores_model_1, scores_model_2],  # List of scores from both models
        [labels_model_1, labels_model_2],  # List of labels from both models
        weights=[1,6],  # give weights for both models
    )
    
    return ensemble_boxes, ensemble_scores, ensemble_labels


###########
def seg_to_bbox(seg_info):
    class_id, *points = seg_info.split()
    points = [float(p) for p in points]
    x_min, y_min, x_max, y_max = min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])
    width, height = x_max - x_min, y_max - y_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    x1, y1, x2, y2,  = x_center/(640), y_center/(640), width/(640), height/(640)
    return float(class_id), x1, y1, x2, y2

def box_iou_calc(boxes1, boxes2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)

class ConfusionMatrix:
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes, num_classes))  # Adjusted matrix size
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def process_batch(self, detections, labels: np.ndarray):
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except (IndexError, TypeError):
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                # Adjusted index for confusion matrix
                self.matrix[gt_class, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[gt_class, gt_class] += 1

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or (all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes-1] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes):
            print(' '.join(map(str, self.matrix[i])))
        return self
##########

confusion_matrix = ConfusionMatrix(num_classes=NUM_CLASSES)
# Load JSON data
with open('test/labels/_annotations.coco.json', 'r') as file:
    data = json.load(file)

# Extract required information
annotations = data['annotations']
images = data['images']
categories = data['categories']

# Mapping category IDs to category names
category_id_to_name = {category['id']: category['name'] for category in categories}

# Path to the folder containing images
images_folder = 'test\labels'

# Dictionary to collect bbox and category_ids for each image_id
image_data = defaultdict(lambda: {'bboxes': [], 'category_ids': [], "all_": []})

# Extract bbox, image_id, category_id, and file_name
for annotation in annotations:
    bbox = annotation['bbox']
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    data = []
    # Collect bbox and category_id for each image_id
    image_data[image_id]['bboxes'].append(bbox)
    image_data[image_id]['category_ids'].append(category_id)
    data.append(category_id)
    for _ in bbox:
        data.append(_)
    image_data[image_id]['all_'].append(data)
# Process the collected data
for image_id, data in tqdm(image_data.items()):
    # Find file_name using image_id
    file_name = next((image['file_name'] for image in images if image['id'] == image_id), 'Unknown')
    
    # Construct full path to the image file
    image_path = os.path.join(images_folder, file_name)
    
    # Check if the image file exists
    if os.path.exists(image_path):
        # Initialize a list to store results from parallel processes
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            rtmdet_thread = executor.submit(get_predictions, model_rtmdet, image_path)
            predictions_rtmdet = rtmdet_thread.result() 
            yolo_thread = executor.submit(get_predictions, model_yolo, image_path)
            predictions_yolo = yolo_thread.result()
        
        prediction_results = []
        # Ensemble predictions
        predictions_boxes, predictions_scores, predictions_labels = ensemble_predictions(predictions_rtmdet, predictions_yolo)
        for box, score, label in zip(predictions_boxes, predictions_scores, predictions_labels):
            predict = [box[0], box[1], box[2],box[3], score, label]
            prediction_results.append(predict) 
        confusion_matrix.process_batch(np.array(prediction_results), np.array(data['all_']))
        
    else:
        pass
plot = confusion_matrix.print_matrix()
cm_display = ConfusionMatrixDisplay(confusion_matrix = plot.matrix, display_labels = ["bio", "trash", "rov", "FP"])
cm_display.plot()
plt.show()
# References:
# Solovyev, R., Wang, W. and Gabruseva, T. (2021). Weighted boxes fusion: Ensembling boxes from different object detection models. Available from https://github.com/ZFTurbo/Weighted-Boxes-Fusion.

