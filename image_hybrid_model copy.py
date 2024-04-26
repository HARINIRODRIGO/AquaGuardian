import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import concurrent.futures
import cv2
import supervision as sv
from mmdet.apis import init_detector, inference_detector
from ensemble_boxes import WeightedBoxFusion
from ultralytics import YOLO
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# Define device for model loading (cuda if available, cpu otherwise)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define paths to model files and dataset location
FILE_PATH = "Models"
CONFIG_PATH = f"{FILE_PATH}/custom_30.py"
WEIGHTS_PATH = f"{FILE_PATH}/epoch_30.pth"
YOLO_MODEL_PATH = f"{FILE_PATH}/best-35.pt"
dataset_location = "AquaGuardian-V1-1"

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

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        ground_truth_labels = []
        predicted_labels = []
        for line in lines:
            # Split the line by spaces
            data = line.strip().split()
            # The first element is the ground truth label, and the rest are the predicted bounding box data
            gt_label = int(data[0])
            # Append the ground truth label to the list
            ground_truth_labels.append(gt_label)
            # Assuming the predicted label is the class with the highest confidence score
            # You may need to modify this logic based on your prediction format
            pred_label = np.argmax([float(data[i]) for i in range(1, len(data), 6)])
            # Append the predicted label to the list
            predicted_labels.append(pred_label)
    return ground_truth_labels, predicted_labels

# Define the paths to the directories containing label files and images
label_dir = "path_to_label_directory"
image_dir = "path_to_image_directory"

# Get the list of label files
label_files = os.listdir(label_dir)

# Initialize a dictionary to store confusion matrices for each file
conf_matrices = {}

# Iterate over each label file
for label_file in label_files:
    # Construct the path to the label file
    label_file_path = os.path.join(label_dir, label_file)
    # Read ground truth labels and predicted labels from the file
    ground_truth_labels, predicted_labels = read_file(label_file_path)
    # Calculate confusion matrix using scikit-learn
    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)
    # Store the confusion matrix in the dictionary with the file name as the key
    conf_matrices[label_file] = conf_matrix

# Print or analyze confusion matrices as needed
for label_file, conf_matrix in conf_matrices.items():
    print("Confusion Matrix for", label_file, ":\n", conf_matrix)

# References:
# Solovyev, R., Wang, W. and Gabruseva, T. (2021). Weighted boxes fusion: Ensembling boxes from different object detection models. Available from https://github.com/ZFTurbo/Weighted-Boxes-Fusion.

