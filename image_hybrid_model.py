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

def displayHybridPrediction(predictions_rtmdet, predictions_yolo, IMAGE_PATH):
    """
    Visualizes final predictions on the image.
    Includes bounding boxes, confidence scores, and class labels.
    Saves the visualized image to a file.
    """
    ensemble_prediction = ensemble_predictions(predictions_rtmdet, predictions_yolo)
    print(ensemble_prediction)  # Check ensemble predictions
    image = cv2.imread(IMAGE_PATH)  # Read the image using OpenCV
    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100))  # Set figure size based on image size
    # Display the image without axis
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
    ax.axis('off')  # Turn off axis
    # Iterate over the final_boxes, final_confidences, and final_classes
    for box, confidence, class_ in zip(*ensemble_prediction):
        # box format: [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        # Create a rectangle patch
        if class_ == 0.0:
            class_ = "bio"
             # Add confidence text at the top of each bounding box
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='g', facecolor='none')
            ax.text(x_min, y_min - 10, f'{class_}:{confidence:.2f}', color='g', fontsize=10, ha='left', va='center',bbox=dict(facecolor='white', alpha=0.7, edgecolor='white'))
        elif class_ == 1.0:
            class_ = "rov"
            # Add confidence text at the top of each bounding box
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='b', facecolor='none')
            ax.text(x_min, y_min - 10, f'{class_}:{confidence:.2f}', color='b', fontsize=10, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='white'))
        elif class_ == 2.0:
            class_ = "trash"
            # Add confidence text at the top of each bounding box
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.text(x_min, y_min - 10, f'{class_}:{confidence:.2f}', color='r', fontsize=10, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='white'))
        # Add the patch to the plot
        ax.add_patch(rect)
    # Save the modified image with bounding boxes and confidence scores
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"hybrid_model_detection/output_image_{current_time}.jpg"
    # Save the modified image with bounding boxes and confidence scores
    plt.savefig(output_filename, bbox_inches='tight')
    return output_filename

# for i in range (1,100):
#     IMAGE_PATH = f"{dataset_location}/test/images/{i}.jpg"
#     image = cv2.imread(IMAGE_PATH)
#     result = inference_detector(model_rtmdet, image)
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         rtmdet_thread = executor.submit(get_predictions, model_rtmdet, IMAGE_PATH)
#         predictions_rtmdet = rtmdet_thread.result() 
#         yolo_thread = executor.submit(get_predictions, model_yolo, IMAGE_PATH)
#         predictions_yolo = yolo_thread.result()
#     # Placeholder variables for final predictions
#     final_boxes, final_confidences, final_classes = displayHybridPrediction(predictions_rtmdet, predictions_yolo, IMAGE_PATH=IMAGE_PATH)
#     count +=1

# References:
# Solovyev, R., Wang, W. and Gabruseva, T. (2021). Weighted boxes fusion: Ensembling boxes from different object detection models. Available from https://github.com/ZFTurbo/Weighted-Boxes-Fusion.

