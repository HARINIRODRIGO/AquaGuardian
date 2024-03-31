import cv2
import torch
from mmdet.apis import init_detector, inference_detector
import supervision as sv
import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from threading import Thread
import concurrent.futures
ultralytics.checks()

# Define device for model loading (cuda if available, cpu otherwise)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define paths to model files and dataset location
FILE_PATH = "Models"
CONFIG_PATH = f"{FILE_PATH}/custom_30.py"
WEIGHTS_PATH = f"{FILE_PATH}/epoch_30.pth"
YOLO_MODEL_PATH = f"{FILE_PATH}/best-35.pt"
dataset_location = "AquaGuardian-V1-1_test_img"

# Load the RTMDet model using mmdet
model_rtmdet = init_detector(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)

# Load the YOLO model
model_yolo = YOLO(f'{YOLO_MODEL_PATH}')
count = 1

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

    if model == model_rtmdet:
        # Use mmdet inference detector to get predictions for RTMDet model
        result = inference_detector(model_rtmdet, image)
        detections = sv.Detections.from_mmdetection(result)
        detections = detections[detections.confidence > 0.5].with_nms()
        return detections.xyxy, detections.confidence, detections.class_id

    elif model == model_yolo:
         # Use YOLO model for prediction
        results = model.predict(image_path, imgsz=[640, 640], conf=0.5)
        for r in results:
          for box in r.boxes:
              # Extract bounding box coordinates and confidence score from YOLO results
              xyxy = [box.xyxy[0][0].item(),box.xyxy[0][1].item(),
              box.xyxy[0][2].item(),box.xyxy[0][3].item()]
              boxes_xyxy.append(xyxy)
              boxes_confidences.append(box.conf.item())
              boxes_classes.append(box.cls.item())
        return boxes_xyxy, boxes_confidences, boxes_classes

def addFinalizedData(class_, bbox_, conf_):
   """
  This function adds the class label, bounding box, and confidence score to the final predictions list.

  Args:
      class_: The class label of the detected object
      bbox_: The bounding box coordinates for the detected object
      conf_: The confidence score for the detected object
  """
   final_boxes.append(bbox_) 
   final_confidences.append(conf_)
   final_classes.append(class_)

def bbox_conf(bboxes, confs, pos):
    """
  This function retrieves the bounding box and confidence score at a specific position in the lists.

  Args:
      bboxes: The list of bounding boxes
      confs: The list of confidence scores
      pos: The position (index) in the lists to retrieve data from

  Returns:
      A tuple containing the bounding box and confidence score at the specified position.
  """
    bbox  = bboxes[pos]
    conf = confs[pos]
    return bbox, conf

# chack whether one array is a sub set of other array or not. return boolean value, min array, and maximum array
def isSubset(yolo_class, rtmdet_class):
    """
    This function checks if one list of class labels (`yolo_class` or `rtmdet_class`) 
    is a subset of the other. A subset means all elements in the smaller list 
    are also present in the larger list.

    Args:
        yolo_class: A list of class labels predicted by the YOLO model.
        rtmdet_class: A list of class labels predicted by the RTMDet model.

    Returns:
        A tuple containing:
            - is_subset (bool): True if one list is a subset of the other, False otherwise.
            - smaller_list (list): The list containing the smaller number of classes (subset).
            - larger_list (list): The list containing the larger number of classes (superset).
            - model_with_larger_set (str): The name of the model that detected the larger set of classes ("yolo" or "rtmdet").
    """

    # Find the length of each list
    max_len = max(len(yolo_class), len(rtmdet_class))
    # Convert both lists to sets to enable efficient checking for subset relationship
    yolo_class_set = set(yolo_class)
    rtmdet_class_set = set(rtmdet_class)
    # Check if the set of YOLO classes is a subset of the set of RTMDet classes
    if max_len == len(yolo_class) and yolo_class_set.issuperset(rtmdet_class):
        return True, rtmdet_class, "yolo"
    
    # Check if the set of RTMDet classes is a subset of the set of YOLO classes
    elif max_len == len(rtmdet_class) and rtmdet_class_set.issuperset(yolo_class):
        return True, yolo_class, "rtmdet"
    # If neither list is a subset of the other, return False with empty lists
    else:
        return False, [], "none"

def checkArraysSimilarity(rtmdet_confs, yolo_confs, rtmdet_bboxes, yolo_bboxes, nonsorted_rtmdet_classes, nonsorted_yolo_classes, class_):
  """
  This function iterates through a specified list of classes (`class_`) and checks for corresponding detections in both the RTMDet and YOLO model outputs. 
  If a class is present in both models, it prioritizes the model with the higher confidence score for that class.

  Args:
      rtmdet_confs (list): List of confidence scores from the RTMDet model.
      yolo_confs (list): List of confidence scores from the YOLO model.
      rtmdet_bboxes (list): List of bounding boxes from the RTMDet model.
      yolo_bboxes (list): List of bounding boxes from the YOLO model.
      nonsorted_rtmdet_classes (list): Non-sorted list of class labels from the RTMDet model.
      nonsorted_yolo_classes (list): Non-sorted list of class labels from the YOLO model.
      class_ (list, optional): A specific list of classes to check. If None, all classes 
          are considered. Defaults to None.

  Returns:
      None: This function modifies the final_boxes, final_confidences, and final_classes 
          lists defined globally in the script to include the final detections.
  """
  # If no specific list of classes is provided, consider all classes from both models
  if class_ is None:
      class_ = list(set(nonsorted_rtmdet_classes + nonsorted_yolo_classes))  # Combine and remove duplicates
# Iterate through the specified classes
  for data in class_:
    bbox = 0  # Placeholder for bounding box
    conf = 0  # Placeholder for confidence score  
    rtmdet_class_pos = nonsorted_rtmdet_classes.index(data)  # Index of class in RTMDet classes
    yolo_class_pos = nonsorted_yolo_classes.index(data)  # Index of class in YOLO classes
    
    # Check if  RTMDet model confident is greater than YOLO model confident
    if rtmdet_confs[rtmdet_class_pos]>yolo_confs[yolo_class_pos]:
        bbox, conf = bbox_conf(rtmdet_bboxes, rtmdet_confs, rtmdet_class_pos)

    # Check if  YOLO model confident is greater than RTMDet model confident   
    elif yolo_confs[yolo_class_pos]>rtmdet_confs[rtmdet_class_pos]:
        bbox, conf = bbox_conf(yolo_bboxes, yolo_confs, yolo_class_pos)

 # Add the final bounding box, confidence score, and class label to the global lists       
    addFinalizedData(class_=data, bbox_=bbox, conf_=conf)

def confChecker(classes,bbox, confs, min_conf):
    """
    This function filters detections based on a minimum confidence threshold (`min_conf`). 
    It iterates through the provided confidence scores (`confs`) and checks if any value 
    is greater than the threshold. If a high-confidence detection is found for a class 
    in `class_`, it adds the corresponding bounding box and class label to the final results.

    Args:
        class_ (list): List of class labels to consider.
        bbox (list): List of bounding boxes (might contain empty elements for classes not detected).
        confs (list): List of confidence scores corresponding to the bounding boxes.
        min_conf (float): Minimum confidence threshold.

    Returns:
        None: This function modifies the final_boxes, final_confidences, and final_classes 
            lists defined globally in the script to include the final detections that pass the confidence threshold.
    """
    # Iterate through the confidence scores
    for class_ in classes:
      index = classes.index(class_)
      conf, bbox_ = confs[index], bbox[index]
      # Check if the confidence score is greater than the minimum threshold
      if(conf > min_conf):
          # Add the high-confidence detection to the final results
          addFinalizedData(class_=class_, bbox_=bbox_, conf_=conf)
          classes.remove(class_)

def checkDiffClass(classes,bbox, confs, min_conf):
    # Iterate through the confidence scores
    for class_ in classes:
      index = classes.index(class_)
      # Check if the confidence score is greater than the minimum threshold
      if(confs[index] > min_conf):
          # Add the high-confidence detection to the final results
          addFinalizedData(class_=class_, bbox_=bbox[index], conf_=confs[index])
          classes[index] = 4.0

def diffSizeArrays(rtmdet_bboxes, rtmdet_confs, yolo_bboxes, yolo_confs, nonsorted_rtmdet_classes, 
                   nonsorted_yolo_classes, yolo_classes, rtmdet_classes):
    """
    This function handles cases where the class lists from the RTMDet and YOLO models have different sizes. 
    It first uses the `isSubset` function to determine if one list is a strict subset of the other. 
    Then, it applies different logic based on the following scenarios:

    1. Subset Relationship:
        - If one list is a subset, the function prioritizes detections for classes in the common set 
          using `checkArraysSimilarity`.
        - For classes that are unique to the larger list (superset), it uses `confChecker` 
          to consider detections with confidence above a threshold.

    2. No Subset Relationship:
        - If neither list is a subset of the other, the function uses `confChecker` independently 
          on both `rtmdet_classes` and `yolo_classes` to consider high-confidence detections from each model.

    Args:
        rtmdet_bboxes (list): List of bounding boxes from the RTMDet model.
        rtmdet_confs (list): List of confidence scores from the RTMDet model.
        yolo_bboxes (list): List of bounding boxes from the YOLO model.
        yolo_confs (list): List of confidence scores from the YOLO model.
        nonsorted_rtmdet_classes (list): Non-sorted list of class labels from the RTMDet model.
        nonsorted_yolo_classes (list): Non-sorted list of class labels from the YOLO model.
        yolo_classes (list): Sorted list of unique class labels from the YOLO model.
        rtmdet_classes (list): Sorted list of unique class labels from the RTMDet model.

    Returns:
        None: This function modifies the final_boxes, final_confidences, and final_classes 
            lists defined globally in the script to include the final detections after applying the logic for different scenarios.
    """
    # Use isSubset function to check for subset relationship between class lists
    boo_, min_arr, max_name =isSubset(yolo_classes, rtmdet_classes)
  # Handle cases where one list is a subset of the other
    if boo_:
            # Call checkArraysSimilarity to handle detections for classes in the common set
            checkArraysSimilarity(rtmdet_bboxes=rtmdet_bboxes, rtmdet_confs=rtmdet_confs, yolo_bboxes=yolo_bboxes,
                            yolo_confs=yolo_confs, nonsorted_rtmdet_classes=nonsorted_rtmdet_classes, 
                            nonsorted_yolo_classes=nonsorted_yolo_classes, class_=min_arr)
            
            # Use confChecker to consider high-confidence detections for unique classes
            if max_name=="yolo":
                checkDiffClass(classes=nonsorted_yolo_classes, min_conf=0.8, confs=yolo_confs, bbox=yolo_bboxes)
            # Handle cases where there's no subset relationship
            else:
                checkDiffClass(classes=nonsorted_rtmdet_classes, min_conf=0.8, confs=rtmdet_confs, bbox=rtmdet_bboxes)
    # not a subset 
    else:
        # Use confChecker for detections from both models with a minimum confidence threshold
        if(yolo_classes!=[]):
            yolo_thr= Thread(target=confChecker, args=(yolo_classes, yolo_bboxes, yolo_confs,0.8 ))
            yolo_thr.start()
        if (rtmdet_classes!=[]):
            rtmdet_thr= Thread(target=confChecker, args=(rtmdet_classes, rtmdet_bboxes, rtmdet_confs,0.8 ))
            rtmdet_thr.start()
  

def vote(predictions_rtmdet, predictions_yolo):
  """
  This function is the core logic for combining predictions from the RTMDet and YOLO models. 
  It unpacks the predictions from both models (bounding boxes, confidence scores, and class labels) 
  and performs different voting strategies based on the following conditions:

  1. Same Class Lists (Equal Length):
      - If the class lists from both models are identical and sorted, the function prioritizes 
        detections with higher confidence scores using `checkArraysSimilarity`.

  2. Different Class Lists:
      - If the class lists have different sizes or elements, the function employs the `diffSizeArrays` 
        function to handle various scenarios:
          - Subset Relationship: Prioritizes detections for common classes and considers high-confidence 
            detections for unique classes in the larger list.
          - No Subset Relationship: Considers high-confidence detections from both models independently.

  Args:
      predictions_rtmdet (tuple): A tuple containing RTMDet model predictions (bounding boxes, confidence scores, class labels).
      predictions_yolo (tuple): A tuple containing YOLO model predictions (bounding boxes, confidence scores, class labels).

  Returns:
      tuple: A tuple containing the final bounding boxes, confidence scores, and class labels after combining predictions.
  """
  final_boxes.clear()
  final_classes.clear()
  final_confidences.clear()
   # Unpack predictions from both models
  rtmdet_bboxes, rtmdet_confs, rtmdet_classes = predictions_rtmdet
  yolo_bboxes, yolo_confs, yolo_classes = predictions_yolo
  # Convert class labels to lists (in case they were NumPy arrays)
  rtmdet_classes = list(rtmdet_classes)
  # Check if the class lists from both models are identical and sorted
  if len(rtmdet_classes)==len(yolo_classes):
      nonsorted_yolo_classes =  yolo_classes
      nonsorted_rtmdet_classes =  rtmdet_classes
      rtmdet_classes.sort()
      yolo_classes.sort()
      if yolo_classes == rtmdet_classes:
            checkArraysSimilarity(rtmdet_bboxes=rtmdet_bboxes, rtmdet_confs=rtmdet_confs, yolo_bboxes=yolo_bboxes,
                                      yolo_confs=yolo_confs, nonsorted_rtmdet_classes=nonsorted_rtmdet_classes, 
                                      nonsorted_yolo_classes=nonsorted_yolo_classes, class_=nonsorted_yolo_classes)
      else:
            # get common element of each 
            com_yolo_class_ = [x for x in nonsorted_yolo_classes if x in nonsorted_rtmdet_classes] #y= [1,2,3,4,5], r=[1,3,4,1, 6]: output = [1,3,4]
            com_rtmdet_class_= [x for x in nonsorted_rtmdet_classes if x in nonsorted_yolo_classes]# y=[1,2,3,4,5], r=[1,3,4,1, 6]: output = [1,1,3,4]
            if com_yolo_class_ == com_rtmdet_class_:
                checkArraysSimilarity(rtmdet_bboxes=rtmdet_bboxes, rtmdet_confs=rtmdet_confs, yolo_bboxes=yolo_bboxes,
                                      yolo_confs=yolo_confs, nonsorted_rtmdet_classes=nonsorted_rtmdet_classes, 
                                      nonsorted_yolo_classes=nonsorted_yolo_classes, class_=com_rtmdet_class_)

               # for non common values checking confidence. If confident score greater than 0.8 then accept it as a correct answer 
                uniq_yolo_classes =  [x for x in nonsorted_yolo_classes if x not in nonsorted_rtmdet_classes]
                uniq_rtmdet_classes = [x for x in nonsorted_rtmdet_classes if x not in nonsorted_yolo_classes]
                confChecker(classes=uniq_rtmdet_classes, min_conf=0.8, confs=rtmdet_confs, bbox=rtmdet_bboxes)
                confChecker(classes=uniq_yolo_classes, min_conf=0.8, confs=yolo_confs, bbox=yolo_bboxes)
            else:
                diffSizeArrays(rtmdet_bboxes=rtmdet_bboxes, rtmdet_confs=rtmdet_confs, yolo_bboxes=yolo_bboxes, yolo_confs=yolo_confs, 
                               nonsorted_rtmdet_classes=nonsorted_rtmdet_classes, 
                   nonsorted_yolo_classes=nonsorted_yolo_classes, yolo_classes=yolo_classes, rtmdet_classes=rtmdet_classes)
  else:
        nonsorted_yolo_classes =  yolo_classes
        nonsorted_rtmdet_classes =  rtmdet_classes
        diffSizeArrays(rtmdet_bboxes=rtmdet_bboxes, rtmdet_confs=rtmdet_confs, yolo_bboxes=yolo_bboxes, yolo_confs=yolo_confs, 
                               nonsorted_rtmdet_classes=nonsorted_rtmdet_classes, 
                   nonsorted_yolo_classes=nonsorted_yolo_classes, yolo_classes=yolo_classes, rtmdet_classes=rtmdet_classes)
 
  return final_boxes, final_confidences, final_classes

def displayHybridPrediction(final_boxes, final_confidences, final_classes, IMAGE_PATH):
    """
    Visualizes final predictions on the image.
    Includes bounding boxes, confidence scores, and class labels.
    Saves the visualized image to a file.
    """
    image = cv2.imread(IMAGE_PATH)  # Read the image using OpenCV

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100))  # Set figure size based on image size

    # Display the image without axis
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
    ax.axis('off')  # Turn off axis
    # Iterate over the final_boxes, final_confidences, and final_classes
    for box, confidence, class_ in zip(final_boxes, final_confidences, final_classes):
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
           ax.text(x_min, y_min - 10, f'{class_}:{confidence:.2f}', color='b', fontsize=10, ha='left', va='center',bbox=dict(facecolor='white', alpha=0.7, edgecolor='white'))
        elif class_ == 2.0:
           class_ = "trash"
           # Add confidence text at the top of each bounding box
           rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
           ax.text(x_min, y_min - 10, f'{class_}:{confidence:.2f}', color='r', fontsize=10, ha='left', va='center',bbox=dict(facecolor='white', alpha=0.7, edgecolor='white'))
        # Add the patch to the plot
        ax.add_patch(rect)
    # Save the modified image with bounding boxes and confidence scores
    plt.savefig(f"hybrid_model_detection/output_image_{count}.jpg", bbox_inches='tight')
    return f"hybrid_model_detection/output_image_{count}.jpg"
count+=1
# for i in range (1,3):
#     IMAGE_PATH = f"{dataset_location}/internet_{i}.jpg"
#     image = cv2.imread(IMAGE_PATH)
#     result = inference_detector(model_rtmdet, image)
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         rtmdet_thread = executor.submit(get_predictions, model_rtmdet, IMAGE_PATH)
#         predictions_rtmdet = rtmdet_thread.result() 
#         yolo_thread = executor.submit(get_predictions, model_yolo, IMAGE_PATH)
#         predictions_yolo = yolo_thread.result()
#     # print(f"predictions_rtmdet: {predictions_rtmdet}")   
#     # print(f"predictions_yolo: {predictions_yolo}")     
#     # Placeholder variables for final predictions    
#     final_boxes = []
#     final_confidences = []
#     final_classes = []
#     final_boxes, final_confidences, final_classes = vote(predictions_rtmdet=predictions_rtmdet, predictions_yolo=predictions_yolo)
#     displayHybridPrediction(final_boxes, final_confidences, final_classes, IMAGE_PATH=IMAGE_PATH)
#     count +=1
