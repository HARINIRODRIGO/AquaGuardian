import warnings
import numpy as np

class BoxFilter:
    def __init__(self, threshold=0.0, min_score=None, min_area=0, class_specific_thresholds=None):
        self.threshold = threshold
        self.min_score = min_score if min_score is not None else threshold
        self.min_area = min_area
        self.class_specific_thresholds = class_specific_thresholds or {}

    def _validate_lengths(self, bounding_boxes_data, confidence_scores, classes):
        if len(bounding_boxes_data) != len(confidence_scores) or len(bounding_boxes_data) != len(classes):
            raise ValueError("Lengths of bounding_boxes_data, confidence_scores, and classes arrays must be equal.")

    def _normalize_coords(self, x1, y1, x2, y2):
        return max(0, min(x1, 640)), max(0, min(y1, 640)), max(0, min(x2, 640)), max(0, min(y2, 640))

    def _check_box_area(self, x1, y1, x2, y2):
        return (x2 - x1) * (y2 - y1) >= self.min_area  # Consider area greater than or equal to min_area

    # Extracting data from both model data contain list (bounding box,confidence and class data)
    def filter_bounding_boxes_data(self, bounding_boxes_data, confidence_scores, classes):
        self._validate_lengths(bounding_boxes_data, confidence_scores, classes)
        filtered_boxes = dict()

        for i, (bounding_box_set, confidence_score_set, class_set) in enumerate(zip(bounding_boxes_data, confidence_scores, classes)): # Combining all 3 lists and iterating
            for confidence_score, _class, bounding_box_part in zip(confidence_score_set, class_set, bounding_box_set):
                if confidence_score < max(self.min_score, self.class_specific_thresholds.get(_class, 0.0)):
                    continue
                
                # Normailzing the code
                x1, y1, x2, y2 = self._normalize_coords(*map(float, bounding_box_part))
                if x2 < x1:
                    warnings.warn('X2 < X1 value in box. Swap them.')
                    x1, x2 = x2, x1
                if y2 < y1:
                    warnings.warn('Y2 < Y1 value in box. Swap them.')
                    y1, y2 = y2, y1

                if self._check_box_area(x1, y1, x2, y2):
                    b = [int(_class), float(confidence_score), 1.0, i, x1, y1, x2, y2]
                    if _class not in filtered_boxes:
                        filtered_boxes[_class] = []  # Initialize with an empty list
                        filtered_boxes[_class].append(b)  # Append to the class list

                # Sort the class list by score (descending order) after processing all boxes in the class
                if _class in filtered_boxes:
                    filtered_boxes[_class] = np.array(filtered_boxes[_class])  # Convert to NumPy array
                    filtered_boxes[_class] = filtered_boxes[_class][filtered_boxes[_class][:, 1].argsort()[::-1]]  # Sort confidence scores 
                 
                else:
                    warnings.warn("Zero area box skipped: {}.".format(bounding_box_part))

        return filtered_boxes 


class WeightedBoxFusion:
    def __init__(self, iou_threshold=0.8, confidence_type='avg'):
        self.iou_threshold = iou_threshold 
        self.confidence_type = confidence_type

    def _validate_confidence_type(self):
        """
        Confidence Type:
            avg = Average confidence score across all contributing bounding boxes
            max = Maximum confidence score among all contributing bounding boxes as final confidence score.
            box_and_model_avg = Average confidence score based on both the number of boxes and models
            absent_model_aware_avg = Calculate confidence score based on the absence of models for certain classes
        """
        if self.confidence_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
            raise ValueError("Unknown confidence_type: {}. Must be 'avg', 'max', 'box_and_model_avg', or 'absent_model_aware_avg'.".format(self.confidence_type))

    def _calculate_weighted_box(self, bounding_boxes_data):
        """
        Create weighted box for a set of bounding boxes based on the specified confidence type.

        Args:
            bounding_boxes_data (list): List of boxes.

        Returns:
            numpy.ndarray: Weighted box containing class, score, weight, model index, x1, y1, x2, y2.
        """
        bounding_box = np.zeros(8, dtype=np.float32)
        confidence = 0
        confidence_list = []
        weight = 0
        for b in bounding_boxes_data:
            bounding_box[4:] += (b[1] * b[4:])
            confidence += b[1]
            confidence_list.append(b[1])
            weight += b[2]
        bounding_box[0] = bounding_boxes_data[0][0]
        if self.confidence_type in ('avg', 'box_and_model_avg', 'absent_model_aware_avg'):
            bounding_box[1] = confidence / len(bounding_boxes_data)
        elif self.confidence_type == 'max':
            bounding_box[1] = np.array(confidence_list).max()
        bounding_box[2] = weight
        bounding_box[3] = -1
        bounding_box[4:] /= confidence
        return bounding_box


    def find_diff_size_array(self, bounding_boxes_data_list, new_bounding_box, match_iou):
        """
        Reimplementation of find_matching_box with numpy instead of loops.
        Gives significant speed up for larger arrays (~100x).

        Args:
            bounding_boxes_data_list (numpy.ndarray): Array of bounding boxes.
            new_box (numpy.ndarray): New bounding box to match against.
            matching_iou (float): Threshold for IOU matching.

        Returns:
            tuple: Index of matching box and corresponding IOU.
        """
        def calculate_iou(bounding_boxes_data, new_bounding_box):
            xA = np.maximum(bounding_boxes_data[:, 0], new_bounding_box[0]) # Getting first two elements of each lists and append
            yA = np.maximum(bounding_boxes_data[:, 1], new_bounding_box[1])
            xB = np.minimum(bounding_boxes_data[:, 2], new_bounding_box[2])
            yB = np.minimum(bounding_boxes_data[:, 3], new_bounding_box[3])
           
            interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

            boxAArea = (bounding_boxes_data[:, 2] - bounding_boxes_data[:, 0]) * (bounding_boxes_data[:, 3] - bounding_boxes_data[:, 1])
            boxBArea = (new_bounding_box[2] - new_bounding_box[0]) * (new_bounding_box[3] - new_bounding_box[1])

            iou = interArea / (boxAArea + boxBArea - interArea)
        
            return iou

        if bounding_boxes_data_list.shape[0] == 0:
            return -1, match_iou

        ious = calculate_iou(bounding_boxes_data_list[:, 4:], new_bounding_box[4:])
        ious[bounding_boxes_data_list[:, 0] != new_bounding_box[0]] = -1

        best_idx = np.argmax(ious)
        best_iou = ious[best_idx]

        if best_iou <= match_iou:
            best_iou = match_iou
            best_idx = -1

        return best_idx, best_iou

    def fuse_weighted_bounding_boxes_data(self, bounding_boxes_data_list, confidence_scores_list, classes_list, weights=None):
        if weights is None:
            weights = np.ones(len(bounding_boxes_data_list)) # Equal weights for each model
        if len(weights) != len(bounding_boxes_data_list):
            warnings.warn('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(bounding_boxes_data_list)))
            weights = np.ones(len(bounding_boxes_data_list))
        weights = np.array(weights) # Convert NumPy array
        self._validate_confidence_type()

        filtered_bounding_boxes_data = BoxFilter().filter_bounding_boxes_data(bounding_boxes_data_list, confidence_scores_list, classes_list)
        if len(filtered_bounding_boxes_data) == 0:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

        overall_bounding_boxes_data = []

        # Getting each object data out from the  dictionary
        for label in filtered_bounding_boxes_data:
            bounding_boxes_data = filtered_bounding_boxes_data[label]
            new_bounding_boxes_data = []
            weighted_bounding_boxes_data = np.empty((0, 8)) # 0 rows and 8 columns array

            for box in bounding_boxes_data:
                index, best_iou = self.find_diff_size_array(weighted_bounding_boxes_data, box, self.iou_threshold)
                if index != -1:
                    new_bounding_boxes_data[index].append(box)
                    weighted_bounding_boxes_data[index] = self._calculate_weighted_box(new_bounding_boxes_data[index])
                else:
                    new_bounding_boxes_data.append(box)
                    weighted_bounding_boxes_data = np.vstack((weighted_bounding_boxes_data, box))

            for i in range(len(new_bounding_boxes_data)):
                clustered_bounding_boxes_data = new_bounding_boxes_data[i]
                if self.confidence_type == 'box_and_model_avg':
                    clustered_bounding_boxes_data = np.array(clustered_bounding_boxes_data)
                    weighted_bounding_boxes_data[i, 1] = weighted_bounding_boxes_data[i, 1] * len(clustered_bounding_boxes_data) / weighted_bounding_boxes_data[i, 2]
                    _, idx = np.unique(clustered_bounding_boxes_data[:, 3], return_index=True)
                    weighted_bounding_boxes_data[i, 1] = weighted_bounding_boxes_data[i, 1] *  clustered_bounding_boxes_data[idx, 2].sum() / weights.sum()
                elif self.confidence_type == 'absent_model_aware_avg':
                    clustered_bounding_boxes_data = np.array(clustered_bounding_boxes_data)
                    models = np.unique(clustered_bounding_boxes_data[:, 3]).astype(int)
                    mask = np.ones(len(weights), dtype=bool)
                    mask[models] = False
                    weighted_bounding_boxes_data[i, 1] = weighted_bounding_boxes_data[i, 1] * len(clustered_bounding_boxes_data) / (weighted_bounding_boxes_data[i, 2] + weights[mask].sum())
                elif self.confidence_type == 'max':
                    weighted_bounding_boxes_data[i, 1] = weighted_bounding_boxes_data[i, 1] / weights.max()
                else:
                    weighted_bounding_boxes_data[i, 1] = weighted_bounding_boxes_data[i, 1] * len(clustered_bounding_boxes_data) / weights.sum()

            overall_bounding_boxes_data.append(weighted_bounding_boxes_data)
        overall_bounding_boxes_data = np.concatenate(overall_bounding_boxes_data, axis=0)
        overall_bounding_boxes_data = overall_bounding_boxes_data[overall_bounding_boxes_data[:, 1].argsort()[::-1]]
        bounding_boxes_data = overall_bounding_boxes_data[:, 4:]
        confidence_scores = overall_bounding_boxes_data[:, 1]
        classes = overall_bounding_boxes_data[:, 0]
        return bounding_boxes_data, confidence_scores, classes