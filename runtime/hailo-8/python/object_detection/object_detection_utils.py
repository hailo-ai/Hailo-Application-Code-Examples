import cv2
import numpy as np


def generate_color(class_id: int) -> tuple:
    """
    Generate a unique color for a given class ID.

    Args:
        class_id (int): The class ID to generate a color for.

    Returns:
        tuple: A tuple representing an RGB color.
    """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())


class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (114, 114, 114), label_font: str = "LiberationSans-Regular.ttf"):
        """
        Initialize the ObjectDetectionUtils class.

        Args:
            labels_path (str): Path to the labels file.
            padding_color (tuple): RGB color for padding. Defaults to (114, 114, 114).
            label_font (str): Path to the font used for labeling. Defaults to "LiberationSans-Regular.ttf".
        """
        self.labels = self.get_labels(labels_path)
        self.padding_color = padding_color
        self.label_font = label_font
    
    def get_labels(self, labels_path: str) -> list:
        """
        Load labels from a file.

        Args:
            labels_path (str): Path to the labels file.

        Returns:
            list: List of class names.
        """
        with open(labels_path, 'r', encoding="utf-8") as f:
            class_names = f.read().splitlines()
        return class_names

    def preprocess(self, image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
        """
        Resize image with unchanged aspect ratio using padding.

        Args:
            image (np.ndarray): Input image.
            model_w (int): Model input width.
            model_h (int): Model input height.

        Returns:
            np.ndarray: Preprocessed and padded image.
        """
        img_h, img_w, _ = image.shape[:3]
        scale = min(model_w / img_w, model_h / img_h)
        new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
        image = cv2.resize(image, (new_img_w, new_img_h), interpolation=cv2.INTER_CUBIC)

        padded_image = np.full((model_h, model_w, 3), self.padding_color, dtype=np.uint8)
        x_offset = (model_w - new_img_w) // 2
        y_offset = (model_h - new_img_h) // 2
        padded_image[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = image
        return padded_image

    def draw_detection(self, image: np.ndarray, box: list, cls: int, score: float, color: tuple, scale_factor: float):
        """
        Draw box and label for one detection.

        Args:
            image (np.ndarray): Image to draw on.
            box (list): Bounding box coordinates.
            cls (int): Class index.
            score (float): Detection score.
            color (tuple): Color for the bounding box.
            scale_factor (float): Scale factor for coordinates.
        """
        label = f"{self.labels[cls]}: {score:.2f}%"
        ymin, xmin, ymax, xmax = box
        ymin, xmin, ymax, xmax = int(ymin * scale_factor), int(xmin * scale_factor), int(ymax * scale_factor), int(xmax * scale_factor)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, label, (xmin + 4, ymin + 20), font, 0.5, color, 1, cv2.LINE_AA)

    def denormalize_and_rm_pad(self, box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
        """
        Denormalize bounding box coordinates and remove padding.

        Args:
            box (list): Normalized bounding box coordinates.
            size (int): Size to scale the coordinates.
            padding_length (int): Length of padding to remove.
            input_height (int): Height of the input image.
            input_width (int): Width of the input image.

        Returns:
            list: Denormalized bounding box coordinates with padding removed.
        """
        for i, x in enumerate(box):
            box[i] = int(x * size)
            if (input_width != size) and (i % 2 != 0):
                box[i] -= padding_length
            if (input_height != size) and (i % 2 == 0):
                box[i] -= padding_length

        return box

    def draw_detections(self, detections: dict, image: np.ndarray, min_score: float = 0.45, scale_factor: float = 1):
        """
        Draw detections on the image.

        Args:
            detections (dict): Detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
            image (np.ndarray): Image to draw on.
            min_score (float): Minimum score threshold. Defaults to 0.45.
            scale_factor (float): Scale factor for coordinates. Defaults to 1.

        Returns:
            np.ndarray: Image with detections drawn.
        """
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']

        # Values used for scaling coords and removing padding
        img_height, img_width = image.shape[:2]
        size = max(img_height, img_width)
        padding_length = int(abs(img_height - img_width) / 2)

        for idx in range(detections['num_detections']):
            if scores[idx] >= min_score:
                color = generate_color(classes[idx])
                scaled_box = self.denormalize_and_rm_pad(boxes[idx], size, padding_length, img_height, img_width)
                self.draw_detection(image, scaled_box, classes[idx], scores[idx] * 100.0, color, scale_factor)

        return image
        
    def extract_detections(self, input_data: list, threshold: float = 0.5) -> dict:
        """
        Extract detections from the input data.

        Args:
            input_data (list): Raw detections from the model.
            threshold (float): Score threshold for filtering detections. Defaults to 0.5.

        Returns:
            dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
        """
        boxes, scores, classes = [], [], []
        num_detections = 0

        for i, detection in enumerate(input_data):
            if len(detection) == 0:
                continue

            for det in detection:
                bbox, score = det[:4], det[4]

                if score >= threshold:
                    boxes.append(bbox)
                    scores.append(score)
                    classes.append(i)
                    num_detections += 1

        return {
            'detection_boxes': boxes, 
            'detection_classes': classes, 
            'detection_scores': scores,
            'num_detections': num_detections
        }
