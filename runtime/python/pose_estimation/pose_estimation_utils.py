from pathlib import Path
from multiprocessing import Process
import numpy as np
import cv2
from PIL import Image
from hailo_platform import HEF
from loguru import logger

# Joint pairs used for drawing pose estimations
JOINT_PAIRS = [
    [0, 1], [1, 3], [0, 2], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [12, 14], [13, 15], [14, 16]
]

class PoseEstPostProcessing:
    def __init__(self, max_detections: int, score_threshold: float, nms_iou_thresh: float,
                 regression_length: int, strides: list[int]):
        """
        Initialize the post-processing configuration.

        Args:
            max_detections (int): Maximum number of detections per class.
            score_threshold (float): Confidence threshold for filtering.
            nms_iou_thresh (float): IoU threshold for NMS.
            regression_length (int): Maximum regression value for bounding boxes.
            strides (list[int]): Stride values for each prediction scale.
        """
        self.max_detections = max_detections
        self.score_threshold = score_threshold
        self.nms_iou_thresh = nms_iou_thresh
        self.regression_length = regression_length
        self.strides = strides

    def postprocess_and_visualize(
        self, image: Image.Image, raw_detections: dict, output_path: Path,
        image_index: int, height: int, width: int, class_num: int
    ) -> None:
        """
        Post-process the inference results and save the output image.

        Args:
            image (Image.Image): The input image.
            raw_detections (Dict): Raw detections from the inference.
            output_path (Path): Path to the output directory.
            image_index (int): Index of the image for naming the output file.
            height (int): The height of the input image.
            width (int): The width of the input image.
            class_num (int): Number of classes.  

        Returns:
            None
        """
        # Post-process results
        results = self.post_process(raw_detections, height, width, class_num)

        # Visualize and save results
        output_image = self.visualize_pose_estimation_result(results, image)
        output_image_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        output_image_pil.save(output_path / f'output_image{image_index}.jpg', 'JPEG')


    def post_process(self, raw_detections: dict, height: int, width: int, class_num: int) -> dict:
        """
        Process raw detections into a structured format for pose estimation.

        Args:
            raw_detections (Dict): Raw detections from the model.
            height (int): The height of the input image.
            width (int): The width of the input image.
            class_num (int): Number of classes.

        Returns:
            Dict: Processed predictions dictionary.
        """
        raw_detections_keys = list(raw_detections.keys())
        layer_from_shape = {raw_detections[key].shape: key for key in raw_detections_keys}
        detection_output_channels = (self.regression_length + 1) * 4  # (regression length + 1) * num_coordinates
        keypoints = 51
        endnodes = [
            raw_detections[layer_from_shape[1, 20, 20, detection_output_channels]],
            raw_detections[layer_from_shape[1, 20, 20, class_num]],
            raw_detections[layer_from_shape[1, 20, 20, keypoints]],
            raw_detections[layer_from_shape[1, 40, 40, detection_output_channels]],
            raw_detections[layer_from_shape[1, 40, 40, class_num]],
            raw_detections[layer_from_shape[1, 40, 40, keypoints]],
            raw_detections[layer_from_shape[1, 80, 80, detection_output_channels]],
            raw_detections[layer_from_shape[1, 80, 80, class_num]],
            raw_detections[layer_from_shape[1, 80, 80, keypoints]]
        ]
       
        predictions_dict = self.extract_pose_estimation_results(endnodes, height, width, class_num)
        return predictions_dict
    
    def extract_pose_estimation_results(
        self, endnodes: list[np.ndarray], height: int, width: int, class_num: int
    ) -> dict[str, np.ndarray]:
        """
        Post-process the pose estimation results.

        Args:
            endnodes (list[np.ndarray]): list of 10 tensors from the model output.
            height (int): Height of the input image.
            width (int): Width of the input image.
            class_num (int): Number of classes.
    
        Returns:
            dict: Processed detections with keys:
                'bboxes': numpy.ndarray with shape (batch_size, max_detections, 4),
                'keypoints': numpy.ndarray with shape (batch_size, max_detections, 17, 2),
                'joint_scores': numpy.ndarray with shape (batch_size, max_detections, 17, 1),
                'scores': numpy.ndarray with shape (batch_size, max_detections, 1).
        """
        batch_size = endnodes[0].shape[0]
        strides = self.strides[::-1]
        image_dims = (height, width)
       
        raw_boxes = endnodes[:7:3]
        scores = [
            np.reshape(s, (-1, s.shape[1] * s.shape[2], class_num)) for s in endnodes[1:8:3]
        ]
        scores = np.concatenate(scores, axis=1)

        kpts = [
            np.reshape(c, (-1, c.shape[1] * c.shape[2], 17, 3)) for c in endnodes[2:9:3]
        ]

        decoded_boxes, decoded_kpts = self.decoder(raw_boxes,
                                              kpts, strides,
                                              image_dims, self.regression_length)
        decoded_kpts = np.reshape(decoded_kpts, (batch_size, -1, 51))
        predictions = np.concatenate([decoded_boxes, scores, decoded_kpts], axis=2)

        nms_res = self.non_max_suppression(
            predictions, conf_thres=self.score_threshold, 
            iou_thres=self.nms_iou_thresh, max_det=self.max_detections
        )

        output = {
            'bboxes': np.zeros((batch_size, self.max_detections, 4)),
            'keypoints': np.zeros((batch_size, self.max_detections, 17, 2)),
            'joint_scores': np.zeros((batch_size, self.max_detections, 17, 1)),
            'scores': np.zeros((batch_size, self.max_detections, 1))
        }

        for b in range(batch_size):
            output['bboxes'][b, :nms_res[b]['num_detections']] = nms_res[b]['bboxes']
            output['keypoints'][b, :nms_res[b]['num_detections']] = nms_res[b]['keypoints'][..., :2]
            output['joint_scores'][b, :nms_res[b]['num_detections'],
                                   ..., 0] = self._sigmoid(nms_res[b]['keypoints'][..., 2])
            output['scores'][b, :nms_res[b]['num_detections'], ..., 0] = nms_res[b]['scores']

        return output


    def visualize_pose_estimation_result(
        self, results: dict, img: Image.Image, *, detection_threshold: float = 0.5,
        joint_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Visualize pose estimation results on an image.

        Args:
            results (Dict): The processed pose estimation results.
            img (Image.Image): The input image on which to draw the results.
            detection_threshold (float): Threshold for detecting bounding boxes.
            joint_threshold (float): Threshold for detecting joints.

        Returns:
            np.ndarray: Image with visualized pose estimations.
        """
        if 'predictions' in results:
            results = results['predictions']
            bboxes, scores, keypoints, joint_scores = results
        else:
            bboxes, scores, keypoints, joint_scores = (
                results['bboxes'], results['scores'], results['keypoints'], results['joint_scores']
            )

        batch_size = bboxes.shape[0]
        assert batch_size == 1

        box, score, keypoint, keypoint_score = bboxes[0], scores[0], keypoints[0], joint_scores[0]
        image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            
        for (detection_box, detection_score, detection_keypoints,
            detection_keypoints_score) in zip(box, score, keypoint, keypoint_score):
            if detection_score < detection_threshold:
                continue
            xmin, ymin, xmax, ymax = [int(x) for x in detection_box]

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
            cv2.putText(image, str(detection_score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
            
            joint_visible = detection_keypoints_score > joint_threshold
            detection_keypoints = detection_keypoints.reshape(17, 2)
            
            for joint, joint_score in zip(detection_keypoints, detection_keypoints_score):
                if joint_score < joint_threshold:
                    continue
                cv2.circle(image, (int(joint[0]), int(joint[1])), 1, (255, 0, 255), -1)

            for joint0, joint1 in JOINT_PAIRS:
                if joint_visible[joint0] and joint_visible[joint1]:
                    pt1 = (int(detection_keypoints[joint0][0]), int(detection_keypoints[joint0][1]))
                    pt2 = (int(detection_keypoints[joint1][0]), int(detection_keypoints[joint1][1]))
                    cv2.line(image, pt1, pt2, (255, 0, 255), 3)

        return image


    def preprocess(self, image: Image.Image, model_w: int, model_h: int) -> Image.Image:
        """
        Resize image with unchanged aspect ratio using padding.

        Args:
            image (PIL.Image.Image): Input image.
            model_w (int): Model input width.
            model_h (int): Model input height.

        Returns:
            PIL.Image.Image: Preprocessed and padded image.
        """
        img_w, img_h = image.size
        scale = min(model_w / img_w, model_h / img_h)
        new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
        image = image.resize((new_img_w, new_img_h), Image.Resampling.BICUBIC)
        padding_color = (114, 114, 114)
        padded_image = Image.new('RGB', (model_w, model_h), padding_color)
        padded_image.paste(image, ((model_w - new_img_w) // 2, (model_h - new_img_h) // 2))
        return padded_image


    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid transformed array.
        """
        return 1 / (1 + np.exp(-x))


    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Softmax transformed array.
        """
        return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)


    def max_value(self, a: float, b: float) -> float:
        """
        Return the maximum of two values.

        Args:
            a (float): First value.
            b (float): Second value.

        Returns:
            float: The maximum of `a` and `b`.
        """
        return a if a >= b else b


    def min_value(self, a: float, b: float) -> float:
        """
        Return the minimum of two values.

        Args:
            a (float): First value.
            b (float): Second value.

        Returns:
            float: The minimum of `a` and `b`.
        """
        return a if a <= b else b


    def nms(self, dets: np.ndarray, thresh: float) -> np.ndarray:
        """
        Perform Non-Maximum Suppression (NMS) on detection boxes.

        Args:
            dets (np.ndarray): Detection boxes and scores array.
            thresh (float): Overlap threshold for suppression.

        Returns:
            np.ndarray: Indices of the boxes to keep.
        """
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]

        suppressed = np.zeros(dets.shape[0], dtype=int)
        for i in range(len(order)):
            idx_i = order[i]
            if suppressed[idx_i] == 1:
                continue
            for j in range(i + 1, len(order)):
                idx_j = order[j]
                if suppressed[idx_j] == 1:
                    continue

                xx1 = self.max_value(x1[idx_i], x1[idx_j])
                yy1 = self.max_value(y1[idx_i], y1[idx_j])
                xx2 = self.min_value(x2[idx_i], x2[idx_j])
                yy2 = self.min_value(y2[idx_i], y2[idx_j])
                w = self.max_value(0.0, xx2 - xx1 + 1)
                h = self.max_value(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[idx_i] + areas[idx_j] - inter)

                if ovr >= thresh:
                    suppressed[idx_j] = 1

        return np.where(suppressed == 0)[0]


    def decoder(
        self, raw_boxes: np.ndarray, raw_kpts: np.ndarray, strides: list[int],
        image_dims: tuple[int, int], reg_max: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode the bounding boxes and keypoints from raw predictions.

        Args:
            raw_boxes (np.ndarray): Raw bounding box predictions.
            raw_kpts (np.ndarray): Raw keypoint predictions.
            strides (list[int]): Stride values for each prediction scale.
            image_dims (tuple[int, int]): Dimensions of the input image.
            reg_max (int): Maximum regression value for bounding boxes.

        Returns:
            tuple[np.ndarray, np.ndarray]: Decoded bounding boxes and keypoints.
        """
        boxes = None
        decoded_kpts = None

        for box_distribute, kpts, stride, _ in zip(raw_boxes, raw_kpts, strides, np.arange(3)):
            shape = [int(x / stride) for x in image_dims]
            grid_x = np.arange(shape[1]) + 0.5
            grid_y = np.arange(shape[0]) + 0.5
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            ct_row = grid_y.flatten() * stride
            ct_col = grid_x.flatten() * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            reg_range = np.arange(reg_max + 1)
            box_distribute = np.reshape(box_distribute,
                                        (-1,
                                        box_distribute.shape[1] * box_distribute.shape[2],
                                        4,
                                        reg_max + 1))
            box_distance = self._softmax(box_distribute) * np.reshape(reg_range, (1, 1, 1, -1))
            box_distance = np.sum(box_distance, axis=-1) * stride

            box_distance = np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]],
                                        axis=-1)
            decode_box = np.expand_dims(center, axis=0) + box_distance

            xmin, ymin, xmax, ymax = decode_box[:, :, 0], decode_box[:, :, 1], decode_box[:, :, 2], decode_box[:, :, 3]
            decode_box = np.transpose([xmin, ymin, xmax, ymax], [1, 2, 0])

            xywh_box = np.transpose([(xmin + xmax) / 2,
                                    (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
            boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)

            kpts[..., :2] *= 2
            kpts[..., :2] = stride * (kpts[..., :2] - 0.5) + np.expand_dims(center[..., :2], axis=1)
            decoded_kpts = kpts if decoded_kpts is None else np.concatenate([decoded_kpts, kpts],
                                                                            axis=1)

        return boxes, decoded_kpts


    def xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes from (x, y, w, h) to (xmin, ymin, xmax, ymax) format.

        Args:
            x (np.ndarray): Bounding boxes in (x, y, w, h) format.

        Returns:
            np.ndarray: Bounding boxes in (xmin, ymin, xmax, ymax) format.
        """
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y


    def non_max_suppression(
        self, prediction: np.ndarray, conf_thres: float = 0.1, iou_thres: float = 0.45,
        max_det: int = 100, n_kpts: int = 17
    ) -> list[dict]:
        """
        Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

        Args:
            prediction (np.ndarray): Inference results with shape (batch_size, num_proposals, 56).
            conf_thres (float): Confidence threshold for filtering.
            iou_thres (float): Intersection Over Union (IoU) threshold for NMS.
            max_det (int): Maximum number of detections to retain.
            n_kpts (int): Number of keypoints.

        Returns:
            list[dict]: list of dictionaries for each image containing detection results.
        """
        assert 0 <= conf_thres <= 1, f'Invalid confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0'

        nc = prediction.shape[2] - n_kpts * 3 - 4
        xc = prediction[..., 4] > conf_thres
        ki = 4 + nc
        output = []

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]

            if not x.shape[0]:
                output.append({
                    'bboxes': np.zeros((0, 4)),
                    'keypoints': np.zeros((0, n_kpts, 3)),
                    'scores': np.zeros((0)),
                    'num_detections': 0
                })
                continue

            boxes = self.xywh2xyxy(x[:, :4])
            kpts = x[:, ki:]

            conf = np.expand_dims(x[:, 4:ki].max(1), 1)
            j = np.expand_dims(x[:, 4:ki].argmax(1), 1).astype(np.float32)

            keep = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, kpts), 1)[keep]
            x = x[x[:, 4].argsort()[::-1][:max_det]]

            if not x.shape[0]:
                output.append({
                    'bboxes': np.zeros((0, 4)),
                    'keypoints': np.zeros((0, n_kpts, 3)),
                    'scores': np.zeros((0)),
                    'num_detections': 0
                })
                continue

            boxes = x[:, :4]
            scores = x[:, 4]
            kpts = x[:, 6:].reshape(-1, n_kpts, 3)

            i = self.nms(np.concatenate((boxes, np.expand_dims(scores, 1)), axis=1), iou_thres)
            output.append({
                'bboxes': boxes[i],
                'keypoints': kpts[i],
                'scores': scores[i],
                'num_detections': len(i)
            })

        return output
   
def check_process_errors(*processes: Process) -> None:
    """
    Check the exit codes of processes and log errors if any process has a non-zero exit code.

    Args:
        processes (Process): The processes to check.
    """
    process_failed = False
    for process in processes:
        if process.exitcode != 0:
            logger.error(f"{process.name} terminated with an error. Exit code: {process.exitcode}")
            process_failed = True
    if process_failed:
        raise RuntimeError("One or more processes terminated with an error.")

def output_data_type2dict(hef: HEF, data_type: str) -> dict:
    """
    initiates a dictionary where the keys are layers names and 
    all values are the same requested data type.
    
    Args:
        hef(HEF) : the HEF model file.
        data_type(str) : the requested data type (e.g 'FLOAT32', 'UINT8', or 'UINT16')
    
    Returns:
        Dict: layer name to data type 
    """
    data_type_dict = {info.name: data_type for info in hef.get_output_vstream_infos()}

    return data_type_dict

# End-of-file (EOF)
