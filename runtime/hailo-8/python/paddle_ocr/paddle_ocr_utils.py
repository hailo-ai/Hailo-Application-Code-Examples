import cv2
import numpy as np
from db_postprocess import DBPostProcess
from symspellpy import SymSpell
import os


# Constant large character set kept in memory for decoding predictions
CHARACTERS = ['blank', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C',
              'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
              'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
              'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '!', '"', '#', '$', '%', '&',
              "'", '(', ')', '*', '+', ',', '-', '.', '/', ' ', ' ']



def ocr_eval_postprocess(infer_results):
    """
    Decodes raw model outputs into readable text with confidence scores.

    Args:
        infer_results (np.ndarray | list): Output tensor from the model (shape: BxLxC or LxC).

    Returns:
        list[tuple[str, float]]: List of (decoded_text, mean_confidence).
    """
    dict_character = list(CHARACTERS)
    ignored_tokens = [0]

    if isinstance(infer_results, list):  # Converts to numpy array
        infer_results = np.array(infer_results)  # Full array copied

    if infer_results.ndim == 2:
        infer_results = np.expand_dims(infer_results, axis=0)  # Adds batch dimension

    text_prob = infer_results.max(axis=2)  # BxL float array
    text_index = infer_results.argmax(axis=2)  # BxL int array

    batch_size = len(text_index)
    results = []

    for batch_idx in range(batch_size):
        selection = np.ones(len(text_index[batch_idx]), dtype=bool)  # Boolean mask array
        selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

        for ignored_token in ignored_tokens:
            selection &= text_index[batch_idx] != ignored_token

        char_list = [dict_character[text_id] for text_id in text_index[batch_idx][selection]]  # New list per sample

        if text_prob is not None:
            conf_list = text_prob[batch_idx][selection]
        else:
            conf_list = [1] * len(selection)

        if len(conf_list) == 0:
            conf_list = [0]

        text = "".join(char_list)
        results.append((text, np.mean(conf_list).tolist()))  # Results list grows with batch size

    return results


def resize_heatmap_to_original(heatmap: np.ndarray, original_size: tuple[int, int], model_w: int, model_h: int) -> np.ndarray:
    """
    Resize model output heatmap to match original image size, undoing aspect ratio padding.

    Args:
        heatmap (np.ndarray): Model output heatmap of shape (model_h, model_w) or (model_h, model_w, C).
        original_size (tuple): Original image size as (height, width).
        model_w (int): Model input width.
        model_h (int): Model input height.

    Returns:
        np.ndarray: Heatmap resized to original image size.
    """
    orig_h, orig_w = original_size
    scale = min(model_w / orig_w, model_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    x_offset = (model_w - new_w) // 2
    y_offset = (model_h - new_h) // 2

    # Crop the heatmap to remove padding
    cropped_heatmap = heatmap[y_offset:y_offset + new_h, x_offset:x_offset + new_w]

    # Resize back to original image size
    resized_heatmap = cv2.resize(cropped_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    return resized_heatmap



def get_cropped_text_images(heatmap, orig_img, model_height, model_width, bin_thresh=0.3):
    postprocess = DBPostProcess(
        thresh=bin_thresh,  # binary threshold
        box_thresh=0.6,
        max_candidates=1000,
        unclip_ratio=1.5
    )
    """
    Extracts cropped and rectified text regions from a detection heatmap by applying postprocessing
    (e.g., differentiable binarization and contour extraction), then cropping and masking the regions
    from the original image.

    Args:
        heatmap (np.ndarray): Raw heatmap output from the detection model (shape: [1, H, W]).
        orig_img (np.ndarray): Original input image.
        model_height (int): Height of the input to the detection model.
        model_width (int): Width of the input to the detection model.
        bin_thresh (float, optional): Threshold used to binarize the heatmap for contour extraction.
                                      Defaults to 0.3.

    Returns:
        Tuple[List[np.ndarray], List[List[int]]]:
            - List of rectified cropped image regions corresponding to detected text areas.
            - List of bounding boxes in the format [x, y, w, h] for each detected text region
              in the original image coordinate space.
    """

    heatmap_resized = resize_heatmap_to_original(
        heatmap,
        original_size=orig_img.shape[:2],
        model_w=model_width,
        model_h=model_height
    )

    # mimic batch output shape
    preds = {'maps': heatmap_resized[None, None, :, :]}  # shape: [1, 1, H, W]

    # Apply postprocess to get boxes
    boxes_batch = postprocess(preds, [(*orig_img.shape[:2], 1., 1.)])
    boxes = boxes_batch[0]['points']
    cropped_images = []
    boxes_location = []
    for box in boxes:
        try:
            box = np.array(box).astype(np.int32)
            # Bounding rect
            x, y, w, h = cv2.boundingRect(box)
            boxes_location.append([x, y, w, h])
            # Crop + mask
            cropped = orig_img[y:y + h, x:x + w].copy()
            box[:, 0] -= x
            box[:, 1] -= y

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [box], 255)
            cropped = cv2.bitwise_and(cropped, cropped, mask=mask)

            # Optionally: rectify to rectangle
            rectified = warp_to_rectangle(cropped, box)

            cropped_images.append(rectified)
        except:
            print("Error with box: ", box)

    return cropped_images, boxes_location



def warp_to_rectangle(image, poly):
    """
    Warps an arbitrary quadrilateral region into a rectangle for recognition.

    Args:
        image: Input image.
        poly: 4-point polygon

    Returns:
        Warped (rectified) image.
    """
    poly = poly.astype(np.float32)
    w = int(np.linalg.norm(poly[0] - poly[1]))
    h = int(np.linalg.norm(poly[0] - poly[3]))
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(poly, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped



def det_postprocess(infer_results, orig_img, model_height, model_width):
    """
    Applies postprocessing to the detection model output to extract text region bounding boxes
    and their corresponding cropped image regions.

    Args:
        infer_results (np.ndarray): Raw output from the detection model, expected shape (1, H, W, C).
        orig_img (np.ndarray): The original input image.
        model_height (int): Height of the model input.
        model_width (int): Width of the model input.

    Returns:
        Tuple:
            - List of cropped image regions corresponding to detected text areas.
            - List of bounding boxes for the detected regions in the original image coordinates.
    """
    heatmap = infer_results[: ,: ,0]
    return get_cropped_text_images(heatmap, orig_img, model_height, model_width)


def resize_with_padding(
        image: np.ndarray,
        target_height: int = 48,
        target_width: int = 320,
        pad_value: int = 128
) -> np.ndarray:
    """
    Resizes input to fixed size while preserving aspect ratio and padding.

    Args:
        image: Input image.
        target_height, target_width: Target dimensions.
        pad_value: Padding value (gray or custom).

    Returns:
        Resized and padded image.
    """
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    pad_top = (target_height - new_h) // 2
    pad_bottom = target_height - new_h - pad_top
    pad_left = (target_width - new_w) // 2
    pad_right = target_width - new_w - pad_left

    if image.ndim == 3:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=[pad_value] * 3)
    else:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_value)

    return padded


def inference_result_handler(original_frame, infer_results, boxes, ocr_corrector):
    """
    Handles inference results by decoding and visualizing them.

    Args:
        original_frame: Image frame
        infer_results: List of raw model outputs
        boxes: Detected bounding boxes
        ocr_corrector: Optional spell corrector

    Returns:
        Annotated image with original + OCR overlay
    """
    texts_l = []
    for f in infer_results:
        pp_res = ocr_eval_postprocess(f)[0]
        texts_l += [pp_res[0]]

    return visualize_ocr_annotations(original_frame, boxes, texts_l, ocr_corrector)


def map_bbox_to_original_image(box: list, img_w: int, img_h: int, model_w: int, model_h: int) -> list:
    """
    Convert bounding box (x, y, w, h) from model input space back to original image space.

    Args:
        box (list): [x, y, w, h] in preprocessed image (model input space).
        img_w (int): Original image width.
        img_h (int): Original image height.
        model_w (int): Model input width.
        model_h (int): Model input height.

    Returns:
        list: [x, y, w, h] in original image coordinates.
    """
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
    x_offset = (model_w - new_img_w) // 2
    y_offset = (model_h - new_img_h) // 2

    x, y, w, h = box
    x = (x - x_offset) / scale
    y = (y - y_offset) / scale
    w = w / scale
    h = h / scale

    return [int(x), int(y), int(w), int(h)]


def draw_label(
    image,
    box,
    label,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    text_color=(0, 0, 255),
    thickness=1,
    padding=4,
    min_font_scale=0.3,
    align="left",
):
    """
    Draws a label inside a bounding box with automatic scaling.

    Args:
        image: Image to draw on
        box: Bounding box [x, y, w, h]
        label: String text
        align: 'left', 'center', or 'right'
    """
    label = label.strip()
    if not label:
        return

    x, y, w, h = box
    inner_x, inner_y = x + padding, y + padding
    inner_w, inner_h = w - 2 * padding, h - 2 * padding

    # Step 1: Shrink font scale to fit width and height
    font_scale = 1.0
    while font_scale >= min_font_scale:
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        if text_w <= inner_w and text_h <= inner_h:
            break
        font_scale -= 0.05

    # Step 2: Draw label with no artificial spacing
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

    if align == "left":
        text_x = inner_x
    elif align == "center":
        text_x = inner_x + max((inner_w - text_w) // 2, 0)
    elif align == "right":
        text_x = inner_x + max(inner_w - text_w, 0)
    else:
        raise ValueError(f"Unsupported alignment: {align}")

    text_y = inner_y + (inner_h + text_h) // 2
    cv2.putText(image, label, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def visualize_ocr_annotations(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str],
    ocr_corrector,
    text_color=(0, 0, 255),
    padding=6,
    align="left"
) -> np.ndarray:
    """
    Draws OCR results over a copy of the image.

    Args:
        image: Input image
        boxes: Bounding boxes
        labels: OCR strings
        ocr_corrector: Optional spell corrector
    """
    left = image.copy()
    right = image.copy()

    # First pass: draw white rectangles
    for (x, y, w, h), text in zip(boxes, labels):
        if not text.strip():
            continue
        cv2.rectangle(right, (x, y), (x + w, y + h), (255, 255, 255), -1)

    # Second pass: draw labels
    for (x, y, w, h), text in zip(boxes, labels):
        if not text.strip():
            continue
        if ocr_corrector:
            text = ocr_corrector.correct_text(text)

        draw_label(
            right,
            (x, y, w, h),
            text,
            text_color=text_color,
            padding=padding,
            align=align,
        )

    return np.hstack([left, right])



class OcrCorrector:
    """
    Spell-corrects OCR text using SymSpell dictionary.

    Args:
        dictionary_path: Path to .txt frequency dictionary file.
    """

    #Dictionary brought in via:
    #wget https://raw.githubusercontent.com/wolfgarbe/SymSpell/refs/heads/master/SymSpell/frequency_dictionary_en_82_765.txt

    def __init__(self, dictionary_path: str = 'frequency_dictionary_en_82_765.txt', max_edit_distance=2):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        if not os.path.exists(dictionary_path):
            raise FileNotFoundError(f"Missing dictionary file: {dictionary_path}")
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    def correct_text(self, text: str) -> str:
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        if suggestions:
            return suggestions[0].term
        return text
