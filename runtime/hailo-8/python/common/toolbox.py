from typing import List, Generator, Optional, Tuple, Dict, Callable ,Any
from pathlib import Path
from loguru import logger
import json
import os
import sys
import numpy as np
import queue
import cv2


IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')
CAMERA_RESOLUTION_MAP = {
    "sd": (640, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080)
}
CAMERA_INDEX = 0 # or 1, or 2 — depending on your setup


def load_json_file(path: str) -> Dict[str, Any]:
    """
    Loads and parses a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed contents of the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        OSError: If the file cannot be read.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format in file '{path}': {e.msg}", e.doc, e.pos)

    return data


def is_valid_camera_index(index):
    """
    Check if a camera index is available and can be opened.

    Args:
        index (int): Camera index to test.

    Returns:
        bool: True if the camera can be opened, else False.
    """
    cap = cv2.VideoCapture(index)
    valid = cap.isOpened()
    cap.release()
    return valid


def list_available_cameras(max_index=5):
    """
    List all available camera indices up to a maximum index.

    Args:
        max_index (int): Highest camera index to test.

    Returns:
        list[int]: List of available camera indices.
    """
    available = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def init_input_source(input_path, batch_size, resolution):
    """
    Initialize input source from camera, video file, or image directory.

    Args:
        input_path (str): "camera", video file path, or image directory.
        batch_size (int): Number of images to validate against.
        resolution (str or None): One of ['sd', 'hd', 'fhd'], or None to use native camera resolution.

    Returns:
        Tuple[Optional[cv2.VideoCapture], Optional[List[np.ndarray]]]
    """
    cap = None
    images = None

    def get_camera_native_resolution():
        cap = cv2.VideoCapture(CAMERA_INDEX)
        res = (int(cap.get(3)), int(cap.get(4))) if cap.isOpened() else (640, 480)
        cap.release()
        return res


    if input_path == "camera":

        if not is_valid_camera_index(CAMERA_INDEX):
            logger.error(f"CAMERA_INDEX {CAMERA_INDEX} not found.")
            available = list_available_cameras()
            logger.warning(f"Available camera indices: {available}")
            exit(1)

        if not resolution:
            CAMERA_CAP_WIDTH, CAMERA_CAP_HEIGHT = get_camera_native_resolution()
        else:
            CAMERA_CAP_WIDTH, CAMERA_CAP_HEIGHT = CAMERA_RESOLUTION_MAP.get(resolution, (640, 480))  # fallback to SD

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAP_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

    elif any(input_path.lower().endswith(suffix) for suffix in ['.mp4', '.avi', '.mov', '.mkv']):
        if not os.path.exists(input_path):
            logger.error(f"File not found: {input_path}")
            sys.exit(1)
        cap = cv2.VideoCapture(input_path)
    else:
        images = load_images_opencv(input_path)
        try:
            validate_images(images, batch_size)
        except ValueError as e:
            logger.error(e)
            sys.exit(1)

    return cap, images


def load_images_opencv(images_path: str) -> List[np.ndarray]:
    """
    Load images from the specified path.

    Args:
        images_path (str): Path to the input image or directory of images.

    Returns:
        List[np.ndarray]: List of images as NumPy arrays.
    """
    import cv2
    path = Path(images_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [cv2.imread(str(path))]
    elif path.is_dir():
        return [
            cv2.imread(str(img)) for img in path.glob("*")
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []

def load_input_images(images_path: str):
    """
    Load images from the specified path.

    Args:
        images_path (str): Path to the input image or directory of images.

    Returns:
        List[Image.Image]: List of PIL.Image.Image objects.
    """
    from PIL import Image
    path = Path(images_path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [Image.open(path)]
    elif path.is_dir():
        return [
            Image.open(img) for img in path.glob("*")
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return []

def validate_images(images: List[np.ndarray], batch_size: int) -> None:
    """
    Validate that images exist and are properly divisible by the batch size.

    Args:
        images (List[np.ndarray]): List of images.
        batch_size (int): Number of images per batch.

    Raises:
        ValueError: If images list is empty or not divisible by batch size.
    """
    if not images:
        raise ValueError(
            'No valid images found in the specified path.'
        )

    if len(images) % batch_size != 0:
        raise ValueError(
            'The number of input images should be divisible by the batch size '
            'without any remainder.'
        )


def divide_list_to_batches(
        images_list: List[np.ndarray], batch_size: int
) -> Generator[List[np.ndarray], None, None]:
    """
    Divide the list of images into batches.

    Args:
        images_list (List[np.ndarray]): List of images.
        batch_size (int): Number of images in each batch.

    Returns:
        Generator[List[np.ndarray], None, None]: Generator yielding batches
                                                  of images.
    """
    for i in range(0, len(images_list), batch_size):
        yield images_list[i: i + batch_size]


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

def get_labels(labels_path: str) -> list:
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


def id_to_color(idx):
    np.random.seed(idx)
    return np.random.randint(0, 255, size=3, dtype=np.uint8)



####################################################################
# PreProcess of Network Input
####################################################################

def preprocess(images: List[np.ndarray], cap: cv2.VideoCapture, batch_size: int,
               input_queue: queue.Queue, width: int, height: int,
               preprocess_fn: Optional[Callable[[np.ndarray, int, int], np.ndarray]] = None) -> None:

    """
    Preprocess and enqueue images or camera frames into the input queue as they are ready.
    Args:
        images (List[np.ndarray], optional): List of images as NumPy arrays.
        camera (bool, optional): Boolean indicating whether to use the camera stream.
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        preprocess_fn (Callable, optional): Custom preprocessing function that takes an image, width, and height,
                                            and returns the preprocessed image. If not provided, a default padding-based
                                            preprocessing function will be used.
    """
    preprocess_fn = preprocess_fn or default_preprocess

    if cap is None:
        preprocess_images(images, batch_size, input_queue, width, height, preprocess_fn)
    else:
        preprocess_from_cap(cap, batch_size, input_queue, width, height, preprocess_fn)

    input_queue.put(None)  #Add sentinel value to signal end of input


def preprocess_from_cap(cap: cv2.VideoCapture, batch_size: int, input_queue: queue.Queue, width: int, height: int,
                        preprocess_fn: Callable[[np.ndarray, int, int], np.ndarray]) -> None:
    """
    Process frames from the camera stream and enqueue them.
    Args:
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        preprocess_fn (Callable): Function to preprocess a single image (image, width, height) -> image.
    """
    frames = []
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = preprocess_fn(processed_frame, width, height)
        processed_frames.append(processed_frame)

        if len(frames) == batch_size:
            input_queue.put((frames, processed_frames))
            processed_frames, frames = [], []


def preprocess_images(images: List[np.ndarray], batch_size: int, input_queue: queue.Queue, width: int, height: int,
                      preprocess_fn: Callable[[np.ndarray, int, int], np.ndarray]) -> None:
    """
    Process a list of images and enqueue them.
    Args:
        images (List[np.ndarray]): List of images as NumPy arrays.
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        preprocess_fn (Callable): Function to preprocess a single image (image, width, height) -> image.
    """
    for batch in divide_list_to_batches(images, batch_size):
        input_tuple = ([image for image in batch], [preprocess_fn(image, width, height) for image in batch])

        input_queue.put(input_tuple)



def default_preprocess(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
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

    padded_image = np.full((model_h, model_w, 3), (114, 114, 114), dtype=np.uint8)
    x_offset = (model_w - new_img_w) // 2
    y_offset = (model_h - new_img_h) // 2
    padded_image[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = image

    return padded_image


####################################################################
# Visualization
####################################################################

def visualize(output_queue: queue.Queue, cap: cv2.VideoCapture, save_stream_output: bool, output_dir,
                callback: Callable[[Any, Any], None], frame_counter=None) -> None:
    """
    Process and visualize the output results.

    Args:
        output_queue (queue.Queue): Queue for output results.
        camera (bool): Flag indicating if the input is from a camera.
        save_stream_output (bool): Flag indicating if the camera output should be saved.
                output_dir (str or Path): Directory to save output frames.
        callback (Callable, optional): Function to be called once processing is complete.
    """

    image_id = 0
    out = None

    if cap is not None:
        #Create a named window
        cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
        #Set the window to fullscreen
        cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if save_stream_output:

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            #Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #Save the output video in the output path
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:  # If FPS is not available, set a default value
                print(f"fps: {fps}")
                fps = 20.0

            out = cv2.VideoWriter(os.path.join(output_dir,'output_video.avi'), fourcc, fps,  (frame_width, frame_height))

    while True:
        result = output_queue.get()

        if result is None:
            break  #Exit the loop if sentinel value is received

        original_frame, infer_results = result

        if len(infer_results) == 1:
            infer_results = infer_results[0]

        frame_with_detections = callback(original_frame, infer_results)

        if frame_counter is not None:
            frame_counter[0]+=1

        if cap is not None:
            # Display output
            cv2.imshow("Output", frame_with_detections)
            if save_stream_output:
                out.write(frame_with_detections)
        else:
            cv2.imwrite(os.path.join(output_dir, f"output_{image_id}.png"), frame_with_detections)

        # Wait for key press "q"
        image_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Close the window and release the camera
            if save_stream_output:
                out.release()  # Release the VideoWriter object
            cap.release()
            cv2.destroyAllWindows()
            break

    if cap is not None and save_stream_output:
        out.release()  # Release the VideoWriter object
    output_queue.task_done()  # Indicate that processing is complete



