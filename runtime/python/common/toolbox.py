from typing import List, Generator, Optional, Tuple, Dict, Callable ,Any
from pathlib import Path
from loguru import logger
import json
import os
import sys
import numpy as np
import queue
import cv2
import time
from enum import Enum
from pathlib import Path
import subprocess
import difflib


IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '0'))
RESOURCES_DOWNLOAD_DIR = Path(__file__).resolve().parents[2] / "resources_download"
GET_HEF_BASH_SCRIPT_PATH   = RESOURCES_DOWNLOAD_DIR / "get_hef.sh"
GET_INPUT_BASH_SCRIPT_PATH = RESOURCES_DOWNLOAD_DIR / "get_input.sh"
RESOLUTION_MAP = {
    "sd": (640, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080)
}

def resolve_output_resolution_arg(res_arg: Optional[list[str]]) -> Optional[Tuple[int, int]]:
    """
    Parse --output-resolution argument.

    Supported:
      --output-resolution sd|hd|fhd
      --output-resolution 1920 1080
    """
    if res_arg is None:
        return None

    # Single token: preset name (sd/hd/fhd)
    if len(res_arg) == 1:
        key = res_arg[0]
        if key in RESOLUTION_MAP:
            return RESOLUTION_MAP[key]
        raise ValueError(
            f"Invalid --output-resolution value '{key}'. "
            "Use 'sd', 'hd', 'fhd' or two integers, e.g. '--output-resolution 1920 1080'."
        )

    # Two tokens: custom width/height
    if len(res_arg) == 2 and all(x.isdigit() for x in res_arg):
        w, h = map(int, res_arg)
        if w <= 0 or h <= 0:
            raise ValueError("Custom --output-resolution width/height must be positive integers.")
        return (w, h)

    raise ValueError(
        f"Invalid --output-resolution value: {res_arg}. "
        "Use 'sd', 'hd', 'fhd' or two integers, e.g. '--output-resolution 1920 1080'."
    )

def verify_hef_arch(app: str, hef_path: Path) -> None:
    """
    Verify that the given HEF file is compatible with the connected device
    using:
        get_hef.sh verify-arch --hef <path>.
    """
    args = ["verify-arch", "--hef", str(hef_path)]
    result = run_bash_helper(GET_HEF_BASH_SCRIPT_PATH, args)


    # if result.returncode != 0:
    #     stderr = (result.stderr or "").strip()
    #     if stderr:
    #         logger.error(stderr)
    #     sys.exit(1)


def run_bash_helper(script_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    """
    Internal helper to run a Bash helper script with the given argument list.
    Does NOT implement any app-specific error handling.
    """
    if not script_path.exists():
        logger.error(f"File not found: {script_path}")
        sys.exit(1)

    cmd = [str(script_path)] + args

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result


def run_get_hef_command(args: list[str]) -> subprocess.CompletedProcess:
    """
    Wrapper around get_hef.sh with app-level error handling.
    Handles the 'no device detected' case nicely.
    """
    result = run_bash_helper(GET_HEF_BASH_SCRIPT_PATH, args)

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        # Special case: no device detected -> show a nicer app-level message
        if "No device detected" in stderr:
            logger.error(
                "\nNo Hailo device was detected.\n"
                "This application uses the connected device to choose the correct HEF "
                "(e.g., hailo8 vs hailo10h).\n"
                "Please plug in a Hailo device and run the app again.\n"
                "If you want to download a model without hardware, run get_hef.sh directly "
                "from the command line and pass --hw-arch explicitly (e.g., hailo8)."
            )
        else:
            logger.error(stderr)

        sys.exit(result.returncode)

    return result


def run_get_input_command(args: list[str]) -> subprocess.CompletedProcess:
    """
    Wrapper around get_input.sh with the given argument list.
    Shares the generic process runner but has its own error handling.
    """
    result = run_bash_helper(GET_INPUT_BASH_SCRIPT_PATH, args)

    if result.returncode != 0:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if stdout:
            logger.error(stdout)
        if stderr:
            logger.error(stderr)
        sys.exit(result.returncode)

    return result


def list_networks(app: str) -> None:
    """
    Print the supported networks for a given application.

    This delegates to the Bash helper:
        get_hef.sh list --app <app> [--hw-arch <arch>]

    Notes:
        The list is sourced from the JSON catalog file **networks.json**
    """
    # Build command args for get_hef.sh (without the script path)
    cmd_args: list[str] = ["list", "--app", app]
    logger.info("Fetching networks list... please wait")

    # Run command via shared helper (with HEF-specific error handling)
    result = run_get_hef_command(cmd_args)

    # Success output
    output = (result.stdout or "").strip()
    if output:
        footer = (
            "\n\u001b[33mPick any network name from the list above and pass it with --net (without extension).\n"
            "Example:  --net <name>\u001b[0m"
        )
        combined = f"\n{output}{footer}"
        logger.info(combined)


def list_inputs(app: str) -> None:
    """
    List predefined inputs for a given application using get_input.sh.

    Delegates to:
        get_input.sh list --app <app>
    """
    logger.info(f"Listing predefined inputs for app '{app}'...")
    result = run_get_input_command(["list", "--app", app])
    output = (result.stdout or "").strip()

    if output:
        footer = (
            "\n\n\u001b[33mPick any name from the list above and pass it with -i/--input (without extension).\n"
            "Example:  -i <name>\u001b[0m"
        )

        combined = "\n" + output + footer
        logger.info(combined)


def resolve_net_arg(app: str, net_arg: str | None, dest_dir: str = "hefs") -> str:
    """
    Resolve the --net argument into a concrete HEF path, with safety checks.

    Behavior:
      - If net_arg is None:
          -> print error, show supported networks, exit.
      - If net_arg points to an existing .hef file:
          -> verify architecture (verify-arch), return absolute path.
      - If net_arg ends with .hef but the file does NOT exist:
          -> error, hint about using name without extension, show networks, exit.
      - Otherwise (no extension):
          -> treat as network name (e.g., 'yolov8m'):
               * check if dest_dir/net_arg.hef already exists:
                     - if exists: ask user whether to reuse or re-download.
                     - if reuse: verify-arch on existing file, return path.
                     - if re-download: call get_hef(app, net_arg, dest_dir) and return.
               * if not exists: call get_hef(app, net_arg, dest_dir) and return.
    """
    # 1) No --net at all
    if net_arg is None:
        logger.error("No --net was provided.")
        list_networks(app)
        sys.exit(1)

    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    candidate = Path(net_arg)

    # 2) Existing path case (user gave a direct HEF path)
    if candidate.exists():
        if candidate.is_file() and candidate.suffix == ".hef":
            hef_path = candidate.resolve()
            logger.info(f"Using local HEF file: {hef_path}")
            verify_hef_arch(app, hef_path)
            return str(hef_path)
        else:
            logger.error(
                f"Path '{net_arg}' exists but is not a .hef file.\n"
                "Please provide either:\n"
                "  • A valid .hef file\n"
                "  • OR a network name (without extension)\n"
                f"\u001b[33mTo see all available network names, run:  --list-nets\u001b[0m\n"
            )
            sys.exit(1)

    # 3) Non-existing path with an extension (e.g., yolov8m.hef but file missing)
    if candidate.suffix == ".hef":
        logger.error(f"HEF file not found: {net_arg}")
        logger.info(
            f"\u001b[33mTo see all available network names, run:  --list-nets\u001b[0m\n"
        )
        sys.exit(1)

    # 4) No extension and file does not exist -> treat as network name
    net_name = net_arg
    existing_hef = dest_path / f"{net_name}.hef"

    logger.info(
        f"You passed a model name: '{net_name}'. "
        "Searching for this model in the supported networks... please wait."
    )

    if existing_hef.exists():
        # There is already a HEF file with this name in dest_dir
        try:
            answer = input(
                f"A HEF file already exists for network '{net_name}': {existing_hef}\n"
                "Do you want to reuse this file instead of downloading it again? [Y/n]: "
            )
        except EOFError:
            # Non-interactive scenario: default to reuse existing file
            answer = "y"

        if answer.strip().lower() in ("", "y", "yes"):
            hef_path = existing_hef.resolve()
            logger.info(f"Reusing existing HEF: {hef_path}")
            verify_hef_arch(app, hef_path)
            return str(hef_path)
        else:
            try:
                answer2 = input(
                    f"Do you want to re-download and replace '{existing_hef}'? [Y/n]: "
                )
            except EOFError:
                answer2 = "n"

            if answer2.strip().lower() in ("y", "yes", ""):
                logger.info(
                    f"Re-downloading network '{net_name}' and replacing existing HEF..."
                )
                hef_path_str = get_hef(app, net_name, dest_dir)
                hef_path = Path(hef_path_str).resolve()
                # get_hef.sh should choose the correct arch; we can still verify to be safe
                verify_hef_arch(app, hef_path)
                return str(hef_path)
            else:
                logger.error(
                    "Aborting: existing HEF was neither reused nor replaced. "
                    "Please provide a different --net or remove the file manually."
                )
                sys.exit(1)

    # 5) No existing HEF with that name -> download normally
    logger.info(f"Downloading model name {net_name}, please wait...")
    hef_path_str = get_hef(app, net_name, dest_dir)
    hef_path = Path(hef_path_str).resolve()
    logger.success(f"Download complete: {hef_path}")
    return str(hef_path)


def get_hef(app: str, net: str, dest_dir: str = "hefs") -> str:
    """
    Resolve a network to a concrete HEF path (validate or download).

    This delegates to the Bash helper:
        get_hef.sh get --app <app> --net <name|path> [--hw-arch <arch>] [--dest DIR]

    Notes:
        - Network names and their supported architectures are defined in the
          JSON catalog **networks.json**, which `get_hef.sh` reads to validate
          and locate the requested HEF.

    Args:
        app: Application key (e.g., "object_detection").
        net: Network name (e.g., "yolov8n") or a local `.hef` path.
        dest_dir: Directory for downloaded HEFs (created if missing).

    Returns:
        str: Absolute path to the resolved `.hef` file.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    cmd_args: list[str] = ["get", "--app", app, "--net", net]
    if dest_dir:
        cmd_args += ["--dest", str(dest)]

    result = run_get_hef_command(cmd_args)

    out = (result.stdout or "").strip()
    hef_path = Path(out.splitlines()[-1]).resolve()
    return str(hef_path)


def resolve_input_arg(app: str, input_arg: str | None) -> str:
    """
    Resolve the --input/-i argument into a concrete value to use in the app.

    Behavior:
      - If input_arg is None:
          Ask the user if they want to download the default input ("default")
          from resources. If yes -> download and return its path.
          If no  -> exit with an error asking them to provide an input.
      - If input_arg == "camera":
          Return "camera" unchanged.
      - If input_arg points to an existing file or directory:
          Return the path unchanged.
      - If input_arg looks like a path with an extension (e.g. 'bus.jpg'),
        but it does NOT exist:
          Print an error and list supported inputs from resources, then exit.
      - Otherwise (no extension and path does not exist):
          Treat it as a logical input ID in inputs.json and download it
          via get_input.sh (e.g. "bus"), then return the downloaded path.
    """
    # No input: interactive default download
    if input_arg is None:
        try:
            answer = input(
                f"No --input was provided for app '{app}'. "
                "Do you want to download and use the default input from resources? [Y/n]: "
            )
        except EOFError:
            logger.error("No input provided and cannot prompt interactively. "
                         "Please specify -i/--input explicitly.")
            sys.exit(1)

        if answer.strip().lower() in ("y", "yes", ""):
            return download_input(app, "default", target_dir="inputs")
        else:
            logger.error(
                "No input provided. Please run again with -i/--input or accept the default resource."
            )
            sys.exit(1)

    # "camera" stays as is
    if input_arg == "camera":
        return input_arg

    path_candidate = Path(input_arg)

    # If it already exists (file or dir), just use it as-is
    if path_candidate.exists():
        return str(path_candidate)

    # If it has an extension but does NOT exist -> error + list inputs
    if path_candidate.suffix:
        logger.error(f"Input file not found: {input_arg}")
        logger.info("Available predefined inputs for this app:")
        list_inputs(app)
        sys.exit(1)

    # No extension and path does not exist -> treat as logical ID in resources
    logger.info(
        f"Input '{input_arg}' does not exist as a local file or directory. "
        "Assuming this is a resource ID and downloading from inputs.json..."
    )
    return download_input(app, input_arg, target_dir="inputs")



def download_input(app: str, input_id: str, target_dir: str | Path = "inputs") -> str:
    """
    Download an input resource for the given app and input ID using get_input.sh.

    Delegates to:
        get_input.sh get --app <app> --target-dir <dir> --i <ID>

    Returns:
        str: Path to the downloaded file (as returned by get_input.sh).
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading input '{input_id}' for app '{app}' from resources...")
    result = run_get_input_command([
        "get",
        "--app", app,
        "--target-dir", str(target_dir_path),
        "--i", input_id,
    ])

    # get_input.sh is implemented so that the *last line* on stdout is the path
    stdout = (result.stdout or "").strip()
    if not stdout:
        logger.error("get_input.sh returned empty stdout; cannot determine downloaded path.")
        sys.exit(1)

    downloaded_path = stdout.splitlines()[-1].strip()
    logger.success(f"Download complete: {downloaded_path}")

    return downloaded_path


def get_network_meta_value(app: str, name: str, key: str, sub_key: str | None = None) -> str:
    """
    Query a specific key (and optional sub-key) from networks.json via get_hef.sh.

    Examples:
        get_network_key_value("classifier", "hardnet68", "apply_softmax")
        get_network_key_value("instance_seg", "yolov5m_seg", "postprocess", "cpp")
    """
    cmd_args: list[str] = ["get_key_value", "--app", app, "--name", name, "--key", key]
    if sub_key is not None:
        cmd_args += ["--sub_key", sub_key]

    result = run_get_hef_command(cmd_args)
    value = (result.stdout or "").strip()
    return value


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


def init_input_source(input_path, batch_size, camera_resolution):
    """
    Initialize input source from camera, video file, or image directory.

    Args:
        input_path (str): "camera", video file path, or image directory.
        batch_size (int): Number of images to validate against.

    Returns:
        Tuple[Optional[cv2.VideoCapture], Optional[List[np.ndarray]]]
    """
    cap = None
    images = None

    if input_path == "camera":

        if not is_valid_camera_index(CAMERA_INDEX):
            logger.error(f"CAMERA_INDEX {CAMERA_INDEX} not found.")
            available = list_available_cameras()
            logger.warning(f"Available camera indices: {available}")
            exit(1)

        resolution = None
        # Open camera at its native resolution; don't force sd/hd/fhd here.
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        if camera_resolution in RESOLUTION_MAP:
            CAMERA_CAP_WIDTH, CAMERA_CAP_HEIGHT = RESOLUTION_MAP.get(camera_resolution)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAP_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAP_HEIGHT)


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
    path = Path(images_path)

    def read_rgb(p: Path):
        img = cv2.imread(str(p))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        img = read_rgb(path)
        return [img] if img is not None else []

    elif path.is_dir():
        images = [
            read_rgb(img)
            for img in path.glob("*")
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return [img for img in images if img is not None]

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

def preprocess(images: List[np.ndarray], cap: cv2.VideoCapture, framerate: float, batch_size: int,
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
        preprocess_from_cap(cap, batch_size, input_queue, width, height, preprocess_fn, framerate)

    input_queue.put(None)  #Add sentinel value to signal end of input


def preprocess_from_cap(cap: cv2.VideoCapture,
                        batch_size: int,
                        input_queue: queue.Queue,
                        width: int,
                        height: int,
                        preprocess_fn: Callable[[np.ndarray, int, int], np.ndarray],
                        framerate: Optional[float] = None) -> None:
    """
    Process frames from the camera stream and enqueue them.

    If `framerate` is provided, we *skip frames* so that only approximately
    `framerate` frames per second are processed and displayed.

    The camera can still run at its native FPS (e.g. 30 FPS), but we only
    use every N-th frame. This gives the effect of 1 FPS / 5 FPS / 10 FPS
    in the live view without adding artificial lag.
    """
    frames = []
    processed_frames = []

    # Estimate camera FPS
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if not cam_fps or cam_fps <= 0:
        cam_fps = 30.0  # sensible default

    # Decide how many frames to skip
    if framerate is not None and framerate > 0:
        # e.g. cam_fps=30, framerate=1  -> skip=30  (use every 30th frame)
        #      cam_fps=30, framerate=10 -> skip=3   (use every 3rd frame)
        skip = max(1, int(round(cam_fps / float(framerate))))
    else:
        skip = 1  # no frame skipping, use all frames
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames to achieve the desired effective FPS
        if frame_idx % skip != 0:
            continue

        # Process only the kept frames
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        processed_frame = preprocess_fn(frame, width, height)
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


def oriented_object_detection_preprocess(image: np.ndarray, model_w: int, model_h: int, config_data: dict) -> np.ndarray:
    # run letterbox resize
    h0, w0 = image.shape[:2]
    new_w, new_h = model_w, model_h
    r = min(new_w / w0, new_h / h0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = (new_w - new_unpad[0]) / 2
    dh = (new_h - new_unpad[1]) / 2
    
    # calculate padding to ensure exact output dimensions
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    
    # adjust padding to ensure exact output shape
    if new_unpad[1] + top + bottom != new_h:
        bottom = new_h - new_unpad[1] - top
    if new_unpad[0] + left + right != new_w:
        right = new_w - new_unpad[0] - left
    
    color = (114, 114, 114)
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded_image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image


####################################################################
# Visualization
####################################################################
def resize_frame_for_output(frame: np.ndarray,
                            resolution: Optional[Tuple[int, int]]) -> np.ndarray:
    """
    Resize a frame according to the selected output resolution while
    preserving aspect ratio. Only the target height is enforced.

    Args:
        frame (np.ndarray): Input RGB or BGR image.
        resolution (Optional[Tuple[int, int]]): (width, height) or None.

    Returns:
        np.ndarray: Resized frame, or the original frame if resolution is None.
    """
    if resolution is None:
        return frame

    _, target_h = resolution

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame

    scale = target_h / float(h)
    new_w = int(round(w * scale))
    new_h = target_h

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def visualize(
    output_queue: queue.Queue,
    cap: Optional[cv2.VideoCapture],
    save_stream_output: bool,
    output_dir: str,
    callback: Callable[[Any, Any], None],
    fps_tracker: Optional["FrameRateTracker"] = None,
    output_resolution: Optional[Tuple[int, int]] = None,
    framerate: Optional[float] = None,
    side_by_side: bool = False
) -> None:
    """
    Visualize inference results: draw detections, show them on screen,
    and optionally save the output video.

    Args:
        output_queue: Queue with (frame, inference_result[, extra]).
        cap: VideoCapture for camera/video input, or None for image mode.
        save_stream_output: If True, write the visualization to a video file.
        output_dir: Directory to save output frames or videos.
        callback: Function that draws detections on the frame.
        fps_tracker: Tracks real-time FPS (optional).
        output_resolution: One of ['sd','hd','fhd'] or a custom resolution for final display/save size.
        framerate: Override output video FPS (optional).
        side_by_side: If True, the callback returns a wide comparison frame.
    """
    image_id = 0
    out = None
    frame_width = None
    frame_height = None

    # Window + writer init (only for camera/video, not images)
    if cap is not None:
        cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        base_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        base_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        if output_resolution is not None:
            target_w, target_h = output_resolution
        else:
            target_w, target_h = base_width, base_height

        frame_width  = target_w * (2 if side_by_side else 1)
        frame_height = target_h

        if save_stream_output:
            cam_fps   = cap.get(cv2.CAP_PROP_FPS)
            final_fps = framerate or (cam_fps if cam_fps and cam_fps > 1 else 30.0)

            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "output.avi")
            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                final_fps,
                (frame_width, frame_height),
            )

    # Main loop
    while True:
        result = output_queue.get()
        if result is None:
            output_queue.task_done()
            break

        original_frame, inference_result, *rest = result

        if isinstance(inference_result, list) and len(inference_result) == 1:
            inference_result = inference_result[0]

        if rest:
            frame_with_detections = callback(original_frame, inference_result, rest[0])
        else:
            frame_with_detections = callback(original_frame, inference_result)

        if fps_tracker is not None:
            fps_tracker.increment()

        bgr_frame = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)
        frame_to_show = resize_frame_for_output(bgr_frame, output_resolution)

        if cap is not None:
            cv2.imshow("Output", frame_to_show)
            if save_stream_output and out is not None and frame_width and frame_height:
                frame_to_save = cv2.resize(frame_to_show, (frame_width, frame_height))
                out.write(frame_to_save)
        else:
            cv2.imwrite(os.path.join(output_dir, f"output_{image_id}.png"), frame_to_show)

        image_id += 1
        output_queue.task_done()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            if save_stream_output and out is not None:
                out.release()
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            break

    if cap is not None and save_stream_output and out is not None:
        out.release()





####################################################################
# Frame Rate Tracker
####################################################################

class FrameRateTracker:
    """
    Tracks frame count and elapsed time to compute real-time FPS (frames per second).
    """

    def __init__(self):
        """Initialize the tracker with zero frames and no start time."""
        self._count = 0
        self._start_time = None

    def start(self) -> None:
        """Start or restart timing and reset the frame count."""
        self._start_time = time.time()

    def increment(self, n: int = 1) -> None:
        """Increment the frame count.

        Args:
            n (int): Number of frames to add. Defaults to 1.
        """
        self._count += n


    @property
    def count(self) -> int:
        """Returns:
            int: Total number of frames processed.
        """
        return self._count

    @property
    def elapsed(self) -> float:
        """Returns:
            float: Elapsed time in seconds since `start()` was called.
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def fps(self) -> float:
        """Returns:
            float: Calculated frames per second (FPS).
        """
        elapsed = self.elapsed
        return self._count / elapsed if elapsed > 0 else 0.0

    def frame_rate_summary(self) -> str:
        """Return a summary of frame count and FPS.

        Returns:
            str: e.g. "Processed 200 frames at 29.81 FPS"
        """
        return f"Processed {self.count} frames at {self.fps:.2f} FPS"

