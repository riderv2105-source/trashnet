import sys
import os
import cv2
import json
import time
import uuid
import threading
from datetime import datetime

from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, classify


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    "/home/mendel/"
    "GarbageClassification_int8_edgetpu.tflite"
)

LABEL_PATH = "/home/mendel/labels.txt"

# Replace this with the current URL shown by the iPhone app.
CAMERA_URL = "http://192.168.1.74:8080/"

DEVICE_ID = "coral-dev-board-mini"
CAMERA_ID = "iphone-ip-camera"

JSONL_OUTPUT_PATH = "/home/mendel/inference_stream.jsonl"

# Retry only when the iPhone stream disconnects.
RECONNECT_DELAY_SECONDS = 1.0

# Display configuration.
DISPLAY_WINDOW_NAME = "Edge AI Trash Classification"

# Use 640 x 480 for the smoothest performance.
# Change to 960 x 540 if the HDMI display looks too small.
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# If True, print every prediction JSON to the terminal.
# Terminal printing can reduce performance, so False is smoother.
PRINT_JSON_TO_TERMINAL = False

# Flush buffered JSONL data approximately every 0.5 seconds.
JSON_FLUSH_INTERVAL_SECONDS = 0.5


# ============================================================
# Shared program state
# ============================================================

stop_event = threading.Event()

frame_lock = threading.Lock()
result_lock = threading.Lock()

latest_frame = None
latest_frame_number = 0

latest_result = {
    "frame_id": 0,
    "predicted_class": "INITIALIZING",
    "confidence": 0.0,
    "latency_ms": 0.0,
    "inference_fps": 0.0
}

camera_connected = False


# ============================================================
# Logging and file validation
# ============================================================

def log(message):
    """
    System messages use stderr so JSON output can remain
    separate from application information.
    """
    print(message, file=sys.stderr, flush=True)


def validate_required_files():
    """
    Confirm that the compiled model and labels file exist.
    """
    missing_files = []

    if not os.path.exists(MODEL_PATH):
        missing_files.append(MODEL_PATH)

    if not os.path.exists(LABEL_PATH):
        missing_files.append(LABEL_PATH)

    if missing_files:
        for missing_file in missing_files:
            log(
                "[ERROR] Missing required file: {}".format(
                    missing_file
                )
            )

        raise SystemExit(1)


def load_labels(label_path):
    """
    Load plain or numerically indexed class labels.
    """
    labels = {}

    with open(label_path, "r") as label_file:
        for default_index, line in enumerate(label_file):
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                class_index = int(parts[0])
                class_name = parts[1].strip()
            else:
                class_index = default_index
                class_name = line

            labels[class_index] = class_name

    return labels


# ============================================================
# Camera capture thread
# ============================================================

def camera_capture_worker():
    """
    Continuously receive frames from the iPhone.

    Only the newest frame is retained. Old frames are discarded
    automatically, preventing the HDMI display from falling
    progressively behind the live iPhone camera.
    """
    global latest_frame
    global latest_frame_number
    global camera_connected

    camera = None

    while not stop_event.is_set():

        if camera is None:
            log(
                "[INFO] Connecting to iPhone camera: {}".format(
                    CAMERA_URL
                )
            )

            camera = cv2.VideoCapture(CAMERA_URL)

            # Request the smallest possible internal frame buffer.
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not camera.isOpened():
                log(
                    "[ERROR] Could not open iPhone camera stream."
                )

                camera.release()
                camera = None
                camera_connected = False

                stop_event.wait(RECONNECT_DELAY_SECONDS)
                continue

            camera_connected = True

            log(
                "[OK] iPhone camera stream opened successfully."
            )

        received, frame = camera.read()

        if not received or frame is None:
            log("[WARN] Camera frame was not received.")
            log("[WARN] Reconnecting to iPhone camera.")

            camera_connected = False
            camera.release()
            camera = None

            stop_event.wait(RECONNECT_DELAY_SECONDS)
            continue

        # Replace the old frame immediately with the newest one.
        with frame_lock:
            latest_frame = frame
            latest_frame_number += 1

    camera_connected = False

    if camera is not None:
        camera.release()

    log("[INFO] Camera capture thread stopped.")


# ============================================================
# Edge TPU inference thread
# ============================================================

def preprocess_frame(frame, input_width, input_height):
    """
    Convert BGR to RGB and resize directly to model input size.
    """
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    resized_frame = cv2.resize(
        rgb_frame,
        (input_width, input_height),
        interpolation=cv2.INTER_LINEAR
    )

    return Image.fromarray(resized_frame)


def create_payload(
    frame_id,
    predicted_class,
    confidence,
    latency_ms,
    inference_fps
):
    """
    Create one cloud-ready inference record.
    """
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ),
        "device_id": DEVICE_ID,
        "camera_id": CAMERA_ID,
        "frame_id": int(frame_id),
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 4),
        "latency_ms": round(float(latency_ms), 2),
        "inference_fps": round(float(inference_fps), 2)
    }


def inference_worker(labels):
    """
    Run Edge TPU inference continuously on the newest available
    frame without blocking camera capture or HDMI display.
    """
    global latest_result

    log("[INFO] Loading Edge TPU model:")
    log("       {}".format(MODEL_PATH))

    try:
        interpreter = make_interpreter(MODEL_PATH)
        interpreter.allocate_tensors()

    except Exception as error:
        log("[ERROR] Edge TPU initialization failed.")
        log("[ERROR] {}".format(error))
        stop_event.set()
        return

    input_details = interpreter.get_input_details()[0]
    input_shape = input_details["shape"]

    input_height = int(input_shape[1])
    input_width = int(input_shape[2])

    log("[OK] Edge TPU interpreter initialized.")

    log(
        "[INFO] Model input size: {} x {}".format(
            input_width,
            input_height
        )
    )

    last_processed_camera_frame = -1

    inference_count = 0
    inference_fps = 0.0
    inference_fps_start = time.perf_counter()

    last_file_flush = time.perf_counter()

    try:
        output_file = open(
            JSONL_OUTPUT_PATH,
            "a",
            buffering=8192
        )

    except Exception as error:
        log("[ERROR] Could not open JSONL output file.")
        log("[ERROR] {}".format(error))
        stop_event.set()
        return

    try:
        while not stop_event.is_set():

            # Copy only the newest camera frame.
            with frame_lock:
                if (
                    latest_frame is None
                    or latest_frame_number
                    == last_processed_camera_frame
                ):
                    frame_for_inference = None
                    camera_frame_id = latest_frame_number
                else:
                    frame_for_inference = latest_frame.copy()
                    camera_frame_id = latest_frame_number

            if frame_for_inference is None:
                # Very short wait prevents unnecessary CPU spinning.
                stop_event.wait(0.001)
                continue

            last_processed_camera_frame = camera_frame_id

            input_image = preprocess_frame(
                frame_for_inference,
                input_width,
                input_height
            )

            common.set_input(
                interpreter,
                input_image
            )

            inference_start = time.perf_counter()

            interpreter.invoke()

            inference_end = time.perf_counter()

            latency_ms = (
                inference_end - inference_start
            ) * 1000.0

            results = classify.get_classes(
                interpreter,
                top_k=1,
                score_threshold=0.0
            )

            if results:
                top_result = results[0]

                predicted_class = labels.get(
                    top_result.id,
                    "class_{}".format(top_result.id)
                )

                confidence = float(top_result.score)

            else:
                predicted_class = "unknown"
                confidence = 0.0

            # Calculate actual classification rate.
            inference_count += 1

            current_time = time.perf_counter()

            inference_elapsed = (
                current_time - inference_fps_start
            )

            if inference_elapsed >= 1.0:
                inference_fps = (
                    inference_count / inference_elapsed
                )

                inference_count = 0
                inference_fps_start = current_time

            with result_lock:
                latest_result = {
                    "frame_id": camera_frame_id,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "inference_fps": inference_fps
                }

            payload = create_payload(
                frame_id=camera_frame_id,
                predicted_class=predicted_class,
                confidence=confidence,
                latency_ms=latency_ms,
                inference_fps=inference_fps
            )

            json_message = json.dumps(
                payload,
                separators=(",", ":"),
                ensure_ascii=True
            )

            # Buffered file writing is faster than opening and
            # flushing the file after every prediction.
            output_file.write(json_message + "\n")

            if PRINT_JSON_TO_TERMINAL:
                print(json_message, flush=False)

            # Flush periodically so Member 2 can still read
            # near-real-time predictions from the JSONL file.
            if (
                current_time - last_file_flush
                >= JSON_FLUSH_INTERVAL_SECONDS
            ):
                output_file.flush()
                last_file_flush = current_time

    except Exception as error:
        log("[ERROR] Edge inference thread failed:")
        log("[ERROR] {}".format(error))
        stop_event.set()

    finally:
        output_file.flush()
        output_file.close()

        log("[INFO] Edge inference thread stopped.")


# ============================================================
# HDMI output
# ============================================================

def create_display_frame(
    frame,
    result,
    display_fps
):
    """
    Draw a lightweight overlay without resizing to a large
    960 x 720 canvas on every frame.
    """
    display_frame = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    height, width = display_frame.shape[:2]

    # Top dark panel.
    cv2.rectangle(
        display_frame,
        (0, 0),
        (width, 64),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display_frame,
        "EDGE AI TRASH CLASSIFICATION",
        (18, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Bottom dark panel.
    cv2.rectangle(
        display_frame,
        (0, height - 125),
        (width, height),
        (0, 0, 0),
        -1
    )

    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    latency_ms = result["latency_ms"]
    inference_fps = result["inference_fps"]

    cv2.putText(
        display_frame,
        predicted_class.upper(),
        (18, height - 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        display_frame,
        "Confidence: {:.2f}%".format(
            confidence * 100.0
        ),
        (220, height - 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        display_frame,
        "TPU: {:.2f} ms".format(latency_ms),
        (220, height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        display_frame,
        "Display: {:.1f} FPS | Inference: {:.1f} FPS".format(
            display_fps,
            inference_fps
        ),
        (18, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    return display_frame


# ============================================================
# Main application
# ============================================================

def main():
    log(
        "======================================================"
    )
    log(
        " iPhone Camera Edge TPU Trash Classification System"
    )
    log(
        "======================================================"
    )

    validate_required_files()

    labels = load_labels(LABEL_PATH)

    log(
        "[INFO] Loaded {} class labels.".format(
            len(labels)
        )
    )

    log(
        "[INFO] JSONL output: {}".format(
            JSONL_OUTPUT_PATH
        )
    )

    log("[INFO] Starting optimized threaded pipeline.")
    log("[INFO] Press Q on the HDMI window to stop.")
    log("[INFO] Ctrl+C can also stop the program.")

    camera_thread = threading.Thread(
        target=camera_capture_worker,
        name="CameraCaptureThread"
    )

    edge_tpu_thread = threading.Thread(
        target=inference_worker,
        args=(labels,),
        name="EdgeTPUInferenceThread"
    )

    camera_thread.daemon = True
    edge_tpu_thread.daemon = True

    camera_thread.start()
    edge_tpu_thread.start()

    display_frame_count = 0
    display_fps = 0.0
    display_fps_start = time.perf_counter()

    try:
        while not stop_event.is_set():

            with frame_lock:
                if latest_frame is None:
                    frame_to_display = None
                else:
                    frame_to_display = latest_frame.copy()

            if frame_to_display is None:
                time.sleep(0.01)
                continue

            with result_lock:
                result_to_display = dict(latest_result)

            display_frame_count += 1

            current_time = time.perf_counter()

            display_elapsed = (
                current_time - display_fps_start
            )

            if display_elapsed >= 1.0:
                display_fps = (
                    display_frame_count / display_elapsed
                )

                display_frame_count = 0
                display_fps_start = current_time

            output_frame = create_display_frame(
                frame_to_display,
                result_to_display,
                display_fps
            )

            cv2.imshow(
                DISPLAY_WINDOW_NAME,
                output_frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                log("[INFO] Q key pressed.")
                stop_event.set()
                break

    except KeyboardInterrupt:
        log("")
        log("[INFO] Ctrl+C received.")
        stop_event.set()

    except Exception as error:
        log("[ERROR] HDMI display error:")
        log("[ERROR] {}".format(error))
        stop_event.set()

    finally:
        stop_event.set()

        camera_thread.join(timeout=3.0)
        edge_tpu_thread.join(timeout=3.0)

        cv2.destroyAllWindows()

        log("[INFO] HDMI display window closed.")
        log("[INFO] Application stopped.")


if __name__ == "__main__":
    main()
