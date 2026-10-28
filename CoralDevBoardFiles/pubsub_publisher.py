import os
import sys
import json
import time

from google.cloud import pubsub_v1


# ============================================================
# Configuration
# ============================================================

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
TOPIC_ID = os.environ.get("PUBSUB_TOPIC_ID")

JSONL_PATH = "/home/mendel/inference_stream.jsonl"

# Publish approximately one record per second.
PUBLISH_INTERVAL_SECONDS = 1.0

# How often to check the JSONL file for new records.
FILE_CHECK_INTERVAL_SECONDS = 0.05

# Wait before retrying a failed Pub/Sub message.
RETRY_DELAY_SECONDS = 3.0

# True:
# Skip old JSONL records and publish only new live predictions.
#
# False:
# Start reading and publishing from the beginning of the file.
START_AT_END_OF_FILE = True


# ============================================================
# Logging
# ============================================================

def log(message):
    """
    Print publisher information to standard error.
    """
    print(message, file=sys.stderr, flush=True)


# ============================================================
# Configuration validation
# ============================================================

def validate_configuration():
    """
    Verify the cloud environment variables, credential file,
    and project/topic configuration.
    """
    if not PROJECT_ID:
        log("[ERROR] GCP_PROJECT_ID is not set.")
        log(
            "[INFO] Run: "
            "source /home/mendel/coral_cloud_config.sh"
        )
        raise SystemExit(1)

    if not TOPIC_ID:
        log("[ERROR] PUBSUB_TOPIC_ID is not set.")
        log(
            "[INFO] Run: "
            "source /home/mendel/coral_cloud_config.sh"
        )
        raise SystemExit(1)

    credential_path = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )

    if not credential_path:
        log(
            "[ERROR] GOOGLE_APPLICATION_CREDENTIALS "
            "is not set."
        )
        log(
            "[INFO] Run: "
            "source /home/mendel/coral_cloud_config.sh"
        )
        raise SystemExit(1)

    if not os.path.isfile(credential_path):
        log(
            "[ERROR] Credential file was not found:"
        )
        log(
            "        {}".format(
                credential_path
            )
        )
        raise SystemExit(1)

    if not os.path.isfile(JSONL_PATH):
        log(
            "[WARN] JSONL file does not exist yet:"
        )
        log(
            "       {}".format(
                JSONL_PATH
            )
        )
        log(
            "[INFO] The publisher will wait for "
            "edge_inference.py to create it."
        )


# ============================================================
# Inference-record validation
# ============================================================

def validate_record(record):
    """
    Check that an inference record contains all fields expected
    by Pub/Sub and the future BigQuery table.
    """
    required_fields = [
        "event_id",
        "timestamp",
        "device_id",
        "camera_id",
        "frame_id",
        "predicted_class",
        "confidence",
        "latency_ms",
        "inference_fps"
    ]

    missing_fields = []

    for field_name in required_fields:
        if field_name not in record:
            missing_fields.append(field_name)

    if missing_fields:
        raise ValueError(
            "Missing fields: {}".format(
                ", ".join(missing_fields)
            )
        )

    # Confirm numeric fields can be converted correctly.
    int(record["frame_id"])
    float(record["confidence"])
    float(record["latency_ms"])
    float(record["inference_fps"])

    return True


# ============================================================
# Pub/Sub publishing
# ============================================================

def publish_record(
    publisher,
    topic_path,
    record
):
    """
    Publish one Coral classification record to Pub/Sub and wait
    for Google Cloud to return the Pub/Sub message ID.
    """
    validate_record(record)

    message_data = json.dumps(
        record,
        separators=(",", ":"),
        ensure_ascii=True
    ).encode("utf-8")

    future = publisher.publish(
        topic_path,
        message_data,

        source="coral-dev-board-mini",
        message_type="trash-classification",

        device_id=str(
            record["device_id"]
        ),

        camera_id=str(
            record["camera_id"]
        ),

        predicted_class=str(
            record["predicted_class"]
        ),

        event_id=str(
            record["event_id"]
        )
    )

    message_id = future.result(
        timeout=30
    )

    return message_id


# ============================================================
# JSONL file monitoring
# ============================================================

def wait_for_jsonl_file():
    """
    Wait until edge_inference.py creates the JSONL file.
    """
    warning_shown = False

    while not os.path.isfile(JSONL_PATH):

        if not warning_shown:
            log(
                "[WARN] Waiting for JSONL file:"
            )
            log(
                "       {}".format(
                    JSONL_PATH
                )
            )

            warning_shown = True

        time.sleep(1)

    log("[OK] JSONL inference file is available.")


def follow_jsonl_file(
    publisher,
    topic_path
):
    """
    Follow inference_stream.jsonl and publish approximately one
    classification record per second.

    The JSONL file may receive records at full inference speed,
    but records inside the publishing interval are skipped.
    """
    published_count = 0
    skipped_by_interval_count = 0
    invalid_record_count = 0
    failed_attempt_count = 0

    wait_for_jsonl_file()

    last_publish_time = 0.0

    try:
        with open(JSONL_PATH, "r") as jsonl_file:

            if START_AT_END_OF_FILE:
                jsonl_file.seek(
                    0,
                    os.SEEK_END
                )

                log(
                    "[INFO] Existing historical records skipped."
                )

                log(
                    "[INFO] Waiting for new live predictions."
                )

            else:
                log(
                    "[INFO] Publishing records from the "
                    "beginning of the JSONL file."
                )

            while True:

                line = jsonl_file.readline()

                if not line:
                    time.sleep(
                        FILE_CHECK_INTERVAL_SECONDS
                    )
                    continue

                line = line.strip()

                if not line:
                    continue

                # --------------------------------------------
                # Validate JSON syntax.
                # --------------------------------------------

                try:
                    record = json.loads(line)

                except json.JSONDecodeError as error:
                    invalid_record_count += 1

                    log(
                        "[WARN] Invalid JSONL record skipped: {}"
                        .format(error)
                    )

                    continue

                # --------------------------------------------
                # Cloud publication rate limit.
                # --------------------------------------------

                current_time = time.monotonic()

                time_since_last_publish = (
                    current_time - last_publish_time
                )

                if (
                    time_since_last_publish
                    < PUBLISH_INTERVAL_SECONDS
                ):
                    skipped_by_interval_count += 1
                    continue

                # --------------------------------------------
                # Publish and retry the same record on failure.
                # --------------------------------------------

                while True:

                    try:
                        message_id = publish_record(
                            publisher,
                            topic_path,
                            record
                        )

                        published_count += 1
                        last_publish_time = time.monotonic()

                        log(
                            "[PUBSUB] #{} "
                            "message_id={} "
                            "frame={} "
                            "class={} "
                            "confidence={:.2f}% "
                            "latency={:.2f}ms "
                            "inference_fps={:.2f}"
                            .format(
                                published_count,
                                message_id,
                                record.get(
                                    "frame_id",
                                    "unknown"
                                ),
                                record.get(
                                    "predicted_class",
                                    "unknown"
                                ),
                                float(
                                    record.get(
                                        "confidence",
                                        0.0
                                    )
                                ) * 100.0,
                                float(
                                    record.get(
                                        "latency_ms",
                                        0.0
                                    )
                                ),
                                float(
                                    record.get(
                                        "inference_fps",
                                        0.0
                                    )
                                )
                            )
                        )

                        break

                    except ValueError as error:
                        invalid_record_count += 1

                        log(
                            "[WARN] Record validation failed: {}"
                            .format(error)
                        )

                        break

                    except Exception as error:
                        failed_attempt_count += 1

                        log(
                            "[ERROR] Pub/Sub publication failed:"
                        )

                        log(
                            "[ERROR] {}".format(
                                error
                            )
                        )

                        log(
                            "[INFO] Retrying the same record "
                            "in {:.1f} seconds..."
                            .format(
                                RETRY_DELAY_SECONDS
                            )
                        )

                        time.sleep(
                            RETRY_DELAY_SECONDS
                        )

    except KeyboardInterrupt:
        log("")
        log("[INFO] Stop command received.")

    finally:
        log(
            "=================================================="
        )
        log(
            " Pub/Sub Publisher Summary"
        )
        log(
            "=================================================="
        )

        log(
            "[INFO] Successfully published: {}".format(
                published_count
            )
        )

        log(
            "[INFO] Skipped by interval: {}".format(
                skipped_by_interval_count
            )
        )

        log(
            "[INFO] Invalid records: {}".format(
                invalid_record_count
            )
        )

        log(
            "[INFO] Failed attempts: {}".format(
                failed_attempt_count
            )
        )

        log(
            "[INFO] Publisher stopped."
        )


# ============================================================
# Main application
# ============================================================

def main():
    validate_configuration()

    publisher = pubsub_v1.PublisherClient()

    topic_path = publisher.topic_path(
        PROJECT_ID,
        TOPIC_ID
    )

    log(
        "=================================================="
    )
    log(
        " Coral JSONL to Google Cloud Pub/Sub Publisher"
    )
    log(
        "=================================================="
    )

    log(
        "[INFO] Project: {}".format(
            PROJECT_ID
        )
    )

    log(
        "[INFO] Topic: {}".format(
            topic_path
        )
    )

    log(
        "[INFO] JSONL source: {}".format(
            JSONL_PATH
        )
    )

    log(
        "[INFO] Publish interval: {:.1f} second(s)"
        .format(
            PUBLISH_INTERVAL_SECONDS
        )
    )

    log(
        "[INFO] Expected cloud rate: approximately "
        "{:.1f} message(s) per second"
        .format(
            1.0 / PUBLISH_INTERVAL_SECONDS
        )
    )

    log(
        "[INFO] Press Ctrl+C to stop."
    )

    log(
        "=================================================="
    )

    follow_jsonl_file(
        publisher,
        topic_path
    )


if __name__ == "__main__":
    main()
