#!/usr/bin/env python3

import argparse
import csv
import getpass
import hashlib
import json
import ssl
import sys
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt


EXPECTED_WORKLOAD_SHA256 = (
    "1343dcac788544f83ba943c850aa285b0094316623941f44d6911a32b492a80f"
)

EXPECTED_ROWS = 1396

DEFAULT_WORKLOAD = Path(
    "/mnt/d/ESDE/Publications/ClusterComputing/Newimpimantation/"
    "workloads/common_S4_hw_secure_trace.csv"
)

DEFAULT_CA = Path(
    "/home/lenovo/vf2-mosq/certs/ca.crt"
)

DEFAULT_PUBLISH_LOG = Path(
    "/mnt/d/ESDE/Publications/ClusterComputing/Newimpimantation/"
    "logs/S5_edge/s5_replay_publish_log.csv"
)

BROKER = "localhost"
PORT = 8883
TOPIC = "parking/slots-hw-sec"
USERNAME = "s5replay"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_workload(path):
    rows = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {
            "source_line",
            "source_epoch_ms",
            "dt_ms",
            "raw_slots_bits",
            "ai_slot1",
            "ai_slot2",
            "ai_slot3",
            "occ",
            "free",
            "entrance",
            "exit",
            "barrier",
            "source_scenario",
            "source_mode",
            "source_topic",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                "Missing workload columns: " +
                ", ".join(sorted(missing))
            )

        for seq, r in enumerate(reader, start=1):

            raw_bits = str(r["raw_slots_bits"]).strip()

            if len(raw_bits) != 3 or any(c not in "01" for c in raw_bits):
                raise RuntimeError(
                    f"Invalid raw_slots_bits at sequence {seq}: {raw_bits}"
                )

            ai_slots = [
                int(r["ai_slot1"]),
                int(r["ai_slot2"]),
                int(r["ai_slot3"]),
            ]

            expected_ai = [
                1 if bit == "0" else 0
                for bit in raw_bits
            ]

            if ai_slots != expected_ai:
                raise RuntimeError(
                    f"AI normalization mismatch at sequence {seq}: "
                    f"raw={raw_bits}, "
                    f"stored={ai_slots}, "
                    f"expected={expected_ai}"
                )

            occ = int(r["occ"])
            free = int(r["free"])

            if occ != raw_bits.count("0"):
                raise RuntimeError(
                    f"Occupancy mismatch at sequence {seq}"
                )

            if free != raw_bits.count("1"):
                raise RuntimeError(
                    f"Free-count mismatch at sequence {seq}"
                )

            if occ + free != 3:
                raise RuntimeError(
                    f"Invalid occupancy total at sequence {seq}"
                )

            rows.append({
                "seq": seq,
                "source_line": int(r["source_line"]),
                "source_epoch_ms": int(r["source_epoch_ms"]),
                "dt_ms": int(r["dt_ms"]),
                "raw_slots_bits": raw_bits,
                "slots": ai_slots,
                "occ": occ,
                "free": free,
                "entrance": int(r["entrance"]),
                "exit": int(r["exit"]),
                "barrier": str(r["barrier"]),
                "source_scenario": str(r["source_scenario"]),
                "source_mode": str(r["source_mode"]),
                "source_topic": str(r["source_topic"]),
            })

    return rows


def build_payload(row):
    return {
        # Historical telemetry identity
        "ts": row["source_epoch_ms"],
        "source_epoch_ms": row["source_epoch_ms"],
        "source_line": row["source_line"],
        "replay_seq": row["seq"],
        "raw_slots_bits": row["raw_slots_bits"],

        # Current S5 pipeline expects normalized representation:
        # 1 = occupied, 0 = free
        "slots": row["slots"],

        "free": row["free"],
        "entrance": row["entrance"],
        "exit": row["exit"],
        "barrier": row["barrier"],

        "src": "S4_HW_SECURE_TRACE_REPLAY",
        "source_scenario": "S4",
        "mode": "secure",
    }


def write_publish_log_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "replay_seq",
            "source_line",
            "source_epoch_ms",
            "raw_slots_bits",
            "ai_slots",
            "scheduled_elapsed_ms",
            "actual_elapsed_ms",
            "timing_error_ms",
            "publish_rc",
        ])


def append_publish_log(
    path,
    row,
    scheduled_elapsed,
    actual_elapsed,
    rc,
):
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            row["seq"],
            row["source_line"],
            row["source_epoch_ms"],
            row["raw_slots_bits"],
            "".join(str(v) for v in row["slots"]),
            round(scheduled_elapsed * 1000.0, 3),
            round(actual_elapsed * 1000.0, 3),
            round(
                (actual_elapsed - scheduled_elapsed) * 1000.0,
                3
            ),
            rc,
        ])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replay the frozen S4 hardware-secure telemetry trace "
            "for controlled S5/S6/S7 experiments."
        )
    )

    parser.add_argument(
        "--workload",
        type=Path,
        default=DEFAULT_WORKLOAD,
    )

    parser.add_argument(
        "--ca",
        type=Path,
        default=DEFAULT_CA,
    )

    parser.add_argument(
        "--publish-log",
        type=Path,
        default=DEFAULT_PUBLISH_LOG,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate workload only; do not connect or publish.",
    )

    args = parser.parse_args()

    print("===== WORKLOAD VERIFICATION =====")

    if not args.workload.exists():
        raise SystemExit(
            f"ERROR: workload not found: {args.workload}"
        )

    actual_sha = sha256_file(args.workload)

    print("File   :", args.workload)
    print("SHA256 :", actual_sha)

    if actual_sha != EXPECTED_WORKLOAD_SHA256:
        raise SystemExit(
            "ERROR: workload SHA256 does not match the frozen trace."
        )

    rows = load_workload(args.workload)

    print("Rows   :", len(rows))

    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(
            f"ERROR: expected {EXPECTED_ROWS} rows, "
            f"found {len(rows)}"
        )

    first_ts = rows[0]["source_epoch_ms"]
    last_ts = rows[-1]["source_epoch_ms"]

    duration_s = (last_ts - first_ts) / 1000.0

    print("Source line range:",
          rows[0]["source_line"],
          "-",
          rows[-1]["source_line"])

    print("Source duration  :",
          round(duration_s, 3),
          "s")

    print("First raw/AI     :",
          rows[0]["raw_slots_bits"],
          "->",
          rows[0]["slots"])

    print("Last raw/AI      :",
          rows[-1]["raw_slots_bits"],
          "->",
          rows[-1]["slots"])

    print()
    print("===== FIRST PAYLOAD =====")
    print(
        json.dumps(
            build_payload(rows[0]),
            indent=2
        )
    )

    if args.dry_run:
        print()
        print("DRY RUN: PASS")
        print("No MQTT connection was created.")
        print("No telemetry was published.")
        return

    if not args.ca.exists():
        raise SystemExit(
            f"ERROR: CA certificate not found: {args.ca}"
        )

    password = getpass.getpass(
        "Password for MQTT user s5replay: "
    )

    connected = threading.Event()
    connection_result = {"rc": None}

    def on_connect(client, userdata, flags, rc):
        connection_result["rc"] = rc

        if rc == 0:
            connected.set()

    def on_disconnect(client, userdata, rc):
        pass

    client = mqtt.Client(
        client_id="s5_common_trace_replay",
        clean_session=True,
        protocol=mqtt.MQTTv311,
    )

    client.username_pw_set(
        USERNAME,
        password
    )

    client.tls_set(
        ca_certs=str(args.ca),
        certfile=None,
        keyfile=None,
        cert_reqs=ssl.CERT_REQUIRED,
        tls_version=ssl.PROTOCOL_TLS_CLIENT,
    )

    client.tls_insecure_set(False)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    print()
    print("===== MQTT CONNECTION =====")
    print(f"Broker : {BROKER}:{PORT}")
    print(f"Topic  : {TOPIC}")
    print("TLS    : enabled")
    print("User   :", USERNAME)

    client.connect(
        BROKER,
        PORT,
        keepalive=60
    )

    client.loop_start()

    if not connected.wait(timeout=10):
        client.loop_stop()
        raise SystemExit(
            "ERROR: MQTT TLS connection/authentication failed "
            f"(rc={connection_result['rc']})."
        )

    print("MQTT connection: OK")

    write_publish_log_header(args.publish_log)

    print()
    print("===== REPLAY START =====")
    print("Rows              :", len(rows))
    print("Original duration :", round(duration_s, 3), "s")
    print("Timing mode       : original S4 timestamps")
    print("QoS               : 0")
    print("Expected duration : ~4.81 min")
    print()

    run_start = time.monotonic()

    first_source_ts = rows[0]["source_epoch_ms"]

    failures = 0

    try:
        for row in rows:

            scheduled_elapsed = (
                row["source_epoch_ms"] - first_source_ts
            ) / 1000.0

            target_time = (
                run_start + scheduled_elapsed
            )

            while True:
                remaining = target_time - time.monotonic()

                if remaining <= 0:
                    break

                time.sleep(min(remaining, 0.05))

            payload = build_payload(row)

            info = client.publish(
                TOPIC,
                json.dumps(
                    payload,
                    separators=(",", ":")
                ),
                qos=0,
                retain=False,
            )

            actual_elapsed = (
                time.monotonic() - run_start
            )

            append_publish_log(
                args.publish_log,
                row,
                scheduled_elapsed,
                actual_elapsed,
                info.rc,
            )

            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                failures += 1

            if (
                row["seq"] == 1
                or row["seq"] % 100 == 0
                or row["seq"] == len(rows)
            ):
                print(
                    f"{row['seq']:4d}/{len(rows)} "
                    f"| source_line={row['source_line']} "
                    f"| raw={row['raw_slots_bits']} "
                    f"| AI={row['slots']} "
                    f"| elapsed={actual_elapsed:.3f}s "
                    f"| rc={info.rc}"
                )

    except KeyboardInterrupt:
        print()
        print("Replay interrupted by user.")
        failures += 1

    finally:
        # Give the local MQTT client a moment to flush.
        time.sleep(1.0)

        client.disconnect()
        client.loop_stop()

    total_elapsed = time.monotonic() - run_start

    print()
    print("===== REPLAY COMPLETE =====")
    print("Elapsed (s)       :", round(total_elapsed, 3))
    print("Publish failures  :", failures)
    print("Publish log       :", args.publish_log)

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
