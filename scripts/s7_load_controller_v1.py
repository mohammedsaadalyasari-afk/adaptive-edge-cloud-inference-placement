#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


EXPECTED_SCHEDULE_SHA256 = (
    "4746b325277f35fd2031d540075908a86"
    "c58e65faec8ba864962d15c7aac322d"
)

EXPECTED_DECISIONS = 289

ALLOWED_PIN_ORDER = [
    0, 2, 4, 6, 1, 3, 5, 7
]


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def utc_timestamp():
    return (
        datetime.now(timezone.utc)
        .isoformat()
    )


class LoadController:

    def __init__(
        self,
        schedule_path,
        dry_run,
        event_log,
    ):
        self.schedule_path = Path(
            schedule_path
        )

        self.dry_run = bool(
            dry_run
        )

        actual_sha = sha256_file(
            self.schedule_path
        )

        if (
            actual_sha
            != EXPECTED_SCHEDULE_SHA256
        ):
            raise RuntimeError(
                "Frozen S7 load schedule SHA mismatch: "
                f"{actual_sha}"
            )

        self.schedule_sha256 = (
            actual_sha
        )

        with self.schedule_path.open(
            newline="",
            encoding="utf-8"
        ) as f:
            rows = list(
                csv.DictReader(f)
            )

        if len(rows) != EXPECTED_DECISIONS:
            raise RuntimeError(
                "Expected 289 decisions, "
                f"found {len(rows)}"
            )

        self.rows = {}

        previous = 0

        for row in rows:

            decision = int(
                row["decision_number"]
            )

            replay_seq = int(
                row["replay_seq"]
            )

            workers = int(
                row["worker_count"]
            )

            pins_text = (
                row["cpu_pins"].strip()
            )

            pins = (
                [
                    int(x)
                    for x in pins_text.split(",")
                    if x.strip()
                ]
                if pins_text
                else []
            )

            if decision != previous + 1:
                raise RuntimeError(
                    "Decision schedule is "
                    "not contiguous"
                )

            if workers != len(pins):
                raise RuntimeError(
                    f"Decision {decision}: "
                    "worker/pin count mismatch"
                )

            for pin in pins:
                if pin not in ALLOWED_PIN_ORDER:
                    raise RuntimeError(
                        f"Unexpected CPU pin {pin}"
                    )

            self.rows[decision] = {
                **row,
                "decision_number_int":
                    decision,
                "replay_seq_int":
                    replay_seq,
                "worker_count_int":
                    workers,
                "pins_int":
                    pins,
            }

            previous = decision

        if previous != EXPECTED_DECISIONS:
            raise RuntimeError(
                "Schedule does not terminate "
                "at decision 289"
            )

        replay_seqs = [
            r["replay_seq_int"]
            for r in self.rows.values()
        ]

        if (
            len(set(replay_seqs))
            != EXPECTED_DECISIONS
        ):
            raise RuntimeError(
                "Duplicate replay_seq "
                "in load schedule"
            )

        self.lock = threading.Lock()

        self.current_decision = 0
        self.current_phase = None

        self.current_target_workers = 0
        self.current_pins = []

        self.workers = []

        self.closed = False

        self.event_path = Path(
            event_log
        )

        self.event_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.event_file = (
            self.event_path.open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
        )

        self.log_event(
            "controller_start",
            schedule_sha256=
                self.schedule_sha256,
            dry_run=self.dry_run,
        )


    def log_event(
        self,
        event,
        **fields
    ):
        record = {
            "timestamp":
                utc_timestamp(),

            "event":
                event,

            **fields,
        }

        self.event_file.write(
            json.dumps(
                record,
                sort_keys=True
            )
            + "\n"
        )


    def stop_worker_processes(self):

        if self.dry_run:
            self.workers = []
            return

        for item in self.workers:

            p = item["process"]

            if p.poll() is None:
                p.terminate()

        deadline = (
            time.monotonic()
            + 5.0
        )

        for item in self.workers:

            p = item["process"]

            if p.poll() is not None:
                continue

            remaining = max(
                0.0,
                deadline
                - time.monotonic()
            )

            try:
                p.wait(
                    timeout=remaining
                )

            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

        self.workers = []


    def start_worker_processes(
        self,
        pins
    ):

        if self.dry_run:
            self.workers = []
            return

        started = []

        try:

            for pin in pins:

                p = subprocess.Popen(
                    [
                        "taskset",
                        "-c",
                        str(pin),
                        "/usr/bin/python3",
                        "-c",
                        "while True: pass",
                    ],
                    stdout=
                        subprocess.DEVNULL,
                    stderr=
                        subprocess.DEVNULL,
                    start_new_session=True,
                )

                started.append({
                    "cpu": pin,
                    "process": p,
                })

            time.sleep(0.10)

            for item in started:

                if (
                    item["process"].poll()
                    is not None
                ):
                    raise RuntimeError(
                        "Load worker failed "
                        f"on CPU {item['cpu']}"
                    )

            self.workers = started

        except Exception:

            self.workers = started
            self.stop_worker_processes()

            raise


    def transition_phase(
        self,
        row
    ):
        old_phase = (
            self.current_phase
        )

        target_phase = (
            row["phase"]
        )

        target_workers = (
            row["worker_count_int"]
        )

        target_pins = (
            row["pins_int"]
        )

        self.stop_worker_processes()

        self.start_worker_processes(
            target_pins
        )

        self.current_phase = (
            target_phase
        )

        self.current_target_workers = (
            target_workers
        )

        self.current_pins = list(
            target_pins
        )

        self.log_event(
            "phase_transition",
            decision_number=
                row[
                    "decision_number_int"
                ],
            replay_seq=
                row["replay_seq_int"],
            old_phase=old_phase,
            new_phase=target_phase,
            target_worker_count=
                target_workers,
            cpu_pins=target_pins,
            dry_run=self.dry_run,
        )


    def apply_decision(
        self,
        decision_number,
        replay_seq
    ):

        with self.lock:

            if self.closed:
                raise RuntimeError(
                    "Controller is closed"
                )

            decision_number = int(
                decision_number
            )

            replay_seq = int(
                replay_seq
            )

            expected_next = (
                self.current_decision
                + 1
            )

            if (
                decision_number
                != expected_next
            ):
                raise ValueError(
                    "Out-of-order decision. "
                    f"Expected {expected_next}, "
                    f"received "
                    f"{decision_number}"
                )

            if (
                decision_number
                not in self.rows
            ):
                raise ValueError(
                    "Decision outside "
                    "frozen schedule"
                )

            row = self.rows[
                decision_number
            ]

            if (
                replay_seq
                != row["replay_seq_int"]
            ):
                raise ValueError(
                    "replay_seq mismatch "
                    f"for decision "
                    f"{decision_number}: "
                    f"expected "
                    f"{row['replay_seq_int']}, "
                    f"received "
                    f"{replay_seq}"
                )

            phase_changed = (
                row["phase"]
                != self.current_phase
            )

            if phase_changed:

                self.transition_phase(
                    row
                )

            self.current_decision = (
                decision_number
            )

            result = {
                "ok": True,

                "decision_number":
                    decision_number,

                "replay_seq":
                    replay_seq,

                "phase":
                    self.current_phase,

                "phase_changed":
                    phase_changed,

                "target_worker_count":
                    self.current_target_workers,

                "cpu_pins":
                    self.current_pins,

                "dry_run":
                    self.dry_run,

                "active_worker_processes":
                    len(self.workers),

                "schedule_sha256":
                    self.schedule_sha256,
            }

            self.log_event(
                "decision_applied",
                **result,
            )

            return result


    def state(self):

        with self.lock:

            return {
                "ok": True,
                "ready": not self.closed,
                "dry_run":
                    self.dry_run,
                "current_decision":
                    self.current_decision,
                "current_phase":
                    self.current_phase,
                "target_worker_count":
                    self.current_target_workers,
                "cpu_pins":
                    self.current_pins,
                "active_worker_processes":
                    len(self.workers),
                "schedule_sha256":
                    self.schedule_sha256,
            }


    def close(self):

        with self.lock:

            if self.closed:
                return

            self.stop_worker_processes()

            self.closed = True

            self.log_event(
                "controller_stop",
                final_decision=
                    self.current_decision,
                final_phase=
                    self.current_phase,
                dry_run=
                    self.dry_run,
            )

            self.event_file.flush()
            self.event_file.close()


class Handler(BaseHTTPRequestHandler):

    server_version = (
        "S7LoadController/1.0"
    )

    def log_message(
        self,
        format,
        *args
    ):
        return


    def send_json(
        self,
        status,
        payload
    ):
        body = json.dumps(
            payload,
            sort_keys=True
        ).encode("utf-8")

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(
            body
        )


    def do_GET(self):

        if self.path in (
            "/health",
            "/state",
        ):

            self.send_json(
                200,
                self.server.controller.state()
            )

            return

        self.send_json(
            404,
            {
                "ok": False,
                "error": "not_found",
            }
        )


    def do_POST(self):

        if self.path == "/decision":

            try:

                length = int(
                    self.headers.get(
                        "Content-Length",
                        "0"
                    )
                )

                raw = self.rfile.read(
                    length
                )

                payload = json.loads(
                    raw.decode("utf-8")
                )

                result = (
                    self.server
                    .controller
                    .apply_decision(
                        payload[
                            "decision_number"
                        ],
                        payload[
                            "replay_seq"
                        ],
                    )
                )

                self.send_json(
                    200,
                    result
                )

            except ValueError as exc:

                self.send_json(
                    409,
                    {
                        "ok": False,
                        "error": str(exc),
                    }
                )

            except Exception as exc:

                self.send_json(
                    500,
                    {
                        "ok": False,
                        "error": repr(exc),
                    }
                )

            return


        if self.path == "/stop":

            self.send_json(
                200,
                {
                    "ok": True,
                    "stopping": True,
                }
            )

            def shutdown():

                try:
                    self.server.controller.close()

                finally:
                    self.server.shutdown()

            threading.Thread(
                target=shutdown,
                daemon=True,
            ).start()

            return


        self.send_json(
            404,
            {
                "ok": False,
                "error": "not_found",
            }
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--schedule",
        required=True,
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8002,
    )

    parser.add_argument(
        "--event-log",
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    controller = LoadController(
        schedule_path=
            args.schedule,
        dry_run=
            args.dry_run,
        event_log=
            args.event_log,
    )

    server = ThreadingHTTPServer(
        (
            args.host,
            args.port
        ),
        Handler,
    )

    server.controller = (
        controller
    )

    print(
        "S7_LOAD_CONTROLLER_READY",
        f"host={args.host}",
        f"port={args.port}",
        f"dry_run={args.dry_run}",
        f"schedule_sha256="
        f"{controller.schedule_sha256}",
        flush=True,
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        pass

    finally:

        controller.close()

        server.server_close()


if __name__ == "__main__":
    main()
