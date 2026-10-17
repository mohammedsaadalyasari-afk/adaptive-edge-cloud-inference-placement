#!/usr/bin/env python3

import hashlib
import json
import math
import sys

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    sys.argv[1]
).resolve()

OUT = Path(
    sys.argv[2]
).resolve()

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

BOOTSTRAP_N = 50000
BOOTSTRAP_SEED = 20261010

PHASE_ORDER = [
    "L0_BASELINE",
    "L1_STABLE",
    "L2_ONSET",
    "L3_SATURATION",
    "L4_RECOVERY",
    "L5_REOVERLOAD",
    "L6_FINAL_RECOVERY",
]

STRATEGY_ORDER = [
    "S5",
    "S6",
    "S7",
]

DIR_MAP = {
    "S5":
        "S5_edge_only",

    "S6":
        "S6_cloud_only",

    "S7":
        "S7_adaptive",
}

MODEL_SHA = (
    "f400fb75d4ef47f0359129e4ac96cad"
    "49ae4c1694f33326014d54b977fe2855e"
)

ORDER_PATH = (
    ROOT
    /
    "protocol"
    /
    "final_run_order.csv"
)

REFERENCE_PATH = (
    ROOT
    /
    "frozen_inputs"
    /
    "deterministic_reference_with_predictions.csv"
)

SCHEDULE_PATH = (
    ROOT
    /
    "frozen_inputs"
    /
    "load_schedule.csv"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            h.update(
                chunk
            )

    return h.hexdigest()


def verify_hash_manifest(path):

    path = Path(
        path
    )

    if not path.exists():

        raise RuntimeError(
            f"Missing hash manifest: {path}"
        )

    checked = 0

    for raw in path.read_text().splitlines():

        raw = raw.strip()

        if not raw:
            continue

        parts = raw.split(
            maxsplit=1
        )

        if len(parts) != 2:

            raise RuntimeError(
                "Malformed SHA256 line:\n"
                f"{raw}"
            )

        expected = (
            parts[0]
            .strip()
        )

        file_path = Path(
            parts[1].strip()
        )

        if not file_path.exists():

            raise RuntimeError(
                "Hashed evidence file missing:\n"
                f"{file_path}"
            )

        actual = sha256_file(
            file_path
        )

        if actual != expected:

            raise RuntimeError(
                "Evidence hash mismatch:\n"
                f"{file_path}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            )

        checked += 1

    return checked


def truthy_series(series):

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "1",
                "true",
                "yes",
            }
        )
    )


def to_num(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def describe(values):

    a = np.asarray(
        pd.Series(
            values
        ).dropna(),
        dtype=float,
    )

    if len(a) == 0:

        return {
            "n":
                0,

            "mean":
                np.nan,

            "median":
                np.nan,

            "p95":
                np.nan,

            "p99":
                np.nan,

            "max":
                np.nan,
        }

    return {
        "n":
            int(
                len(a)
            ),

        "mean":
            float(
                np.mean(
                    a
                )
            ),

        "median":
            float(
                np.median(
                    a
                )
            ),

        "p95":
            float(
                np.quantile(
                    a,
                    0.95,
                    method="linear",
                )
            ),

        "p99":
            float(
                np.quantile(
                    a,
                    0.99,
                    method="linear",
                )
            ),

        "max":
            float(
                np.max(
                    a
                )
            ),
    }


def bootstrap_mean_ci(
    values,
    bootstrap_indices,
):

    a = np.asarray(
        values,
        dtype=float,
    )

    if len(a) != 10:

        raise RuntimeError(
            "Expected exactly 10 "
            "run-level observations"
        )

    boot = (
        a[
            bootstrap_indices
        ]
        .mean(
            axis=1
        )
    )

    return (
        float(
            np.quantile(
                boot,
                0.025,
            )
        ),

        float(
            np.quantile(
                boot,
                0.975,
            )
        ),
    )


def save_csv(
    dataframe,
    filename,
):

    path = (
        OUT
        /
        filename
    )

    dataframe.to_csv(
        path,
        index=False,
        float_format="%.6f",
    )

    return path


# ============================================================
# RECURSIVE CONTROLLER-EVENT EXTRACTION
# ============================================================

def recursive_values(
    obj,
    key_names,
):

    values = []

    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            if (
                key in key_names
                and
                not isinstance(
                    value,
                    (
                        dict,
                        list,
                    ),
                )
            ):

                values.append(
                    value
                )

            values.extend(
                recursive_values(
                    value,
                    key_names,
                )
            )

    elif isinstance(
        obj,
        list,
    ):

        for value in obj:

            values.extend(
                recursive_values(
                    value,
                    key_names,
                )
            )

    return values


def normalize_scalar(value):

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):

        return str(
            value
        ).lower()

    if isinstance(
        value,
        int,
    ):

        return int(
            value
        )

    if isinstance(
        value,
        float,
    ):

        if value.is_integer():

            return int(
                value
            )

        return float(
            value
        )

    text = str(
        value
    ).strip()

    try:

        number = float(
            text
        )

        if number.is_integer():

            return int(
                number
            )

    except Exception:
        pass

    return text


def unique_recursive_value(
    event,
    aliases,
):

    raw_values = recursive_values(
        event,
        aliases,
    )

    normalized = [
        normalize_scalar(
            value
        )
        for value
        in raw_values
        if value is not None
    ]

    unique = []

    for value in normalized:

        if value not in unique:

            unique.append(
                value
            )

    if not unique:

        return None

    if len(unique) > 1:

        raise RuntimeError(
            "Ambiguous controller event "
            f"values for {aliases}: "
            f"{unique}"
        )

    return unique[
        0
    ]


# ============================================================
# LOAD FROZEN INPUTS
# ============================================================

order = pd.read_csv(
    ORDER_PATH
)

schedule = pd.read_csv(
    SCHEDULE_PATH
)

reference = pd.read_csv(
    REFERENCE_PATH
)


if len(order) != 30:

    raise RuntimeError(
        f"Run order rows={len(order)}, "
        "expected 30"
    )


if len(schedule) != 289:

    raise RuntimeError(
        f"Schedule rows={len(schedule)}, "
        "expected 289"
    )


if len(reference) != 289:

    raise RuntimeError(
        f"Reference rows={len(reference)}, "
        "expected 289"
    )


for column in [
    "global_run",
    "block",
    "position",
    "repetition",
]:

    order[
        column
    ] = (
        order[
            column
        ]
        .astype(int)
    )


# ============================================================
# BOOTSTRAP INDEX — FROZEN
# ============================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

bootstrap_indices = rng.integers(
    0,
    10,
    size=(
        BOOTSTRAP_N,
        10,
    ),
)


# ============================================================
# MASTER COLLECTIONS
# ============================================================

all_decisions = []
all_probes = []

run_records = []
phase_records = []

s7_phase_records = []
failure_records = []

controller_audit_records = []
controller_identity_records = []

hash_entries_verified = 0


# ============================================================
# PROCESS ALL 30 FINAL RUNS
# ============================================================

for rec in order.itertuples(
    index=False
):

    scenario = str(
        rec.scenario
    )

    repetition = int(
        rec.repetition
    )

    block = int(
        rec.block
    )

    global_run = int(
        rec.global_run
    )

    position = int(
        rec.position
    )


    repdir = (
        ROOT
        /
        DIR_MAP[
            scenario
        ]
        /
        f"rep{repetition:02d}"
    )


    marker = (
        repdir
        /
        "RUN_COMPLETE.txt"
    )

    validation_path = (
        repdir
        /
        "validation_summary.json"
    )

    hash_manifest = (
        repdir
        /
        "RUN_SHA256SUMS.txt"
    )

    experiment_path = (
        repdir
        /
        f"{scenario}_experiment_log.csv"
    )

    controller_path = (
        repdir
        /
        "load_controller_events.jsonl"
    )


    # ========================================================
    # COMPLETION / VALIDATION EVIDENCE
    # ========================================================

    if not marker.exists():

        raise RuntimeError(
            "Missing completion marker: "
            f"{scenario} rep{repetition:02d}"
        )


    if (
        "RUN_STATUS=PASS"
        not in marker.read_text()
    ):

        raise RuntimeError(
            "Run not marked PASS: "
            f"{scenario} rep{repetition:02d}"
        )


    if not validation_path.exists():

        raise RuntimeError(
            "Missing validation summary: "
            f"{scenario} rep{repetition:02d}"
        )


    with validation_path.open() as f:

        validation = json.load(
            f
        )


    if validation.get(
        "validation_errors"
    ):

        raise RuntimeError(
            "Validation errors present: "
            f"{scenario} rep{repetition:02d}"
        )


    if (
        validation.get(
            "logical_decisions"
        )
        !=
        289
    ):

        raise RuntimeError(
            "Validation decision count "
            "not 289: "
            f"{scenario} rep{repetition:02d}"
        )


    # ========================================================
    # CRYPTOGRAPHIC EVIDENCE VERIFICATION
    # ========================================================

    hash_entries_verified += (
        verify_hash_manifest(
            hash_manifest
        )
    )


    # ========================================================
    # EXACT CONTROLLER-ORDER AUDIT
    # ========================================================

    controller_events = []

    with controller_path.open() as f:

        for raw in f:

            raw = raw.strip()

            if not raw:
                continue

            controller_events.append(
                json.loads(
                    raw
                )
            )


    controller_counts = Counter(
        event.get(
            "event",
            ""
        )
        for event
        in controller_events
    )


    expected_controller_counts = {
        "controller_start":
            1,

        "phase_transition":
            7,

        "decision_applied":
            289,

        "controller_stop":
            1,
    }


    for event_name, expected_count in (
        expected_controller_counts.items()
    ):

        actual_count = controller_counts.get(
            event_name,
            0,
        )

        if actual_count != expected_count:

            raise RuntimeError(
                f"{scenario} rep{repetition:02d}: "
                f"{event_name} count="
                f"{actual_count}, "
                f"expected={expected_count}"
            )


    applied = [
        event
        for event
        in controller_events
        if event.get(
            "event"
        )
        ==
        "decision_applied"
    ]


    source_line_verified = 0
    phase_verified = 0
    worker_verified = 0


    for index, (
        event,
        sched,
    ) in enumerate(
        zip(
            applied,
            schedule.to_dict(
                "records"
            ),
        ),
        start=1,
    ):

        decision_number = unique_recursive_value(
            event,
            {
                "decision_number",
            },
        )

        replay_seq = unique_recursive_value(
            event,
            {
                "replay_seq",
            },
        )


        if decision_number is None:

            raise RuntimeError(
                f"{scenario} rep{repetition:02d}: "
                "cannot extract decision_number "
                f"from controller event {index}"
            )


        if replay_seq is None:

            raise RuntimeError(
                f"{scenario} rep{repetition:02d}: "
                "cannot extract replay_seq "
                f"from controller event {index}"
            )


        if int(
            decision_number
        ) != int(
            sched[
                "decision_number"
            ]
        ):

            raise RuntimeError(
                f"{scenario} rep{repetition:02d}: "
                f"controller decision {index} "
                "decision_number mismatch"
            )


        if int(
            replay_seq
        ) != int(
            sched[
                "replay_seq"
            ]
        ):

            raise RuntimeError(
                f"{scenario} rep{repetition:02d}: "
                f"controller decision {index} "
                "replay_seq mismatch"
            )


        source_line = unique_recursive_value(
            event,
            {
                "source_line",
            },
        )


        phase = unique_recursive_value(
            event,
            {
                "phase",
            },
        )


        worker_count = unique_recursive_value(
            event,
            {
                "worker_count",
                "target_worker_count",
            },
        )


        source_ok = None
        phase_ok = None
        worker_ok = None


        if source_line is not None:

            source_line_verified += 1

            source_ok = (
                int(
                    source_line
                )
                ==
                int(
                    sched[
                        "source_line"
                    ]
                )
            )

            if not source_ok:

                raise RuntimeError(
                    f"{scenario} rep{repetition:02d}: "
                    f"controller decision {index} "
                    "source_line mismatch"
                )


        if phase is not None:

            phase_verified += 1

            phase_ok = (
                str(
                    phase
                )
                ==
                str(
                    sched[
                        "phase"
                    ]
                )
            )

            if not phase_ok:

                raise RuntimeError(
                    f"{scenario} rep{repetition:02d}: "
                    f"controller decision {index} "
                    "phase mismatch"
                )


        if worker_count is not None:

            worker_verified += 1

            worker_ok = (
                int(
                    worker_count
                )
                ==
                int(
                    sched[
                        "worker_count"
                    ]
                )
            )

            if not worker_ok:

                raise RuntimeError(
                    f"{scenario} rep{repetition:02d}: "
                    f"controller decision {index} "
                    "worker_count mismatch"
                )


        controller_identity_records.append(
            {
                "strategy":
                    scenario,

                "repetition":
                    repetition,

                "block":
                    block,

                "global_run":
                    global_run,

                "event_position":
                    index,

                "decision_number":
                    int(
                        decision_number
                    ),

                "expected_decision_number":
                    int(
                        sched[
                            "decision_number"
                        ]
                    ),

                "replay_seq":
                    int(
                        replay_seq
                    ),

                "expected_replay_seq":
                    int(
                        sched[
                            "replay_seq"
                        ]
                    ),

                "source_line":
                    source_line,

                "expected_source_line":
                    int(
                        sched[
                            "source_line"
                        ]
                    ),

                "phase":
                    phase,

                "expected_phase":
                    sched[
                        "phase"
                    ],

                "worker_count":
                    worker_count,

                "expected_worker_count":
                    int(
                        sched[
                            "worker_count"
                        ]
                    ),
            }
        )


    controller_audit_records.append(
        {
            "strategy":
                scenario,

            "repetition":
                repetition,

            "block":
                block,

            "global_run":
                global_run,

            "decision_applied_count":
                len(
                    applied
                ),

            "mandatory_decision_number_verified":
                289,

            "mandatory_replay_seq_verified":
                289,

            "source_line_verified_count":
                source_line_verified,

            "phase_verified_count":
                phase_verified,

            "worker_count_verified_count":
                worker_verified,

            "controller_order_pass":
                True,
        }
    )


    # ========================================================
    # EXPERIMENT LOG
    # ========================================================

    df = pd.read_csv(
        experiment_path
    )


    if len(df) != 289:

        raise RuntimeError(
            f"{scenario} rep{repetition:02d}: "
            f"rows={len(df)}, expected 289"
        )


    decisions = (
        to_num(
            df[
                "decision_number"
            ]
        )
        .astype(int)
        .tolist()
    )


    if decisions != list(
        range(
            1,
            290,
        )
    ):

        raise RuntimeError(
            f"{scenario} rep{repetition:02d}: "
            "decision order not 1..289"
        )


    # ========================================================
    # EXACT SCHEDULE IDENTITY
    # ========================================================

    for row_index in range(
        289
    ):

        row = df.iloc[
            row_index
        ]

        sched = schedule.iloc[
            row_index
        ]

        checks = [
            (
                int(
                    row[
                        "decision_number"
                    ]
                ),
                int(
                    sched[
                        "decision_number"
                    ]
                ),
                "decision_number",
            ),

            (
                int(
                    row[
                        "replay_seq"
                    ]
                ),
                int(
                    sched[
                        "replay_seq"
                    ]
                ),
                "replay_seq",
            ),

            (
                int(
                    row[
                        "source_line"
                    ]
                ),
                int(
                    sched[
                        "source_line"
                    ]
                ),
                "source_line",
            ),

            (
                int(
                    row[
                        "source_epoch_ms"
                    ]
                ),
                int(
                    sched[
                        "source_epoch_ms"
                    ]
                ),
                "source_epoch_ms",
            ),

            (
                str(
                    row[
                        "load_phase"
                    ]
                ),
                str(
                    sched[
                        "phase"
                    ]
                ),
                "load_phase",
            ),

            (
                int(
                    row[
                        "load_target_workers"
                    ]
                ),
                int(
                    sched[
                        "worker_count"
                    ]
                ),
                "worker_count",
            ),
        ]


        for actual, expected, name in checks:

            if actual != expected:

                raise RuntimeError(
                    f"{scenario} rep{repetition:02d} "
                    f"decision {row_index+1}: "
                    f"{name} mismatch"
                )


    # ========================================================
    # PRIMARY FAILURE-INCLUSIVE LATENCY
    # ========================================================

    total_latency = to_num(
        df[
            "total_latency_ms"
        ]
    )


    if total_latency.isna().any():

        bad_decisions = (
            df.loc[
                total_latency.isna(),
                "decision_number",
            ]
            .tolist()
        )

        raise RuntimeError(
            f"{scenario} rep{repetition:02d}: "
            "missing total_latency_ms at "
            f"{bad_decisions}"
        )


    success = truthy_series(
        df[
            "final_success"
        ]
    )


    primary_failed = (
        df[
            "primary_attempt_status"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(
            "failed"
        )
    )


    fallback = truthy_series(
        df[
            "fallback_used"
        ]
    )


    switched = truthy_series(
        df[
            "switch_occurred"
        ]
    )


    # ========================================================
    # MODEL/PREDICTION EQUIVALENCE
    # ========================================================

    successful_mask = (
        success.to_numpy()
    )


    if successful_mask.any():

        actual_predictions = (
            df.loc[
                success,
                "prediction",
            ]
            .astype(str)
            .to_numpy()
        )


        expected_predictions = (
            reference.loc[
                successful_mask,
                "expected_prediction",
            ]
            .astype(str)
            .to_numpy()
        )


        if not np.array_equal(
            actual_predictions,
            expected_predictions,
        ):

            raise RuntimeError(
                f"{scenario} rep{repetition:02d}: "
                "prediction mismatch"
            )


        model_values = (
            df.loc[
                success,
                "model_sha256",
            ]
            .astype(str)
        )


        if not model_values.eq(
            MODEL_SHA
        ).all():

            raise RuntimeError(
                f"{scenario} rep{repetition:02d}: "
                "model SHA mismatch"
            )


    # ========================================================
    # ADD RUN IDENTIFIERS
    # ========================================================

    df.insert(
        0,
        "strategy",
        scenario,
    )

    df.insert(
        1,
        "repetition",
        repetition,
    )

    df.insert(
        2,
        "global_run",
        global_run,
    )

    df.insert(
        3,
        "block",
        block,
    )

    df.insert(
        4,
        "position",
        position,
    )


    all_decisions.append(
        df
    )


    # ========================================================
    # RUN-LEVEL STATISTICS
    # ========================================================

    all_desc = describe(
        total_latency
    )


    successful_total_desc = describe(
        total_latency[
            success
        ]
    )


    service_success_desc = describe(
        to_num(
            df.loc[
                success,
                "service_inference_ms",
            ]
        )
    )


    selected_sites = Counter(
        df[
            "selected_site"
        ]
        .fillna("")
        .astype(str)
    )


    final_sites = Counter(
        df[
            "final_execution_site"
        ]
        .fillna("")
        .astype(str)
    )


    switch_reason = (
        df[
            "switch_reason"
        ]
        .fillna("")
        .astype(str)
    )


    switch_counts = Counter(
        switch_reason
    )


    failure_related_switch_n = int(
        switch_reason
        .str.startswith(
            (
                "failover_",
                "health_failover_",
            )
        )
        .sum()
    )


    run_records.append(
        {
            "strategy":
                scenario,

            "repetition":
                repetition,

            "block":
                block,

            "global_run":
                global_run,

            "position":
                position,

            "n_decisions":
                289,

            "success_n":
                int(
                    success.sum()
                ),

            "failure_n":
                int(
                    (
                        ~success
                    ).sum()
                ),

            "success_rate_pct":
                float(
                    success.mean()
                    *
                    100.0
                ),

            "primary_failure_n":
                int(
                    primary_failed.sum()
                ),

            "fallback_n":
                int(
                    fallback.sum()
                ),

            "fallback_success_n":
                int(
                    (
                        fallback
                        &
                        success
                    ).sum()
                ),

            "switch_n":
                int(
                    switched.sum()
                ),

            "selected_edge_n":
                int(
                    selected_sites.get(
                        "edge",
                        0,
                    )
                ),

            "selected_cloud_n":
                int(
                    selected_sites.get(
                        "cloud",
                        0,
                    )
                ),

            "selected_edge_pct":
                float(
                    selected_sites.get(
                        "edge",
                        0,
                    )
                    /
                    289
                    *
                    100.0
                ),

            "selected_cloud_pct":
                float(
                    selected_sites.get(
                        "cloud",
                        0,
                    )
                    /
                    289
                    *
                    100.0
                ),

            "final_edge_n":
                int(
                    final_sites.get(
                        "edge",
                        0,
                    )
                ),

            "final_cloud_n":
                int(
                    final_sites.get(
                        "cloud",
                        0,
                    )
                ),

            "cost_e2c_n":
                int(
                    switch_counts.get(
                        "cost_hysteresis_edge_to_cloud",
                        0,
                    )
                ),

            "cost_c2e_n":
                int(
                    switch_counts.get(
                        "cost_hysteresis_cloud_to_edge",
                        0,
                    )
                ),

            "failure_related_switch_n":
                failure_related_switch_n,

            "total_mean_ms":
                all_desc[
                    "mean"
                ],

            "total_median_ms":
                all_desc[
                    "median"
                ],

            "total_p95_ms":
                all_desc[
                    "p95"
                ],

            "total_p99_ms":
                all_desc[
                    "p99"
                ],

            "total_max_ms":
                all_desc[
                    "max"
                ],

            "success_total_mean_ms":
                successful_total_desc[
                    "mean"
                ],

            "success_total_median_ms":
                successful_total_desc[
                    "median"
                ],

            "success_total_p95_ms":
                successful_total_desc[
                    "p95"
                ],

            "success_total_p99_ms":
                successful_total_desc[
                    "p99"
                ],

            "service_success_mean_ms":
                service_success_desc[
                    "mean"
                ],

            "service_success_median_ms":
                service_success_desc[
                    "median"
                ],

            "service_success_p95_ms":
                service_success_desc[
                    "p95"
                ],

            "service_success_p99_ms":
                service_success_desc[
                    "p99"
                ],
        }
    )


    # ========================================================
    # PHASE-WISE RUN-LEVEL STATISTICS
    # ========================================================

    for phase in PHASE_ORDER:

        phase_data = df[
            df[
                "load_phase"
            ]
            .astype(str)
            .eq(
                phase
            )
        ].copy()


        if phase_data.empty:

            raise RuntimeError(
                f"{scenario} rep{repetition:02d}: "
                f"missing phase {phase}"
            )


        phase_latency = to_num(
            phase_data[
                "total_latency_ms"
            ]
        )


        phase_success = truthy_series(
            phase_data[
                "final_success"
            ]
        )


        phase_desc = describe(
            phase_latency
        )


        cpu_raw = to_num(
            phase_data[
                "edge_CPU_raw_percent"
            ]
        )


        cpu_ewma = to_num(
            phase_data[
                "edge_CPU_EWMA_percent"
            ]
        )


        selected_edge_pct = float(
            phase_data[
                "selected_site"
            ]
            .astype(str)
            .eq(
                "edge"
            )
            .mean()
            *
            100.0
        )


        selected_cloud_pct = float(
            phase_data[
                "selected_site"
            ]
            .astype(str)
            .eq(
                "cloud"
            )
            .mean()
            *
            100.0
        )


        phase_records.append(
            {
                "strategy":
                    scenario,

                "repetition":
                    repetition,

                "block":
                    block,

                "global_run":
                    global_run,

                "phase":
                    phase,

                "n":
                    int(
                        len(
                            phase_data
                        )
                    ),

                "total_mean_ms":
                    phase_desc[
                        "mean"
                    ],

                "total_median_ms":
                    phase_desc[
                        "median"
                    ],

                "total_p95_ms":
                    phase_desc[
                        "p95"
                    ],

                "total_p99_ms":
                    phase_desc[
                        "p99"
                    ],

                "success_rate_pct":
                    float(
                        phase_success.mean()
                        *
                        100.0
                    ),

                "failure_n":
                    int(
                        (
                            ~phase_success
                        ).sum()
                    ),

                "fallback_n":
                    int(
                        truthy_series(
                            phase_data[
                                "fallback_used"
                            ]
                        ).sum()
                    ),

                "switch_n":
                    int(
                        truthy_series(
                            phase_data[
                                "switch_occurred"
                            ]
                        ).sum()
                    ),

                "selected_edge_pct":
                    selected_edge_pct,

                "selected_cloud_pct":
                    selected_cloud_pct,

                "edge_cpu_raw_mean_pct":
                    float(
                        cpu_raw.mean()
                    )
                    if
                    cpu_raw.notna().any()
                    else
                    np.nan,

                "edge_cpu_ewma_mean_pct":
                    float(
                        cpu_ewma.mean()
                    )
                    if
                    cpu_ewma.notna().any()
                    else
                    np.nan,
            }
        )


        if scenario == "S7":

            s7_phase_records.append(
                {
                    "repetition":
                        repetition,

                    "block":
                        block,

                    "phase":
                        phase,

                    "n":
                        int(
                            len(
                                phase_data
                            )
                        ),

                    "selected_edge_n":
                        int(
                            phase_data[
                                "selected_site"
                            ]
                            .astype(str)
                            .eq(
                                "edge"
                            )
                            .sum()
                        ),

                    "selected_cloud_n":
                        int(
                            phase_data[
                                "selected_site"
                            ]
                            .astype(str)
                            .eq(
                                "cloud"
                            )
                            .sum()
                        ),

                    "selected_edge_pct":
                        selected_edge_pct,

                    "selected_cloud_pct":
                        selected_cloud_pct,

                    "switch_n":
                        int(
                            truthy_series(
                                phase_data[
                                    "switch_occurred"
                                ]
                            ).sum()
                        ),

                    "fallback_n":
                        int(
                            truthy_series(
                                phase_data[
                                    "fallback_used"
                                ]
                            ).sum()
                        ),

                    "primary_failure_n":
                        int(
                            phase_data[
                                "primary_attempt_status"
                            ]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                            .eq(
                                "failed"
                            )
                            .sum()
                        ),
                }
            )


    # ========================================================
    # FAILURE / FALLBACK EVENT TABLE
    # ========================================================

    interesting = (
        primary_failed
        |
        fallback
        |
        (
            ~success
        )
    )


    if interesting.any():

        columns = [
            "decision_number",
            "replay_seq",
            "source_line",
            "load_phase",
            "load_target_workers",
            "selected_site",
            "switch_reason",
            "primary_attempt_site",
            "primary_attempt_status",
            "primary_attempt_latency_ms",
            "primary_http_status",
            "fallback_used",
            "fallback_site",
            "fallback_status",
            "fallback_latency_ms",
            "final_success",
            "final_execution_site",
            "total_latency_ms",
            "prediction",
            "model_sha256",
        ]


        for _, failure_row in df.loc[
            interesting,
            columns,
        ].iterrows():

            record = failure_row.to_dict()

            record.update(
                {
                    "strategy":
                        scenario,

                    "repetition":
                        repetition,

                    "block":
                        block,

                    "global_run":
                        global_run,
                }
            )

            failure_records.append(
                record
            )


    # ========================================================
    # S7 PROBES
    # ========================================================

    if scenario == "S7":

        probe_path = (
            repdir
            /
            "S7_probe_log.csv"
        )


        probes = pd.read_csv(
            probe_path
        )


        if len(
            probes
        ) != 28:

            raise RuntimeError(
                f"S7 rep{repetition:02d}: "
                f"probe rows={len(probes)}, "
                "expected 28"
            )


        probe_decisions = (
            to_num(
                probes[
                    "decision_number"
                ]
            )
            .astype(int)
            .tolist()
        )


        expected_probe_decisions = list(
            range(
                10,
                281,
                10,
            )
        )


        if probe_decisions != expected_probe_decisions:

            raise RuntimeError(
                f"S7 rep{repetition:02d}: "
                "probe IDs mismatch"
            )


        excluded = (
            probes[
                "included_in_performance_dataset"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(
                {
                    "no",
                    "0",
                    "0.0",
                    "false",
                }
            )
        )


        if not excluded.all():

            raise RuntimeError(
                f"S7 rep{repetition:02d}: "
                "probe incorrectly included "
                "in primary performance data"
            )


        probes.insert(
            0,
            "strategy",
            "S7",
        )

        probes.insert(
            1,
            "repetition",
            repetition,
        )

        probes.insert(
            2,
            "global_run",
            global_run,
        )

        probes.insert(
            3,
            "block",
            block,
        )


        all_probes.append(
            probes
        )


# ============================================================
# CONTROLLER AUDIT OUTPUTS
# ============================================================

controller_audit_df = pd.DataFrame(
    controller_audit_records
)

controller_identity_df = pd.DataFrame(
    controller_identity_records
)


if len(
    controller_audit_df
) != 30:

    raise RuntimeError(
        "Controller audit does not "
        "contain 30 final runs"
    )


if not controller_audit_df[
    "controller_order_pass"
].all():

    raise RuntimeError(
        "Controller-order audit failed"
    )


if len(
    controller_identity_df
) != (
    30
    *
    289
):

    raise RuntimeError(
        "Controller identity table "
        f"rows={len(controller_identity_df)}, "
        "expected 8670"
    )


save_csv(
    controller_audit_df,
    "00_controller_order_audit_summary.csv",
)

save_csv(
    controller_identity_df,
    "00b_controller_decision_identity.csv",
)


# ============================================================
# MASTER DATASETS
# ============================================================

master = pd.concat(
    all_decisions,
    ignore_index=True,
)


probe_master = pd.concat(
    all_probes,
    ignore_index=True,
)


run_df = (
    pd.DataFrame(
        run_records
    )
    .sort_values(
        [
            "block",
            "strategy",
        ]
    )
    .reset_index(
        drop=True
    )
)


phase_df = pd.DataFrame(
    phase_records
)


s7_phase_df = pd.DataFrame(
    s7_phase_records
)


failure_df = pd.DataFrame(
    failure_records
)


if len(
    master
) != 8670:

    raise RuntimeError(
        f"Master rows={len(master)}, "
        "expected 8670"
    )


for strategy in STRATEGY_ORDER:

    count = int(
        (
            master[
                "strategy"
            ]
            ==
            strategy
        ).sum()
    )

    if count != 2890:

        raise RuntimeError(
            f"{strategy} rows={count}, "
            "expected 2890"
        )


if len(
    probe_master
) != 280:

    raise RuntimeError(
        f"S7 probes={len(probe_master)}, "
        "expected 280"
    )


save_csv(
    master,
    "01_master_decisions.csv",
)

save_csv(
    probe_master,
    "02_master_S7_probes.csv",
)

save_csv(
    run_df,
    "03_run_level_metrics.csv",
)

save_csv(
    phase_df,
    "04_phase_run_level_metrics.csv",
)

save_csv(
    s7_phase_df,
    "05_S7_phase_behavior_by_run.csv",
)

save_csv(
    failure_df,
    "06_failure_and_recovery_events.csv",
)


# ============================================================
# RUN-LEVEL STRATEGY SUMMARY
# ============================================================

summary_metrics = [
    (
        "total_mean_ms",
        "ms",
    ),

    (
        "total_median_ms",
        "ms",
    ),

    (
        "total_p95_ms",
        "ms",
    ),

    (
        "total_p99_ms",
        "ms",
    ),

    (
        "success_rate_pct",
        "percent",
    ),

    (
        "success_total_mean_ms",
        "ms",
    ),

    (
        "success_total_median_ms",
        "ms",
    ),

    (
        "success_total_p95_ms",
        "ms",
    ),

    (
        "success_total_p99_ms",
        "ms",
    ),

    (
        "service_success_mean_ms",
        "ms",
    ),
]


strategy_summary_records = []


for strategy in STRATEGY_ORDER:

    data = (
        run_df[
            run_df[
                "strategy"
            ]
            ==
            strategy
        ]
        .sort_values(
            "block"
        )
    )


    if len(
        data
    ) != 10:

        raise RuntimeError(
            f"{strategy} run rows="
            f"{len(data)}, expected 10"
        )


    for metric, unit in summary_metrics:

        values = (
            data[
                metric
            ]
            .to_numpy(
                dtype=float
            )
        )


        ci_low, ci_high = (
            bootstrap_mean_ci(
                values,
                bootstrap_indices,
            )
        )


        strategy_summary_records.append(
            {
                "strategy":
                    strategy,

                "metric":
                    metric,

                "unit":
                    unit,

                "n_runs":
                    10,

                "mean_across_runs":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "sd_across_runs":
                    float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    ),

                "median_across_runs":
                    float(
                        np.median(
                            values
                        )
                    ),

                "min_run":
                    float(
                        np.min(
                            values
                        )
                    ),

                "max_run":
                    float(
                        np.max(
                            values
                        )
                    ),

                "bootstrap95_mean_low":
                    ci_low,

                "bootstrap95_mean_high":
                    ci_high,
            }
        )


strategy_summary_df = pd.DataFrame(
    strategy_summary_records
)


save_csv(
    strategy_summary_df,
    "07_strategy_runlevel_summary.csv",
)


# ============================================================
# POOLED DESCRIPTIVES — NOT INFERENTIAL
# ============================================================

pooled_records = []


for strategy in STRATEGY_ORDER:

    data = master[
        master[
            "strategy"
        ]
        ==
        strategy
    ]


    total = to_num(
        data[
            "total_latency_ms"
        ]
    )


    success = truthy_series(
        data[
            "final_success"
        ]
    )


    all_desc = describe(
        total
    )


    success_desc = describe(
        total[
            success
        ]
    )


    service_desc = describe(
        to_num(
            data.loc[
                success,
                "service_inference_ms",
            ]
        )
    )


    pooled_records.append(
        {
            "strategy":
                strategy,

            "n_decisions":
                int(
                    len(
                        data
                    )
                ),

            "success_n":
                int(
                    success.sum()
                ),

            "failure_n":
                int(
                    (
                        ~success
                    ).sum()
                ),

            "pooled_success_rate_pct":
                float(
                    success.mean()
                    *
                    100.0
                ),

            "total_mean_ms":
                all_desc[
                    "mean"
                ],

            "total_median_ms":
                all_desc[
                    "median"
                ],

            "total_p95_ms":
                all_desc[
                    "p95"
                ],

            "total_p99_ms":
                all_desc[
                    "p99"
                ],

            "total_max_ms":
                all_desc[
                    "max"
                ],

            "success_total_mean_ms":
                success_desc[
                    "mean"
                ],

            "success_total_median_ms":
                success_desc[
                    "median"
                ],

            "success_total_p95_ms":
                success_desc[
                    "p95"
                ],

            "success_total_p99_ms":
                success_desc[
                    "p99"
                ],

            "service_success_mean_ms":
                service_desc[
                    "mean"
                ],

            "service_success_median_ms":
                service_desc[
                    "median"
                ],

            "service_success_p95_ms":
                service_desc[
                    "p95"
                ],

            "service_success_p99_ms":
                service_desc[
                    "p99"
                ],
        }
    )


pooled_df = pd.DataFrame(
    pooled_records
)


save_csv(
    pooled_df,
    "08_pooled_descriptive_only.csv",
)


# ============================================================
# PAIRED BLOCK COMPARISONS
# ============================================================

latency_metrics = [
    "total_mean_ms",
    "total_median_ms",
    "total_p95_ms",
    "total_p99_ms",
]


paired_block_records = []
paired_inference_records = []


def compare_strategy_pair(
    target,
    baseline,
    label,
):

    target_df = (
        run_df[
            run_df[
                "strategy"
            ]
            ==
            target
        ]
        .set_index(
            "block"
        )
        .sort_index()
    )


    baseline_df = (
        run_df[
            run_df[
                "strategy"
            ]
            ==
            baseline
        ]
        .set_index(
            "block"
        )
        .sort_index()
    )


    expected_blocks = list(
        range(
            1,
            11,
        )
    )


    if (
        list(
            target_df.index
        )
        !=
        expected_blocks
    ):

        raise RuntimeError(
            f"{target} block pairing invalid"
        )


    if (
        list(
            baseline_df.index
        )
        !=
        expected_blocks
    ):

        raise RuntimeError(
            f"{baseline} block pairing invalid"
        )


    for metric in latency_metrics:

        target_values = (
            target_df[
                metric
            ]
            .to_numpy(
                dtype=float
            )
        )


        baseline_values = (
            baseline_df[
                metric
            ]
            .to_numpy(
                dtype=float
            )
        )


        differences = (
            target_values
            -
            baseline_values
        )


        reductions = (
            (
                baseline_values
                -
                target_values
            )
            /
            baseline_values
            *
            100.0
        )


        bootstrap_differences = (
            differences[
                bootstrap_indices
            ]
            .mean(
                axis=1
            )
        )


        bootstrap_reductions = (
            reductions[
                bootstrap_indices
            ]
            .mean(
                axis=1
            )
        )


        for block_number in range(
            1,
            11,
        ):

            i = (
                block_number
                -
                1
            )


            paired_block_records.append(
                {
                    "comparison":
                        label,

                    "block":
                        block_number,

                    "metric":
                        metric,

                    "target_value":
                        float(
                            target_values[
                                i
                            ]
                        ),

                    "baseline_value":
                        float(
                            baseline_values[
                                i
                            ]
                        ),

                    "target_minus_baseline":
                        float(
                            differences[
                                i
                            ]
                        ),

                    "paired_reduction_pct":
                        float(
                            reductions[
                                i
                            ]
                        ),
                }
            )


        paired_inference_records.append(
            {
                "comparison":
                    label,

                "metric":
                    metric,

                "n_paired_runs":
                    10,

                "baseline_mean_across_runs":
                    float(
                        np.mean(
                            baseline_values
                        )
                    ),

                "target_mean_across_runs":
                    float(
                        np.mean(
                            target_values
                        )
                    ),

                "mean_paired_difference_target_minus_baseline":
                    float(
                        np.mean(
                            differences
                        )
                    ),

                "bootstrap95_difference_low":
                    float(
                        np.quantile(
                            bootstrap_differences,
                            0.025,
                        )
                    ),

                "bootstrap95_difference_high":
                    float(
                        np.quantile(
                            bootstrap_differences,
                            0.975,
                        )
                    ),

                "mean_paired_reduction_pct":
                    float(
                        np.mean(
                            reductions
                        )
                    ),

                "bootstrap95_reduction_low_pct":
                    float(
                        np.quantile(
                            bootstrap_reductions,
                            0.025,
                        )
                    ),

                "bootstrap95_reduction_high_pct":
                    float(
                        np.quantile(
                            bootstrap_reductions,
                            0.975,
                        )
                    ),

                "reduction_from_group_means_pct":
                    float(
                        (
                            np.mean(
                                baseline_values
                            )
                            -
                            np.mean(
                                target_values
                            )
                        )
                        /
                        np.mean(
                            baseline_values
                        )
                        *
                        100.0
                    ),

                "blocks_target_better":
                    int(
                        np.sum(
                            target_values
                            <
                            baseline_values
                        )
                    ),

                "blocks_target_worse":
                    int(
                        np.sum(
                            target_values
                            >
                            baseline_values
                        )
                    ),

                "blocks_tied":
                    int(
                        np.sum(
                            target_values
                            ==
                            baseline_values
                        )
                    ),
            }
        )


    # ========================================================
    # SUCCESS RATE — HIGHER IS BETTER
    # ========================================================

    metric = (
        "success_rate_pct"
    )


    target_values = (
        target_df[
            metric
        ]
        .to_numpy(
            dtype=float
        )
    )


    baseline_values = (
        baseline_df[
            metric
        ]
        .to_numpy(
            dtype=float
        )
    )


    differences = (
        target_values
        -
        baseline_values
    )


    bootstrap_differences = (
        differences[
            bootstrap_indices
        ]
        .mean(
            axis=1
        )
    )


    for block_number in range(
        1,
        11,
    ):

        i = (
            block_number
            -
            1
        )


        paired_block_records.append(
            {
                "comparison":
                    label,

                "block":
                    block_number,

                "metric":
                    metric,

                "target_value":
                    float(
                        target_values[
                            i
                        ]
                    ),

                "baseline_value":
                    float(
                        baseline_values[
                            i
                        ]
                    ),

                "target_minus_baseline":
                    float(
                        differences[
                            i
                        ]
                    ),

                "paired_reduction_pct":
                    np.nan,
            }
        )


    paired_inference_records.append(
        {
            "comparison":
                label,

            "metric":
                metric,

            "n_paired_runs":
                10,

            "baseline_mean_across_runs":
                float(
                    np.mean(
                        baseline_values
                    )
                ),

            "target_mean_across_runs":
                float(
                    np.mean(
                        target_values
                    )
                ),

            "mean_paired_difference_target_minus_baseline":
                float(
                    np.mean(
                        differences
                    )
                ),

            "bootstrap95_difference_low":
                float(
                    np.quantile(
                        bootstrap_differences,
                        0.025,
                    )
                ),

            "bootstrap95_difference_high":
                float(
                    np.quantile(
                        bootstrap_differences,
                        0.975,
                    )
                ),

            "mean_paired_reduction_pct":
                np.nan,

            "bootstrap95_reduction_low_pct":
                np.nan,

            "bootstrap95_reduction_high_pct":
                np.nan,

            "reduction_from_group_means_pct":
                np.nan,

            "blocks_target_better":
                int(
                    np.sum(
                        target_values
                        >
                        baseline_values
                    )
                ),

            "blocks_target_worse":
                int(
                    np.sum(
                        target_values
                        <
                        baseline_values
                    )
                ),

            "blocks_tied":
                int(
                    np.sum(
                        target_values
                        ==
                        baseline_values
                    )
                ),
        }
    )


compare_strategy_pair(
    "S7",
    "S6",
    "S7_vs_S6_PRIMARY",
)


compare_strategy_pair(
    "S7",
    "S5",
    "S7_vs_S5_SECONDARY",
)


paired_block_df = pd.DataFrame(
    paired_block_records
)


paired_inference_df = pd.DataFrame(
    paired_inference_records
)


save_csv(
    paired_block_df,
    "09_paired_block_values.csv",
)


save_csv(
    paired_inference_df,
    "10_paired_runlevel_inference.csv",
)


# ============================================================
# PHASE STRATEGY SUMMARY
# ============================================================

phase_summary_records = []


for strategy in STRATEGY_ORDER:

    for phase in PHASE_ORDER:

        data = (
            phase_df[
                (
                    phase_df[
                        "strategy"
                    ]
                    ==
                    strategy
                )
                &
                (
                    phase_df[
                        "phase"
                    ]
                    ==
                    phase
                )
            ]
            .sort_values(
                "repetition"
            )
        )


        if len(
            data
        ) != 10:

            raise RuntimeError(
                f"{strategy}/{phase}: "
                f"run count={len(data)}, "
                "expected 10"
            )


        for metric in [
            "total_mean_ms",
            "total_median_ms",
            "total_p95_ms",
            "success_rate_pct",
        ]:

            values = (
                data[
                    metric
                ]
                .to_numpy(
                    dtype=float
                )
            )


            ci_low, ci_high = (
                bootstrap_mean_ci(
                    values,
                    bootstrap_indices,
                )
            )


            phase_summary_records.append(
                {
                    "strategy":
                        strategy,

                    "phase":
                        phase,

                    "metric":
                        metric,

                    "n_runs":
                        10,

                    "mean_across_runs":
                        float(
                            np.mean(
                                values
                            )
                        ),

                    "sd_across_runs":
                        float(
                            np.std(
                                values,
                                ddof=1,
                            )
                        ),

                    "bootstrap95_mean_low":
                        ci_low,

                    "bootstrap95_mean_high":
                        ci_high,
                }
            )


phase_summary_df = pd.DataFrame(
    phase_summary_records
)


save_csv(
    phase_summary_df,
    "11_phase_strategy_runlevel_summary.csv",
)


# ============================================================
# S7 ADAPTIVE BEHAVIOR
# ============================================================

s7_runs = (
    run_df[
        run_df[
            "strategy"
        ]
        ==
        "S7"
    ]
    .sort_values(
        "repetition"
    )
    .copy()
)


s7_behavior_columns = [
    "repetition",
    "block",
    "global_run",
    "selected_edge_n",
    "selected_cloud_n",
    "selected_edge_pct",
    "selected_cloud_pct",
    "switch_n",
    "cost_e2c_n",
    "cost_c2e_n",
    "failure_related_switch_n",
    "primary_failure_n",
    "fallback_n",
    "fallback_success_n",
    "failure_n",
    "success_rate_pct",
]


save_csv(
    s7_runs[
        s7_behavior_columns
    ],
    "12_S7_behavior_by_run.csv",
)


total_s7_decisions = int(
    s7_runs[
        "n_decisions"
    ].sum()
)


selected_edge_total = int(
    s7_runs[
        "selected_edge_n"
    ].sum()
)


selected_cloud_total = int(
    s7_runs[
        "selected_cloud_n"
    ].sum()
)


primary_failures_total = int(
    s7_runs[
        "primary_failure_n"
    ].sum()
)


fallback_success_total = int(
    s7_runs[
        "fallback_success_n"
    ].sum()
)


probe_success = truthy_series(
    probe_master[
        "probe_success"
    ]
)


s7_behavior_summary = pd.DataFrame(
    [
        {
            "n_runs":
                10,

            "total_decisions":
                total_s7_decisions,

            "selected_edge_total":
                selected_edge_total,

            "selected_cloud_total":
                selected_cloud_total,

            "selected_edge_pct_all_decisions":
                float(
                    selected_edge_total
                    /
                    total_s7_decisions
                    *
                    100.0
                ),

            "selected_cloud_pct_all_decisions":
                float(
                    selected_cloud_total
                    /
                    total_s7_decisions
                    *
                    100.0
                ),

            "switches_total":
                int(
                    s7_runs[
                        "switch_n"
                    ].sum()
                ),

            "cost_e2c_total":
                int(
                    s7_runs[
                        "cost_e2c_n"
                    ].sum()
                ),

            "cost_c2e_total":
                int(
                    s7_runs[
                        "cost_c2e_n"
                    ].sum()
                ),

            "failure_related_switches_total":
                int(
                    s7_runs[
                        "failure_related_switch_n"
                    ].sum()
                ),

            "primary_failures_total":
                primary_failures_total,

            "fallbacks_total":
                int(
                    s7_runs[
                        "fallback_n"
                    ].sum()
                ),

            "fallback_successes_total":
                fallback_success_total,

            "final_failures_total":
                int(
                    s7_runs[
                        "failure_n"
                    ].sum()
                ),

            "fallback_recovery_rate_pct":
                (
                    float(
                        fallback_success_total
                        /
                        primary_failures_total
                        *
                        100.0
                    )
                    if
                    primary_failures_total
                    >
                    0
                    else
                    np.nan
                ),

            "probe_total":
                int(
                    len(
                        probe_master
                    )
                ),

            "probe_success_total":
                int(
                    probe_success.sum()
                ),

            "probe_failure_total":
                int(
                    (
                        ~probe_success
                    ).sum()
                ),
        }
    ]
)


save_csv(
    s7_behavior_summary,
    "13_S7_behavior_summary.csv",
)


# ============================================================
# S7 PHASE PLACEMENT SUMMARY
# ============================================================

s7_phase_summary_records = []


for phase in PHASE_ORDER:

    data = s7_phase_df[
        s7_phase_df[
            "phase"
        ]
        ==
        phase
    ]


    decisions_total = int(
        data[
            "n"
        ].sum()
    )


    edge_total = int(
        data[
            "selected_edge_n"
        ].sum()
    )


    cloud_total = int(
        data[
            "selected_cloud_n"
        ].sum()
    )


    s7_phase_summary_records.append(
        {
            "phase":
                phase,

            "n_runs":
                int(
                    data[
                        "repetition"
                    ].nunique()
                ),

            "decisions_total":
                decisions_total,

            "selected_edge_total":
                edge_total,

            "selected_cloud_total":
                cloud_total,

            "selected_edge_pct_all_decisions":
                float(
                    edge_total
                    /
                    decisions_total
                    *
                    100.0
                ),

            "selected_cloud_pct_all_decisions":
                float(
                    cloud_total
                    /
                    decisions_total
                    *
                    100.0
                ),

            "switches_total":
                int(
                    data[
                        "switch_n"
                    ].sum()
                ),

            "fallbacks_total":
                int(
                    data[
                        "fallback_n"
                    ].sum()
                ),

            "primary_failures_total":
                int(
                    data[
                        "primary_failure_n"
                    ].sum()
                ),
        }
    )


s7_phase_summary_df = pd.DataFrame(
    s7_phase_summary_records
)


save_csv(
    s7_phase_summary_df,
    "14_S7_phase_behavior_summary.csv",
)


# ============================================================
# MANUSCRIPT RUN-LEVEL TABLE
# ============================================================

manuscript_records = []


for strategy in STRATEGY_ORDER:

    data = run_df[
        run_df[
            "strategy"
        ]
        ==
        strategy
    ]


    def mean_sd(
        metric
    ):

        values = (
            data[
                metric
            ]
            .to_numpy(
                dtype=float
            )
        )

        return (
            float(
                np.mean(
                    values
                )
            ),

            float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
        )


    mean_latency, mean_latency_sd = mean_sd(
        "total_mean_ms"
    )

    median_latency, median_latency_sd = mean_sd(
        "total_median_ms"
    )

    p95_latency, p95_latency_sd = mean_sd(
        "total_p95_ms"
    )

    p99_latency, p99_latency_sd = mean_sd(
        "total_p99_ms"
    )

    success_rate, success_rate_sd = mean_sd(
        "success_rate_pct"
    )


    manuscript_records.append(
        {
            "strategy":
                strategy,

            "runs_n":
                10,

            "decisions_total":
                int(
                    data[
                        "n_decisions"
                    ].sum()
                ),

            "final_failures_total":
                int(
                    data[
                        "failure_n"
                    ].sum()
                ),

            "run_mean_latency_ms_mean":
                mean_latency,

            "run_mean_latency_ms_sd":
                mean_latency_sd,

            "run_median_latency_ms_mean":
                median_latency,

            "run_median_latency_ms_sd":
                median_latency_sd,

            "run_p95_latency_ms_mean":
                p95_latency,

            "run_p95_latency_ms_sd":
                p95_latency_sd,

            "run_p99_latency_ms_mean":
                p99_latency,

            "run_p99_latency_ms_sd":
                p99_latency_sd,

            "run_success_rate_pct_mean":
                success_rate,

            "run_success_rate_pct_sd":
                success_rate_sd,
        }
    )


manuscript_df = pd.DataFrame(
    manuscript_records
)


save_csv(
    manuscript_df,
    "15_manuscript_runlevel_table.csv",
)


# ============================================================
# FIGURES
# ============================================================

FIGDIR = (
    OUT
    /
    "figures"
)

FIGDIR.mkdir(
    exist_ok=True
)


figure_specs = [
    (
        "total_median_ms",
        "Run-level median total latency (ms)",
        "fig01_runlevel_median_latency",
    ),

    (
        "total_p95_ms",
        "Run-level P95 total latency (ms)",
        "fig02_runlevel_p95_latency",
    ),

    (
        "success_rate_pct",
        "Run-level final success rate (%)",
        "fig03_runlevel_success_rate",
    ),
]


for (
    metric,
    ylabel,
    stem,
) in figure_specs:

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            4.6,
        )
    )


    values = [
        run_df.loc[
            run_df[
                "strategy"
            ]
            ==
            strategy,
            metric,
        ].to_numpy(
            dtype=float
        )
        for strategy
        in STRATEGY_ORDER
    ]


    ax.boxplot(
        values,
        tick_labels=STRATEGY_ORDER,
        showmeans=True,
    )


    ax.set_xlabel(
        "Strategy"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )


    fig.tight_layout()


    fig.savefig(
        FIGDIR
        /
        f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )


    fig.savefig(
        FIGDIR
        /
        f"{stem}.pdf",
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# PHASE LATENCY FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8.0,
        4.8,
    )
)


for strategy in STRATEGY_ORDER:

    phase_values = []

    for phase in PHASE_ORDER:

        data = phase_df[
            (
                phase_df[
                    "strategy"
                ]
                ==
                strategy
            )
            &
            (
                phase_df[
                    "phase"
                ]
                ==
                phase
            )
        ]


        phase_values.append(
            float(
                data[
                    "total_median_ms"
                ].mean()
            )
        )


    ax.plot(
        PHASE_ORDER,
        phase_values,
        marker="o",
        label=strategy,
    )


ax.set_xlabel(
    "Externally imposed CPU-load phase"
)


ax.set_ylabel(
    "Mean run-level phase median latency (ms)"
)


ax.tick_params(
    axis="x",
    rotation=30,
)


ax.legend()


ax.grid(
    True,
    axis="y",
    alpha=0.25,
)


fig.tight_layout()


fig.savefig(
    FIGDIR
    /
    "fig04_phase_median_latency.png",
    dpi=300,
    bbox_inches="tight",
)


fig.savefig(
    FIGDIR
    /
    "fig04_phase_median_latency.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# S7 SITE-SELECTION FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8.0,
        4.8,
    )
)


x = np.arange(
    len(
        PHASE_ORDER
    )
)


edge = (
    s7_phase_summary_df[
        "selected_edge_pct_all_decisions"
    ]
    .to_numpy(
        dtype=float
    )
)


cloud = (
    s7_phase_summary_df[
        "selected_cloud_pct_all_decisions"
    ]
    .to_numpy(
        dtype=float
    )
)


ax.bar(
    x,
    edge,
    label="Edge",
)


ax.bar(
    x,
    cloud,
    bottom=edge,
    label="Cloud",
)


ax.set_xticks(
    x,
    PHASE_ORDER,
    rotation=30,
    ha="right",
)


ax.set_xlabel(
    "Externally imposed CPU-load phase"
)


ax.set_ylabel(
    "S7 selected-site share (%)"
)


ax.set_ylim(
    0,
    100,
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGDIR
    /
    "fig05_S7_site_selection_by_phase.png",
    dpi=300,
    bbox_inches="tight",
)


fig.savefig(
    FIGDIR
    /
    "fig05_S7_site_selection_by_phase.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# REPORT HELPERS
# ============================================================

def inference_row(
    comparison,
    metric,
):

    data = paired_inference_df[
        (
            paired_inference_df[
                "comparison"
            ]
            ==
            comparison
        )
        &
        (
            paired_inference_df[
                "metric"
            ]
            ==
            metric
        )
    ]


    if len(
        data
    ) != 1:

        raise RuntimeError(
            "Missing inference row: "
            f"{comparison}/{metric}"
        )


    return data.iloc[
        0
    ]


# ============================================================
# FINAL TEXT REPORT
# ============================================================

report = []

report.append(
    "# Final matched campaign analysis"
)

report.append(
    ""
)

report.append(
    "Inferential unit: matched experimental "
    "block/run (n=10 per strategy)."
)

report.append(
    "Pooled decision-level results are "
    "descriptive only."
)

report.append(
    "The primary latency estimand retains "
    "observed failure/timeout elapsed latency."
)

report.append(
    ""
)

report.append(
    "## Controller-order integrity"
)

report.append(
    ""
)

report.append(
    "All 30 final runs passed exact ordered "
    "controller decision_number and replay_seq "
    "verification against the frozen 289-row "
    "schedule."
)

report.append(
    ""
)

report.append(
    "## Strategy results"
)


for strategy in STRATEGY_ORDER:

    row = manuscript_df[
        manuscript_df[
            "strategy"
        ]
        ==
        strategy
    ].iloc[
        0
    ]


    report.append(
        f"- {strategy}: "
        f"mean={row['run_mean_latency_ms_mean']:.3f}"
        f"±{row['run_mean_latency_ms_sd']:.3f} ms; "
        f"median={row['run_median_latency_ms_mean']:.3f}"
        f"±{row['run_median_latency_ms_sd']:.3f} ms; "
        f"P95={row['run_p95_latency_ms_mean']:.3f}"
        f"±{row['run_p95_latency_ms_sd']:.3f} ms; "
        f"P99={row['run_p99_latency_ms_mean']:.3f}"
        f"±{row['run_p99_latency_ms_sd']:.3f} ms; "
        f"success={row['run_success_rate_pct_mean']:.3f}"
        f"±{row['run_success_rate_pct_sd']:.3f}%; "
        f"final failures="
        f"{int(row['final_failures_total'])}."
    )


report.append(
    ""
)

report.append(
    "## Primary paired comparison: S7 versus S6"
)


for metric in latency_metrics:

    row = inference_row(
        "S7_vs_S6_PRIMARY",
        metric,
    )


    report.append(
        f"- {metric}: paired reduction="
        f"{row['mean_paired_reduction_pct']:.3f}% "
        f"(95% bootstrap CI "
        f"{row['bootstrap95_reduction_low_pct']:.3f}% "
        f"to "
        f"{row['bootstrap95_reduction_high_pct']:.3f}%); "
        f"S7 lower in "
        f"{int(row['blocks_target_better'])}/10 blocks."
    )


success_row = inference_row(
    "S7_vs_S6_PRIMARY",
    "success_rate_pct",
)


report.append(
    "- Success-rate difference S7-S6="
    f"{success_row['mean_paired_difference_target_minus_baseline']:.3f} "
    "percentage points "
    "(95% bootstrap CI "
    f"{success_row['bootstrap95_difference_low']:.3f} "
    "to "
    f"{success_row['bootstrap95_difference_high']:.3f})."
)


report.append(
    ""
)

report.append(
    "## Secondary paired comparison: S7 versus S5"
)


for metric in latency_metrics:

    row = inference_row(
        "S7_vs_S5_SECONDARY",
        metric,
    )


    report.append(
        f"- {metric}: paired reduction="
        f"{row['mean_paired_reduction_pct']:.3f}% "
        f"(95% bootstrap CI "
        f"{row['bootstrap95_reduction_low_pct']:.3f}% "
        f"to "
        f"{row['bootstrap95_reduction_high_pct']:.3f}%); "
        f"S7 lower in "
        f"{int(row['blocks_target_better'])}/10 blocks."
    )


report.append(
    ""
)

report.append(
    "## S7 adaptive behavior"
)


s7_summary_row = (
    s7_behavior_summary
    .iloc[
        0
    ]
)


report.append(
    "- Edge selected "
    f"{int(s7_summary_row['selected_edge_total'])}/"
    f"{int(s7_summary_row['total_decisions'])} "
    f"({s7_summary_row['selected_edge_pct_all_decisions']:.3f}%); "
    "cloud selected "
    f"{int(s7_summary_row['selected_cloud_total'])}/"
    f"{int(s7_summary_row['total_decisions'])} "
    f"({s7_summary_row['selected_cloud_pct_all_decisions']:.3f}%)."
)


report.append(
    "- Switches="
    f"{int(s7_summary_row['switches_total'])}; "
    "cost E->C="
    f"{int(s7_summary_row['cost_e2c_total'])}; "
    "cost C->E="
    f"{int(s7_summary_row['cost_c2e_total'])}; "
    "failure-related="
    f"{int(s7_summary_row['failure_related_switches_total'])}."
)


report.append(
    "- Primary failures="
    f"{int(s7_summary_row['primary_failures_total'])}; "
    "fallbacks="
    f"{int(s7_summary_row['fallbacks_total'])}; "
    "fallback successes="
    f"{int(s7_summary_row['fallback_successes_total'])}; "
    "final failures="
    f"{int(s7_summary_row['final_failures_total'])}."
)


if not math.isnan(
    float(
        s7_summary_row[
            "fallback_recovery_rate_pct"
        ]
    )
):

    report.append(
        "- Fallback recovery among primary failures="
        f"{s7_summary_row['fallback_recovery_rate_pct']:.3f}%."
    )


report.append(
    "- S7 probes="
    f"{int(s7_summary_row['probe_success_total'])}/"
    f"{int(s7_summary_row['probe_total'])} successful; "
    "all excluded from primary performance latency."
)


REPORT_PATH = (
    OUT
    /
    "16_FINAL_ANALYSIS_REPORT.md"
)


REPORT_PATH.write_text(
    "\n".join(
        report
    )
    +
    "\n"
)


# ============================================================
# ANALYSIS METADATA
# ============================================================

metadata = {
    "analysis_version":
        "final_analysis_v1",

    "bootstrap_replications":
        BOOTSTRAP_N,

    "bootstrap_seed":
        BOOTSTRAP_SEED,

    "inferential_unit":
        "matched run/block",

    "n_runs_per_strategy":
        10,

    "n_matched_blocks":
        10,

    "n_decisions_per_run":
        289,

    "n_decisions_per_strategy":
        2890,

    "n_decisions_total":
        8670,

    "n_S7_probes_total":
        280,

    "primary_comparison":
        "S7 adaptive vs S6 cloud-only",

    "secondary_comparison":
        "S7 adaptive vs S5 edge-only",

    "primary_latency_estimand":
        (
            "failure-inclusive total_latency_ms "
            "over all logical decisions"
        ),

    "validation_runs_included":
        False,

    "S7_probes_in_primary_latency":
        False,

    "hash_manifest_entries_verified":
        hash_entries_verified,

    "controller_runs_audited":
        30,

    "controller_decision_events_audited":
        8670,

    "controller_order_audit_pass":
        True,
}


(
    OUT
    /
    "17_analysis_metadata.json"
).write_text(
    json.dumps(
        metadata,
        indent=2,
    )
    +
    "\n"
)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print()

print(
    "================================================"
)

print(
    "FINAL CONTROLLER-ORDER AUDIT"
)

print(
    "================================================"
)


print(
    "Final runs audited             :",
    len(
        controller_audit_df
    ),
)


print(
    "decision_applied events audited:",
    len(
        controller_identity_df
    ),
)


print(
    "decision_number exact order    : PASS"
)

print(
    "replay_seq exact order         : PASS"
)


print(
    "source_line verification events:",
    int(
        controller_audit_df[
            "source_line_verified_count"
        ].sum()
    ),
)


print(
    "phase verification events      :",
    int(
        controller_audit_df[
            "phase_verified_count"
        ].sum()
    ),
)


print(
    "worker verification events     :",
    int(
        controller_audit_df[
            "worker_count_verified_count"
        ].sum()
    ),
)


print()

print(
    "CONTROLLER_ORDER_AUDIT=PASS"
)


print()

print(
    "================================================"
)

print(
    "FINAL MATCHED ANALYSIS"
)

print(
    "================================================"
)


print(
    "Evidence hash entries verified :",
    hash_entries_verified,
)


print(
    "Master logical decisions       :",
    len(
        master
    ),
)


for strategy in STRATEGY_ORDER:

    count = int(
        (
            master[
                "strategy"
            ]
            ==
            strategy
        ).sum()
    )

    print(
        f"{strategy} decisions                   :",
        count,
    )


print(
    "S7 probes (excluded primary)   :",
    len(
        probe_master
    ),
)


print()

print(
    "RUN-LEVEL SUMMARY "
    "(mean ± SD across 10 runs)"
)


for strategy in STRATEGY_ORDER:

    row = manuscript_df[
        manuscript_df[
            "strategy"
        ]
        ==
        strategy
    ].iloc[
        0
    ]


    print(
        f"{strategy}: "
        f"mean="
        f"{row['run_mean_latency_ms_mean']:.3f}"
        f"±"
        f"{row['run_mean_latency_ms_sd']:.3f} ms | "
        f"median="
        f"{row['run_median_latency_ms_mean']:.3f}"
        f"±"
        f"{row['run_median_latency_ms_sd']:.3f} ms | "
        f"P95="
        f"{row['run_p95_latency_ms_mean']:.3f}"
        f"±"
        f"{row['run_p95_latency_ms_sd']:.3f} ms | "
        f"P99="
        f"{row['run_p99_latency_ms_mean']:.3f}"
        f"±"
        f"{row['run_p99_latency_ms_sd']:.3f} ms | "
        f"success="
        f"{row['run_success_rate_pct_mean']:.3f}"
        f"±"
        f"{row['run_success_rate_pct_sd']:.3f}% | "
        f"final_failures="
        f"{int(row['final_failures_total'])}"
    )


print()

print(
    "PRIMARY PAIRED COMPARISON: S7 vs S6"
)


for metric in latency_metrics:

    row = inference_row(
        "S7_vs_S6_PRIMARY",
        metric,
    )


    print(
        f"{metric}: "
        f"reduction="
        f"{row['mean_paired_reduction_pct']:.3f}% "
        f"[95% CI "
        f"{row['bootstrap95_reduction_low_pct']:.3f}, "
        f"{row['bootstrap95_reduction_high_pct']:.3f}] | "
        f"S7 lower in "
        f"{int(row['blocks_target_better'])}/10 blocks"
    )


success_row = inference_row(
    "S7_vs_S6_PRIMARY",
    "success_rate_pct",
)


print(
    "success rate S7-S6 = "
    f"{success_row['mean_paired_difference_target_minus_baseline']:.3f} pp "
    f"[95% CI "
    f"{success_row['bootstrap95_difference_low']:.3f}, "
    f"{success_row['bootstrap95_difference_high']:.3f}]"
)


print()

print(
    "SECONDARY PAIRED COMPARISON: S7 vs S5"
)


for metric in latency_metrics:

    row = inference_row(
        "S7_vs_S5_SECONDARY",
        metric,
    )


    print(
        f"{metric}: "
        f"reduction="
        f"{row['mean_paired_reduction_pct']:.3f}% "
        f"[95% CI "
        f"{row['bootstrap95_reduction_low_pct']:.3f}, "
        f"{row['bootstrap95_reduction_high_pct']:.3f}] | "
        f"S7 lower in "
        f"{int(row['blocks_target_better'])}/10 blocks"
    )


print()

print(
    "S7 ADAPTIVE BEHAVIOR"
)


print(
    "edge="
    f"{int(s7_summary_row['selected_edge_total'])}/"
    f"{int(s7_summary_row['total_decisions'])} "
    f"({s7_summary_row['selected_edge_pct_all_decisions']:.3f}%) | "
    "cloud="
    f"{int(s7_summary_row['selected_cloud_total'])}/"
    f"{int(s7_summary_row['total_decisions'])} "
    f"({s7_summary_row['selected_cloud_pct_all_decisions']:.3f}%)"
)


print(
    "switches="
    f"{int(s7_summary_row['switches_total'])} | "
    "cost E->C="
    f"{int(s7_summary_row['cost_e2c_total'])} | "
    "cost C->E="
    f"{int(s7_summary_row['cost_c2e_total'])} | "
    "failure-related="
    f"{int(s7_summary_row['failure_related_switches_total'])}"
)


print(
    "primary failures="
    f"{int(s7_summary_row['primary_failures_total'])} | "
    "fallbacks="
    f"{int(s7_summary_row['fallbacks_total'])} | "
    "fallback successes="
    f"{int(s7_summary_row['fallback_successes_total'])} | "
    "final failures="
    f"{int(s7_summary_row['final_failures_total'])}"
)


print(
    "S7 probes="
    f"{int(s7_summary_row['probe_success_total'])}/"
    f"{int(s7_summary_row['probe_total'])} successful"
)


print()

print(
    "Analysis directory:"
)

print(
    OUT
)


print()

print(
    "FINAL_ANALYSIS_COMPLETE=PASS"
)
