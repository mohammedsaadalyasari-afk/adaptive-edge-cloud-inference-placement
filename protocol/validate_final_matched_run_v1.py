#!/usr/bin/env python3

import csv
import json
import math
import statistics
import sys

from collections import Counter
from pathlib import Path


scenario = sys.argv[1]
actual_path = Path(sys.argv[2])

probe_arg = sys.argv[3]
probe_path = (
    None
    if probe_arg == "-"
    else Path(probe_arg)
)

schedule_path = Path(sys.argv[4])
publish_path = Path(sys.argv[5])
event_path = Path(sys.argv[6])
reference_path = Path(sys.argv[7])

expected_model = sys.argv[8]

outdir = Path(sys.argv[9])


def read_csv(path):

    with path.open(newline="") as f:

        reader = csv.DictReader(f)

        return (
            reader.fieldnames or [],
            list(reader)
        )


def truthy(v):

    return str(v).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def excluded(v):

    return str(v).strip().lower() in {
        "no",
        "0",
        "0.0",
        "false",
    }


def as_int(v):

    return int(float(v))


def finite_float(v):

    try:

        x = float(v)

        if math.isfinite(x):
            return x

    except Exception:
        pass

    return None


def percentile(values, p):

    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    k = (
        (len(values) - 1)
        *
        p
        /
        100.0
    )

    lo = math.floor(k)
    hi = math.ceil(k)

    if lo == hi:
        return values[lo]

    return (
        values[lo] * (hi-k)
        +
        values[hi] * (k-lo)
    )


def describe(values):

    if not values:
        return {"n": 0}

    return {
        "n":
            len(values),

        "mean":
            statistics.fmean(values),

        "median":
            statistics.median(values),

        "p95":
            percentile(values, 95),

        "p99":
            percentile(values, 99),

        "max":
            max(values),
    }


actual_fields, rows = read_csv(
    actual_path
)

schedule_fields, schedule = read_csv(
    schedule_path
)

publish_fields, publish = read_csv(
    publish_path
)

reference_fields, reference = read_csv(
    reference_path
)


if probe_path is not None:

    probe_fields, probes = read_csv(
        probe_path
    )

else:

    probe_fields = []
    probes = []


errors = []


# ============================================================
# FUNDAMENTAL COUNTS
# ============================================================

if len(rows) != 289:

    errors.append(
        f"Actual row count={len(rows)}, expected 289"
    )


if len(schedule) != 289:

    errors.append(
        f"Schedule count={len(schedule)}, expected 289"
    )


if len(reference) != 289:

    errors.append(
        f"Reference count={len(reference)}, expected 289"
    )


if len(publish) != 1396:

    errors.append(
        f"Publish count={len(publish)}, expected 1396"
    )


if scenario == "S7":

    if len(probes) != 28:

        errors.append(
            f"Probe count={len(probes)}, expected 28"
        )


# ============================================================
# RUN-ID CONSISTENCY
# ============================================================

run_ids = {
    r["run_id"]
    for r in rows
    if r.get("run_id")
}

if len(run_ids) != 1:

    errors.append(
        f"Expected one non-empty run_id, got {run_ids}"
    )


# ============================================================
# EXACT DECISION / REFERENCE IDENTITY
# ============================================================

decisions = [
    as_int(
        r["decision_number"]
    )
    for r in rows
]

expected_decisions = list(
    range(1, 290)
)


if decisions != expected_decisions:

    errors.append(
        "Decision numbers are not exactly 1..289"
    )


if len(set(decisions)) != 289:

    errors.append(
        "Duplicate decision numbers detected"
    )


for i, (r, s, ref) in enumerate(
    zip(
        rows,
        schedule,
        reference
    ),
    start=1
):

    comparisons = [
        (
            "decision_number",
            as_int(r["decision_number"]),
            as_int(s["decision_number"]),
        ),
        (
            "replay_seq",
            as_int(r["replay_seq"]),
            as_int(s["replay_seq"]),
        ),
        (
            "source_line",
            as_int(r["source_line"]),
            as_int(s["source_line"]),
        ),
        (
            "source_epoch_ms",
            as_int(r["source_epoch_ms"]),
            as_int(s["source_epoch_ms"]),
        ),
        (
            "load_phase",
            r["load_phase"],
            s["phase"],
        ),
        (
            "load_target_workers",
            as_int(
                r["load_target_workers"]
            ),
            as_int(
                s["worker_count"]
            ),
        ),
        (
            "reference_replay_seq",
            as_int(
                ref["replay_seq"]
            ),
            as_int(
                s["replay_seq"]
            ),
        ),
        (
            "reference_source_line",
            as_int(
                ref["source_line"]
            ),
            as_int(
                s["source_line"]
            ),
        ),
        (
            "reference_epoch",
            as_int(
                ref["source_epoch_ms"]
            ),
            as_int(
                s["source_epoch_ms"]
            ),
        ),
        (
            "raw_slots_bits",
            r["raw_slots_bits"],
            ref["raw_slots_bits"],
        ),
    ]


    for name, actual, expected in comparisons:

        if actual != expected:

            errors.append(
                f"Decision {i}: "
                f"{name}={actual!r}, "
                f"expected={expected!r}"
            )


# ============================================================
# PREDICTION + MODEL ARTIFACT EQUIVALENCE
# ============================================================

prediction_checks = 0
prediction_mismatches = 0

model_checks = 0
model_mismatches = 0


for i, (r, ref) in enumerate(
    zip(
        rows,
        reference
    ),
    start=1
):

    success = truthy(
        r["final_success"]
    )

    if not success:
        continue

    prediction_checks += 1

    expected_prediction = (
        ref["expected_prediction"]
    )


    if (
        r["prediction"]
        != expected_prediction
    ):

        prediction_mismatches += 1

        errors.append(
            f"Decision {i}: prediction mismatch"
        )


    model_checks += 1

    if (
        r["model_sha256"]
        != expected_model
    ):

        model_mismatches += 1

        errors.append(
            f"Decision {i}: model SHA mismatch"
        )


# ============================================================
# COMMON OUTCOME COUNTS
# ============================================================

selected = Counter(
    r["selected_site"]
    for r in rows
)

primary_sites = Counter(
    r["primary_attempt_site"]
    for r in rows
)

final_sites = Counter(
    r["final_execution_site"]
    for r in rows
)


successes = sum(
    truthy(
        r["final_success"]
    )
    for r in rows
)

failures = (
    len(rows)
    -
    successes
)


fallback_rows = [
    r
    for r in rows
    if truthy(
        r["fallback_used"]
    )
]

fallbacks = len(
    fallback_rows
)


switches = sum(
    truthy(
        r["switch_occurred"]
    )
    for r in rows
)


primary_failure_rows = [
    r
    for r in rows
    if str(
        r["primary_attempt_status"]
    ).strip().lower()
    == "failed"
]


primary_failures = len(
    primary_failure_rows
)


# ============================================================
# S5 STATIC EDGE CONTRACT
# ============================================================

if scenario == "S5":

    if selected != Counter(
        {"edge": 289}
    ):

        errors.append(
            f"S5 selected sites invalid: {selected}"
        )


    if primary_sites != Counter(
        {"edge": 289}
    ):

        errors.append(
            f"S5 primary sites invalid: {primary_sites}"
        )


    if switches != 0:

        errors.append(
            f"S5 switches={switches}, expected 0"
        )


    if fallbacks != 0:

        errors.append(
            f"S5 fallbacks={fallbacks}, expected 0"
        )


    bad_reasons = {
        r["switch_reason"]
        for r in rows
        if r["switch_reason"]
        != "static_edge_only"
    }


    if bad_reasons:

        errors.append(
            f"S5 invalid switch reasons: {bad_reasons}"
        )


    for i, r in enumerate(
        rows,
        start=1
    ):

        ok = truthy(
            r["final_success"]
        )

        site = (
            r["final_execution_site"]
            or ""
        )


        if ok and site != "edge":

            errors.append(
                f"S5 decision {i}: "
                "successful final site is not edge"
            )


        if not ok and site != "":

            errors.append(
                f"S5 decision {i}: "
                "failed request has final site"
            )


# ============================================================
# S6 STATIC CLOUD CONTRACT
# ============================================================

if scenario == "S6":

    if selected != Counter(
        {"cloud": 289}
    ):

        errors.append(
            f"S6 selected sites invalid: {selected}"
        )


    if primary_sites != Counter(
        {"cloud": 289}
    ):

        errors.append(
            f"S6 primary sites invalid: {primary_sites}"
        )


    if switches != 0:

        errors.append(
            f"S6 switches={switches}, expected 0"
        )


    if fallbacks != 0:

        errors.append(
            f"S6 fallbacks={fallbacks}, expected 0"
        )


    bad_reasons = {
        r["switch_reason"]
        for r in rows
        if r["switch_reason"]
        != "static_cloud_only"
    }


    if bad_reasons:

        errors.append(
            f"S6 invalid switch reasons: {bad_reasons}"
        )


    for i, r in enumerate(
        rows,
        start=1
    ):

        ok = truthy(
            r["final_success"]
        )

        site = (
            r["final_execution_site"]
            or ""
        )


        if ok and site != "cloud":

            errors.append(
                f"S6 decision {i}: "
                "successful final site is not cloud"
            )


        if not ok and site != "":

            errors.append(
                f"S6 decision {i}: "
                "failed request has final site"
            )


# ============================================================
# S7 ADAPTIVE CONTRACT
# ============================================================

probe_successes = 0
probe_failures = 0

cost_e2c = 0
cost_c2e = 0
failure_switches = 0


if scenario == "S7":

    for x in selected:

        if x not in {
            "edge",
            "cloud",
        }:

            errors.append(
                f"Invalid S7 selected site: {x!r}"
            )


    for i, r in enumerate(
        rows,
        start=1
    ):

        selected_site = (
            r["selected_site"]
            or ""
        )

        primary_site = (
            r["primary_attempt_site"]
            or ""
        )

        final_site = (
            r["final_execution_site"]
            or ""
        )

        ok = truthy(
            r["final_success"]
        )

        fb = truthy(
            r["fallback_used"]
        )


        if primary_site != selected_site:

            errors.append(
                f"S7 decision {i}: "
                "primary site differs from selected site"
            )


        if (
            ok
            and
            not fb
            and
            final_site != selected_site
        ):

            errors.append(
                f"S7 decision {i}: "
                "non-fallback final site mismatch"
            )


        if (
            ok
            and
            fb
            and
            final_site == selected_site
        ):

            errors.append(
                f"S7 decision {i}: "
                "fallback marked without site change"
            )


    cost_e2c = sum(
        r["switch_reason"]
        ==
        "cost_hysteresis_edge_to_cloud"
        for r in rows
    )


    cost_c2e = sum(
        r["switch_reason"]
        ==
        "cost_hysteresis_cloud_to_edge"
        for r in rows
    )


    failure_switches = sum(
        (
            r["switch_reason"].startswith(
                "failover_"
            )
            or
            r["switch_reason"].startswith(
                "health_failover_"
            )
        )
        for r in rows
    )


    expected_probe_decisions = list(
        range(
            10,
            281,
            10
        )
    )


    probe_decisions = [
        as_int(
            p["decision_number"]
        )
        for p in probes
    ]


    if (
        probe_decisions
        !=
        expected_probe_decisions
    ):

        errors.append(
            "S7 probe IDs are not exactly "
            "10,20,...,280"
        )


    probe_successes = sum(
        truthy(
            p["probe_success"]
        )
        for p in probes
    )

    probe_failures = (
        len(probes)
        -
        probe_successes
    )


    for idx, p in enumerate(
        probes,
        start=1
    ):

        n = as_int(
            p["decision_number"]
        )

        probe_site = (
            p["probe_site"]
            or ""
        )

        current_site = (
            p[
                "probe_for_current_site"
            ]
            or ""
        )


        if probe_site not in {
            "edge",
            "cloud",
        }:

            errors.append(
                f"Probe {idx}: invalid site"
            )


        if current_site not in {
            "edge",
            "cloud",
        }:

            errors.append(
                f"Probe {idx}: invalid current site"
            )


        if probe_site == current_site:

            errors.append(
                f"Probe {idx}: probe site equals current site"
            )


        if not excluded(
            p[
                "included_in_performance_dataset"
            ]
        ):

            errors.append(
                f"Probe {idx}: "
                "incorrect performance-dataset flag"
            )


        # Probe replay identity must match its logical decision.
        actual_row = rows[
            n - 1
        ]


        if (
            as_int(
                p["replay_seq"]
            )
            !=
            as_int(
                actual_row[
                    "replay_seq"
                ]
            )
        ):

            errors.append(
                f"Probe {idx}: replay identity mismatch"
            )


        if truthy(
            p["probe_success"]
        ):

            expected_prediction = (
                reference[
                    n - 1
                ][
                    "expected_prediction"
                ]
            )


            if (
                p["prediction"]
                !=
                expected_prediction
            ):

                errors.append(
                    f"Probe decision {n}: "
                    "prediction mismatch"
                )


            if (
                p["model_sha256"]
                !=
                expected_model
            ):

                errors.append(
                    f"Probe decision {n}: "
                    "model SHA mismatch"
                )


# ============================================================
# LOAD PROFILE COUNTS
# ============================================================

phase_counts = Counter(
    r["load_phase"]
    for r in rows
)

expected_phase_counts = Counter(
    s["phase"]
    for s in schedule
)


worker_counts = Counter(
    as_int(
        r["load_target_workers"]
    )
    for r in rows
)

expected_worker_counts = Counter(
    as_int(
        s["worker_count"]
    )
    for s in schedule
)


if (
    phase_counts
    !=
    expected_phase_counts
):

    errors.append(
        "Load phase counts mismatch"
    )


if (
    worker_counts
    !=
    expected_worker_counts
):

    errors.append(
        "Worker-target counts mismatch"
    )


# ============================================================
# CONTROLLER EVENT LOG
# ============================================================

events = []

with event_path.open() as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        try:

            events.append(
                json.loads(line)
            )

        except Exception as e:

            errors.append(
                f"Invalid controller JSON: {e}"
            )


event_counts = Counter(
    e.get(
        "event",
        ""
    )
    for e in events
)


expected_event_counts = {
    "controller_start": 1,
    "phase_transition": 7,
    "decision_applied": 289,
    "controller_stop": 1,
}


for event_name, expected in expected_event_counts.items():

    actual = event_counts.get(
        event_name,
        0
    )

    if actual != expected:

        errors.append(
            f"Controller event {event_name}="
            f"{actual}, expected {expected}"
        )


# Validate controller decision order when fields are available.
applied = [
    e
    for e in events
    if e.get("event")
    == "decision_applied"
]


if len(applied) == 289:

    for i, (e, s) in enumerate(
        zip(
            applied,
            schedule
        ),
        start=1
    ):

        if "decision_number" in e:

            if (
                as_int(
                    e["decision_number"]
                )
                !=
                as_int(
                    s["decision_number"]
                )
            ):

                errors.append(
                    f"Controller decision {i}: "
                    "decision_number mismatch"
                )


        if "replay_seq" in e:

            if (
                as_int(
                    e["replay_seq"]
                )
                !=
                as_int(
                    s["replay_seq"]
                )
            ):

                errors.append(
                    f"Controller decision {i}: "
                    "replay_seq mismatch"
                )


# ============================================================
# LATENCY SUMMARIES
# ============================================================

total_all = []
total_success = []
service_success = []


for r in rows:

    total = finite_float(
        r["total_latency_ms"]
    )


    if total is not None:

        total_all.append(
            total
        )


    if truthy(
        r["final_success"]
    ):

        if total is not None:

            total_success.append(
                total
            )


        service = finite_float(
            r["service_inference_ms"]
        )


        if service is not None:

            service_success.append(
                service
            )


primary_failure_ids = [
    as_int(
        r["decision_number"]
    )
    for r in primary_failure_rows
]


fallback_ids = [
    as_int(
        r["decision_number"]
    )
    for r in fallback_rows
]


failed_ids = [
    as_int(
        r["decision_number"]
    )
    for r in rows
    if not truthy(
        r["final_success"]
    )
]


switch_reasons = Counter(
    r["switch_reason"]
    for r in rows
)


summary = {

    "scenario":
        scenario,

    "validator_version":
        "final_matched_run_v1",

    "logical_decisions":
        len(rows),

    "published_rows":
        len(publish),

    "run_ids":
        sorted(run_ids),

    "decision_order_pass":
        decisions
        ==
        expected_decisions,

    "selected_sites":
        dict(selected),

    "primary_sites":
        dict(primary_sites),

    "final_sites":
        dict(final_sites),

    "successful_decisions":
        successes,

    "failed_decisions":
        failures,

    "success_rate_percent":
        (
            successes
            /
            len(rows)
            *
            100.0
        )
        if rows
        else None,

    "primary_failures":
        primary_failures,

    "fallbacks":
        fallbacks,

    "primary_failure_ids":
        primary_failure_ids,

    "fallback_ids":
        fallback_ids,

    "failed_ids":
        failed_ids,

    "switches":
        switches,

    "switch_reason_counts":
        dict(
            switch_reasons
        ),

    "cost_edge_to_cloud":
        cost_e2c,

    "cost_cloud_to_edge":
        cost_c2e,

    "failure_related_switches":
        failure_switches,

    "probe_rows":
        len(probes),

    "probe_successes":
        probe_successes,

    "probe_failures":
        probe_failures,

    "phase_counts":
        dict(
            phase_counts
        ),

    "worker_target_counts":
        dict(
            worker_counts
        ),

    "controller_event_counts":
        dict(
            event_counts
        ),

    "prediction_checks":
        prediction_checks,

    "prediction_mismatches":
        prediction_mismatches,

    "model_sha_checks":
        model_checks,

    "model_sha_mismatches":
        model_mismatches,

    "total_latency_all":
        describe(
            total_all
        ),

    "total_latency_success":
        describe(
            total_success
        ),

    "service_latency_success":
        describe(
            service_success
        ),

    "validation_errors":
        errors,
}


summary_path = (
    outdir
    /
    "validation_summary.json"
)


summary_path.write_text(
    json.dumps(
        summary,
        indent=2
    )
)


print()
print("Scenario                :", scenario)

print(
    "Logical decisions       :",
    len(rows)
)

print(
    "Decision order 1..289   :",
    "PASS"
    if decisions
       == expected_decisions
    else "FAIL"
)

print(
    "Published MQTT rows     :",
    len(publish)
)

print(
    "Selected sites          :",
    dict(selected)
)

print(
    "Final sites             :",
    dict(final_sites)
)

print(
    "Successful decisions    :",
    successes
)

print(
    "Failed decisions        :",
    failures
)

print(
    "Success rate (%)        :",
    round(
        successes
        /
        len(rows)
        *
        100.0,
        3
    )
    if rows
    else None
)

print(
    "Primary failures        :",
    primary_failures
)

print(
    "Fallbacks               :",
    fallbacks
)

print(
    "Switches                :",
    switches
)


if scenario == "S7":

    print(
        "Cost E->C switches      :",
        cost_e2c
    )

    print(
        "Cost C->E switches      :",
        cost_c2e
    )

    print(
        "Failure-related switches:",
        failure_switches
    )

    print(
        "Probe rows              :",
        len(probes)
    )

    print(
        "Probe successes         :",
        probe_successes
    )

    print(
        "Probe failures          :",
        probe_failures
    )


print(
    "Phase counts            :",
    dict(
        phase_counts
    )
)

print(
    "Worker target counts    :",
    dict(
        worker_counts
    )
)

print(
    "Controller events       :",
    dict(
        event_counts
    )
)

print(
    "Prediction mismatches   :",
    prediction_mismatches
)

print(
    "Model SHA mismatches    :",
    model_mismatches
)

print()

print(
    "Total latency (all)     :",
    describe(
        total_all
    )
)

print(
    "Total latency (success) :",
    describe(
        total_success
    )
)

print(
    "Service latency success :",
    describe(
        service_success
    )
)

print()


if errors:

    print(
        "FINAL_RUN_VALIDATION=FAIL"
    )

    for e in errors[:100]:

        print(
            "ERROR:",
            e
        )

    raise SystemExit(1)


print(
    "FINAL_RUN_VALIDATION=PASS"
)
