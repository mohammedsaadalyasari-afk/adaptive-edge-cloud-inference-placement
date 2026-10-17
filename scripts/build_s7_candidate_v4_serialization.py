#!/usr/bin/env python3

import copy
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_V3 = (
    "c685d1128aafc1380c3f0032c1fba052"
    "867e49fbee8bb30df7e89264a0c82a1a"
)

SAMPLER = "7a32aec94f54f3d4"
PREPARE = "7000000000000010"

RESULT_FILE = "7000000000000033"
PROBE_FILE = "7000000000000042"

GATE = "7000000000000050"
RELEASE = "7000000000000051"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canon(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


if len(sys.argv) != 4:
    raise SystemExit(
        "usage: patcher SOURCE_V3 TARGET_V4 MANIFEST"
    )


source = Path(sys.argv[1])
target = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])


source_bytes = source.read_bytes()

source_sha = sha(source_bytes)

if source_sha != EXPECTED_V3:
    raise SystemExit(
        "Candidate V3 SHA mismatch: "
        + source_sha
    )


nodes = json.loads(
    source_bytes.decode("utf-8")
)

by_id = {
    str(n["id"]): n
    for n in nodes
    if "id" in n
}


required = [
    SAMPLER,
    PREPARE,
    RESULT_FILE,
    PROBE_FILE,
]

for node_id in required:
    if node_id not in by_id:
        raise SystemExit(
            f"Required node missing: {node_id}"
        )


if GATE in by_id or RELEASE in by_id:
    raise SystemExit(
        "V4 node ID already exists"
    )


sampler = by_id[SAMPLER]
prepare = by_id[PREPARE]
result_file = by_id[RESULT_FILE]
probe_file = by_id[PROBE_FILE]


if sampler.get("type") != "function":
    raise SystemExit(
        "Sampler type mismatch"
    )

if prepare.get("type") != "function":
    raise SystemExit(
        "Prepare type mismatch"
    )

if result_file.get("type") != "file":
    raise SystemExit(
        "Result file node type mismatch"
    )

if probe_file.get("type") != "file":
    raise SystemExit(
        "Probe file node type mismatch"
    )


# ------------------------------------------------------------
# Verify exact V3 topology before patching.
# ------------------------------------------------------------

sampler_wires = sampler.get("wires")

if (
    not isinstance(sampler_wires, list)
    or len(sampler_wires) != 1
    or PREPARE not in sampler_wires[0]
):
    raise SystemExit(
        "Unexpected sampler wiring: "
        + repr(sampler_wires)
    )


if result_file.get("wires") != [[]]:
    raise SystemExit(
        "Unexpected result-file wiring: "
        + repr(result_file.get("wires"))
    )


if probe_file.get("wires") != [[]]:
    raise SystemExit(
        "Unexpected probe-file wiring: "
        + repr(probe_file.get("wires"))
    )


# ------------------------------------------------------------
# Serialization Gate
#
# Receives:
#   1) admitted deterministic sampler messages
#   2) completion-release messages
#
# Only one actual S7 logical decision may pass to Prepare
# at a time.
#
# Raw MQTT telemetry is NOT serialized here. Feature-state
# updates and deterministic sampling occur upstream exactly
# as in V3.
# ------------------------------------------------------------

gate_func = r'''msg = msg || {};

let queue =
    context.get(
        "s7_serial_queue"
    );

if (!Array.isArray(queue)) {
    queue = [];
}

let busy =
    context.get(
        "s7_serial_busy"
    ) === true;

let activeSeq =
    Number(
        context.get(
            "s7_serial_active_replay_seq"
        )
    );


function saveState() {

    context.set(
        "s7_serial_queue",
        queue
    );

    context.set(
        "s7_serial_busy",
        busy
    );

    context.set(
        "s7_serial_active_replay_seq",
        Number.isFinite(activeSeq)
            ? activeSeq
            : null
    );

    context.set(
        "s7_serial_queue_depth",
        queue.length
    );
}


/*
 * Completion/release path.
 */
if (
    msg._s7_serialization_release
    === true
) {

    const releaseSeq =
        Number(
            msg.s7 &&
            msg.s7.replay_seq
        );

    if (!busy) {

        node.error(
            "S7 serialization release received while gate is idle",
            msg
        );

        return null;
    }

    if (
        !Number.isFinite(releaseSeq) ||
        !Number.isFinite(activeSeq) ||
        releaseSeq !== activeSeq
    ) {

        node.error(
            "S7 serialization release replay_seq mismatch",
            msg
        );

        return null;
    }

    busy = false;
    activeSeq = NaN;

    if (queue.length === 0) {

        saveState();

        return null;
    }

    const next =
        queue.shift();

    const nextSeq =
        Number(
            next.meta &&
            next.meta.replay_seq
        );

    if (!Number.isFinite(nextSeq)) {

        node.error(
            "S7 queued message missing replay_seq",
            next
        );

        /*
         * Fail closed. Do not release another message
         * because queue identity is no longer trustworthy.
         */

        saveState();

        return null;
    }

    busy = true;
    activeSeq = nextSeq;

    saveState();

    return next;
}


/*
 * New logical inference candidate from deterministic sampler.
 */
const seq =
    Number(
        msg.meta &&
        msg.meta.replay_seq
    );

if (!Number.isFinite(seq)) {

    node.error(
        "S7 serialization candidate missing replay_seq",
        msg
    );

    return null;
}


/*
 * Explicit replay reset.
 *
 * This mirrors the existing replay_seq=1 reset semantics
 * used elsewhere in the frozen S7 flow.
 */
if (seq === 1) {

    queue = [];
    busy = false;
    activeSeq = NaN;

    context.set(
        "s7_serial_last_released_replay_seq",
        null
    );
}


if (busy) {

    queue.push(msg);

    saveState();

    return null;
}


busy = true;
activeSeq = seq;

saveState();

return msg;
'''


# ------------------------------------------------------------
# Serialization Release
#
# Inputs are AFTER the CSV file nodes:
#   - normal performance CSV completion
#   - probe CSV completion
#
# For decisions not divisible by 10:
#   actual CSV completion releases next logical decision.
#
# For decisions divisible by 10:
#   actual CSV completion does NOT release;
#   the due probe must complete, update state, be logged,
#   and reach the probe-file output first.
# ------------------------------------------------------------

release_func = r'''msg.s7 =
    msg.s7 || {};

const n =
    Number(
        msg.s7.decision_number
    );

const seq =
    Number(
        msg.s7.replay_seq
    );

if (
    !Number.isFinite(n) ||
    !Number.isFinite(seq)
) {

    node.error(
        "S7 serialization completion missing identity",
        msg
    );

    return null;
}


const isProbe =
    msg.s7.attempt_kind
    === "probe";

const probeDue =
    n % 10 === 0;


/*
 * Probe-due logical decisions remain locked after
 * their actual performance row is written.
 *
 * Release occurs only after the probe response has
 * updated EWMA/health state and the probe CSV file
 * node has completed.
 */
if (
    probeDue &&
    !isProbe
) {

    return null;
}


/*
 * A probe on a non-probe decision indicates an
 * unexpected control-flow condition.
 */
if (
    !probeDue &&
    isProbe
) {

    node.error(
        "Unexpected S7 probe completion on non-probe decision",
        msg
    );

    return null;
}


msg._s7_serialization_release =
    true;


/*
 * Preserve only the identity required by the gate.
 * The completed decision data have already been
 * written to their respective evidence CSV.
 */
msg.payload = {
    serialization_release: true,
    decision_number: n,
    replay_seq: seq,
    release_after:
        isProbe
        ? "probe_csv"
        : "actual_csv"
};

return msg;
'''


# ------------------------------------------------------------
# Build new function nodes by cloning an existing
# standard Function node so all Node-RED structural fields
# remain valid for this flow version.
# ------------------------------------------------------------

gate = copy.deepcopy(prepare)

gate["id"] = GATE
gate["name"] = "S7 Serialization Gate"
gate["func"] = gate_func
gate["outputs"] = 1
gate["wires"] = [[PREPARE]]

gate["x"] = (
    int(prepare.get("x", 1200))
    - 180
)

gate["y"] = (
    int(prepare.get("y", 400))
    + 100
)


release = copy.deepcopy(prepare)

release["id"] = RELEASE
release["name"] = "S7 Serialization Release"
release["func"] = release_func
release["outputs"] = 1
release["wires"] = [[GATE]]

release["x"] = (
    int(prepare.get("x", 1200))
    + 900
)

release["y"] = (
    int(prepare.get("y", 400))
    + 300
)


# ------------------------------------------------------------
# Patch only the three existing wiring locations.
# ------------------------------------------------------------

new_sampler_outputs = []

for dst in sampler["wires"][0]:

    if dst == PREPARE:
        new_sampler_outputs.append(
            GATE
        )
    else:
        new_sampler_outputs.append(
            dst
        )

sampler["wires"] = [
    new_sampler_outputs
]


result_file["wires"] = [
    [RELEASE]
]

probe_file["wires"] = [
    [RELEASE]
]


nodes.append(gate)
nodes.append(release)


target_text = json.dumps(
    nodes,
    indent=4,
    ensure_ascii=False,
) + "\n"

target.write_text(
    target_text,
    encoding="utf-8",
)


target_sha = sha(
    target.read_bytes()
)


manifest = {
    "candidate":
        "S7_candidate_v4",

    "purpose":
        (
            "Serialize actual logical S7 decisions so "
            "decision N+1 cannot execute placement policy "
            "before decision N state updates complete"
        ),

    "source_candidate":
        "S7_candidate_v3",

    "source_candidate_sha256":
        source_sha,

    "target_candidate_sha256":
        target_sha,

    "serialization_invariant":
        "at_most_one_actual_logical_S7_decision_in_flight",

    "serialization_admission_point":
        (
            "after deterministic sampler and before "
            "Prepare Decision"
        ),

    "non_probe_release_point":
        "after actual experiment CSV file write",

    "probe_due_release_point":
        "after probe state update and probe CSV file write",

    "probe_in_serialization_barrier":
        True,

    "new_object_count":
        2,

    "new_object_ids": [
        GATE,
        RELEASE,
    ],

    "changed_existing_object_count":
        3,

    "changed_existing_object_ids": [
        SAMPLER,
        RESULT_FILE,
        PROBE_FILE,
    ],

    "existing_function_code_changed":
        False,

    "adaptive_policy_modified":
        False,

    "response_fallback_router_modified":
        False,

    "finalizer_modified":
        False,

    "experiment_logger_modified":
        False,

    "probe_router_modified":
        False,

    "probe_logger_modified":
        False,

    "prepare_decision_modified":
        False,

    "deterministic_sampler_algorithm_modified":
        False,

    "deterministic_sampler_wiring_modified":
        True,

    "model_modified":
        False,

    "workload_modified":
        False,

    "controller_modified":
        False,

    "MQTT_modified":
        False,

    "Candidate_V3_async_defect_addressed":
        True,

    "Node_RED_deployed":
        False,

    "MQTT_validation_traffic":
        False,

    "CPU_load_generated":
        False,

    "final_S7_measurement_started":
        False,
}


manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


print(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
)

print()
print(
    "S7_CANDIDATE_V4_BUILD=PASS"
)
