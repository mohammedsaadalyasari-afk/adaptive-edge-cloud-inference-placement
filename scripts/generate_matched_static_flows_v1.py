import copy
import hashlib
import json
from pathlib import Path

PROJECT = Path("/mnt/d/ESDE/Publications/ClusterComputing/Newimpimantation")
CAMPAIGN = PROJECT / "evidence/S5_S6_S7_LOAD_MATCHED_V1_20261008"
FLOWDIR = CAMPAIGN / "flows"
AUDIT = CAMPAIGN / "00_preexperiment_audit"

V4 = PROJECT / "s7/flows_S7_candidate_v4.json"

EXPECTED_V4 = (
    "172f0e3a134e1f26d6103413c3a80b5"
    "da2d53574b68c8ed97c1f345c8a55e267"
)

MODEL_SHA = (
    "f400fb75d4ef47f0359129e4ac96cad49"
    "ae4c1694f33326014d54b977fe2855e"
)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if sha256(V4) != EXPECTED_V4:
    raise SystemExit("STOP: frozen Candidate V4 hash mismatch")

source = json.loads(V4.read_text())


def index_nodes(flow):
    return {
        str(n.get("id")): n
        for n in flow
        if n.get("id") is not None
    }


REQUIRED = {
    "7000000000000001",
    "7000000000000010",
    "7000000000000011",
    "7000000000000020",
    "7000000000000021",
    "7000000000000022",
    "7000000000000030",
    "7000000000000031",
    "7000000000000032",
    "7000000000000033",
    "7000000000000040",
    "7000000000000041",
    "7000000000000042",
    "7000000000000050",
    "7000000000000051",
}

src_nodes = index_nodes(source)

missing = REQUIRED - set(src_nodes)
if missing:
    raise SystemExit(f"STOP: missing required V4 nodes: {sorted(missing)}")


PREPARE_TEMPLATE = r'''
msg.meta = msg.meta || {};

const SCENARIO = "__SCENARIO__";
const FORCED_SITE = "__SITE__";

const seq =
    Number(
        msg.meta.replay_seq
    );

if (!Number.isFinite(seq)) {
    node.error(
        SCENARIO + " missing replay_seq",
        msg
    );
    return null;
}

let n =
    Number(
        flow.get(
            "s7_decision_number"
        )
    );

if (seq === 1) {

    n = 0;

    /*
     * Internal flow-state keys retain the s7_ prefix
     * because the static flows are derived directly
     * from Candidate V4. They are implementation keys,
     * not claims of adaptive behavior.
     */

    flow.set(
        "s7_current_site",
        FORCED_SITE
    );

    flow.set(
        "s7_edge_healthy",
        true
    );

    flow.set(
        "s7_cloud_healthy",
        true
    );

    flow.set(
        "s7_L_edge_hat",
        191.113
    );

    flow.set(
        "s7_L_cloud_hat",
        486.772
    );

    flow.set(
        "s7_CPU_hat",
        null
    );

    flow.set(
        "s7_dwell_count",
        0
    );

    flow.set(
        "s7_e2c_count",
        0
    );

    flow.set(
        "s7_c2e_count",
        0
    );

    flow.set(
        "s7_logger_header_written",
        false
    );

    flow.set(
        "s7_probe_header_written",
        false
    );

    const rid =
        SCENARIO + "_" +
        new Date()
        .toISOString()
        .replace(
            /[:.]/g,
            "-"
        );

    flow.set(
        "s7_run_id",
        rid
    );
}

if (!Number.isFinite(n)) {
    n = 0;
}

n += 1;

if (
    n < 1 ||
    n > 289
) {
    node.error(
        SCENARIO + " decision outside 1..289",
        msg
    );
    return null;
}

flow.set(
    "s7_decision_number",
    n
);

const originalPayload =
    JSON.parse(
        JSON.stringify(
            msg.payload || {}
        )
    );

msg.s7 = {
    decision_number: n,
    replay_seq: seq,

    run_id:
        flow.get(
            "s7_run_id"
        ),

    ai_payload:
        originalPayload,

    scenario:
        SCENARIO,

    placement_policy:
        FORCED_SITE === "edge"
        ? "static_edge_only"
        : "static_cloud_only"
};

msg.payload = {
    decision_number: n,
    replay_seq: seq
};

msg.headers = {
    "Content-Type":
        "application/json"
};

msg.method = "POST";

delete msg.requestTimeout;
delete msg.statusCode;
delete msg.error;

return msg;
'''


POLICY_TEMPLATE = r'''
const perf =
    global.get("perf");

const SCENARIO = "__SCENARIO__";
const FORCED_SITE = "__SITE__";
const ALPHA = 0.20;

const ack =
    (
        msg.payload &&
        typeof msg.payload === "object"
    )
    ? msg.payload
    : {};

if (
    Number(msg.statusCode) !== 200 ||
    ack.ok !== true ||
    Number(ack.decision_number)
        !== Number(
            msg.s7.decision_number
        ) ||
    Number(ack.replay_seq)
        !== Number(
            msg.s7.replay_seq
        )
) {
    node.error(
        SCENARIO +
        " load-controller ACK invalid",
        msg
    );

    return null;
}

/*
 * Identical timing boundary to Candidate V4:
 * the externally controlled CPU-load transition
 * and /decision acknowledgement are excluded
 * from measured application inference latency.
 */
msg.s7.performance_t0_mono =
    perf.now();

msg.s7.load_phase =
    ack.phase || "";

msg.s7.load_phase_changed =
    ack.phase_changed === true;

msg.s7.load_target_workers =
    Number(
        ack.target_worker_count
    );

const cpuRaw =
    Number(
        flow.get(
            "s7_cpu_raw_percent"
        )
    );

const cpuTs =
    Number(
        flow.get(
            "s7_cpu_sample_ts"
        )
    );

if (
    !Number.isFinite(cpuRaw) ||
    !Number.isFinite(cpuTs) ||
    (Date.now() - cpuTs) > 2000
) {
    node.error(
        SCENARIO +
        " CPU sample missing or stale",
        msg
    );

    return null;
}

const cpuHatState =
    flow.get(
        "s7_CPU_hat"
    );

let cpuHat =
    (
        cpuHatState === null ||
        cpuHatState === undefined
    )
    ? NaN
    : Number(cpuHatState);

if (!Number.isFinite(cpuHat)) {
    cpuHat = cpuRaw;
}
else {
    cpuHat =
        ALPHA * cpuRaw +
        (1 - ALPHA) * cpuHat;
}

flow.set(
    "s7_CPU_hat",
    cpuHat
);

let Ledge =
    Number(
        flow.get(
            "s7_L_edge_hat"
        )
    );

let Lcloud =
    Number(
        flow.get(
            "s7_L_cloud_hat"
        )
    );

if (!Number.isFinite(Ledge)) {
    Ledge = 191.113;
}

if (!Number.isFinite(Lcloud)) {
    Lcloud = 486.772;
}

/*
 * Static baseline:
 * placement is forced irrespective of costs,
 * health state, dwell time, or superiority state.
 */
flow.set(
    "s7_current_site",
    FORCED_SITE
);

msg.s7.previous_site =
    FORCED_SITE;

msg.s7.selected_site =
    FORCED_SITE;

msg.s7.switch_occurred =
    false;

msg.s7.switch_reason =
    FORCED_SITE === "edge"
    ? "static_edge_only"
    : "static_cloud_only";

msg.s7.edge_CPU_raw_percent =
    cpuRaw;

msg.s7.edge_CPU_EWMA_percent =
    cpuHat;

/*
 * Keep latency estimates available for
 * instrumentation, but they do not participate
 * in the S5/S6 placement decision.
 */
msg.s7.edge_latency_EWMA_ms =
    Ledge;

msg.s7.cloud_latency_EWMA_ms =
    Lcloud;

msg.s7.CPU_penalty_ms =
    "";

msg.s7.edge_cost_ms =
    "";

msg.s7.cloud_cost_ms =
    "";

msg.s7.edge_healthy_decision =
    flow.get(
        "s7_edge_healthy"
    ) !== false;

msg.s7.cloud_healthy_decision =
    flow.get(
        "s7_cloud_healthy"
    ) !== false;

msg.s7.dwell_count =
    "";

msg.s7.superiority_count =
    "";

msg.s7.primary_attempt_site =
    FORCED_SITE;

msg.s7.primary_attempt_status =
    "";

msg.s7.primary_attempt_latency_ms =
    "";

msg.s7.primary_http_status =
    "";

msg.s7.fallback_used =
    false;

msg.s7.fallback_site =
    "";

msg.s7.fallback_status =
    "";

msg.s7.fallback_latency_ms =
    "";

msg.s7.attempt_kind =
    "primary";

msg.s7.attempt_site =
    FORCED_SITE;

msg.s7.attempt_t0_mono =
    perf.now();

msg.payload =
    JSON.parse(
        JSON.stringify(
            msg.s7.ai_payload
        )
    );

msg.headers = {
    "Content-Type":
        "application/json"
};

msg.method = "POST";

delete msg.statusCode;
delete msg.error;

if (FORCED_SITE === "cloud") {

    /*
     * Preserve the same cloud-request configuration
     * used by Candidate V4.
     */
    msg.requestTimeout =
        1500;

    return [
        null,
        msg
    ];
}

delete msg.requestTimeout;

return [
    msg,
    null
];
'''


RESPONSE_TEMPLATE = r'''
const perf =
    global.get("perf");

const SCENARIO = "__SCENARIO__";
const FORCED_SITE = "__SITE__";

const MODEL =
    "f400fb75d4ef47f0359129e4ac96cad49" +
    "ae4c1694f33326014d54b977fe2855e";

const ALPHA = 0.20;

msg.s7 =
    msg.s7 || {};

const site =
    msg.s7.attempt_site;

if (site !== FORCED_SITE) {

    node.error(
        SCENARIO +
        " received unexpected execution site: " +
        String(site),
        msg
    );

    return null;
}

const t0 =
    Number(
        msg.s7.attempt_t0_mono
    );

const attemptMs =
    Number.isFinite(t0)
    ? perf.now() - t0
    : null;

let r =
    msg.payload;

if (typeof r === "string") {

    try {
        r = JSON.parse(r);
    }
    catch (e) {
        r = {};
    }
}

if (
    !r ||
    typeof r !== "object"
) {
    r = {};
}

const status =
    Number(
        msg.statusCode
    );

const success =
    !msg.error &&
    status === 200 &&
    r.prediction !== undefined &&
    r.model_sha256 === MODEL &&
    r.execution_site === site;


function updateLatency(
    which,
    observation
) {

    const key =
        which === "edge"
        ? "s7_L_edge_hat"
        : "s7_L_cloud_hat";

    let before =
        Number(
            flow.get(key)
        );

    if (!Number.isFinite(before)) {

        before =
            which === "edge"
            ? 191.113
            : 486.772;
    }

    const after =
        ALPHA * observation +
        (1 - ALPHA) * before;

    flow.set(
        key,
        after
    );

    return after;
}


function setHealthy(
    which,
    value
) {

    flow.set(
        which === "edge"
        ? "s7_edge_healthy"
        : "s7_cloud_healthy",
        value
    );
}


msg.s7.primary_attempt_status =
    success
    ? "success"
    : "failed";

msg.s7.primary_attempt_latency_ms =
    Number.isFinite(attemptMs)
    ? attemptMs
    : "";

msg.s7.primary_http_status =
    Number.isFinite(status)
    ? status
    : "";

msg.s7.fallback_used =
    false;

msg.s7.fallback_site =
    "";

msg.s7.fallback_status =
    "";

msg.s7.fallback_latency_ms =
    "";


/*
 * The chosen static site is attempted once.
 * There is deliberately no fallback to the
 * alternative site in S5 or S6.
 */
if (success) {

    if (Number.isFinite(attemptMs)) {

        updateLatency(
            site,
            attemptMs
        );
    }

    setHealthy(
        site,
        true
    );

    msg.s7.final_success =
        true;

    msg.s7.final_execution_site =
        site;

    msg.s7.final_http_status =
        status;

    msg.s7.final_response =
        r;

    msg.payload =
        r;
}
else {

    setHealthy(
        site,
        false
    );

    /*
     * Retain Candidate-V4 cloud failure
     * observation semantics for instrumentation.
     * This does not affect static placement.
     */
    if (
        site === "cloud"
    ) {

        updateLatency(
            "cloud",
            1500
        );
    }

    msg.s7.final_success =
        false;

    msg.s7.final_execution_site =
        "";

    msg.s7.final_http_status =
        Number.isFinite(status)
        ? status
        : "";

    msg.s7.final_response =
        {};

    msg.payload =
        {};
}

/*
 * Output 1 only.
 * Outputs 2/3 are V4 fallback routes and are
 * deliberately disabled for static baselines.
 * Output 4 is the probe logger and is also disabled.
 */
return [
    msg,
    null,
    null,
    null
];
'''


RELEASE_TEMPLATE = r'''
msg.s7 =
    msg.s7 || {};

const SCENARIO =
    "__SCENARIO__";

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
        SCENARIO +
        " serialization completion missing identity",
        msg
    );

    return null;
}

/*
 * Static baselines do not execute alternative-site
 * probes. Therefore every completed actual CSV row
 * releases the serialization gate immediately.
 */
if (
    msg.s7.attempt_kind === "probe"
) {

    node.error(
        SCENARIO +
        " unexpected probe completion",
        msg
    );

    return null;
}

msg._s7_serialization_release =
    true;

msg.payload = {
    serialization_release: true,
    decision_number: n,
    replay_seq: seq,
    release_after:
        "actual_csv"
};

return msg;
'''


def build_static(scenario, site, outfile, logfile, unused_probe_log):

    flow = copy.deepcopy(source)
    nodes = index_nodes(flow)

    # ------------------------------------------------------------------
    # Minimal deliberate differences from frozen Candidate V4
    # ------------------------------------------------------------------

    nodes["7000000000000001"]["name"] = (
        f"{scenario} Load-Matched Static "
        f"{site.title()} Baseline"
    )

    n = nodes["7000000000000010"]
    n["name"] = f"{scenario} Prepare Decision"
    n["func"] = (
        PREPARE_TEMPLATE
        .replace("__SCENARIO__", scenario)
        .replace("__SITE__", site)
        .strip()
    )

    n = nodes["7000000000000020"]
    n["name"] = (
        f"{scenario} Forced "
        f"{site.title()} Placement"
    )
    n["func"] = (
        POLICY_TEMPLATE
        .replace("__SCENARIO__", scenario)
        .replace("__SITE__", site)
        .strip()
    )

    n = nodes["7000000000000030"]
    n["name"] = (
        f"{scenario} Static Response Router"
    )
    n["func"] = (
        RESPONSE_TEMPLATE
        .replace("__SCENARIO__", scenario)
        .replace("__SITE__", site)
        .strip()
    )

    # Preserve the V4 finalization/timing logic.
    # Only suppress its second output, which normally
    # requests the alternative-site probe.
    n = nodes["7000000000000031"]
    n["name"] = (
        f"{scenario} Finalize Actual Decision"
    )

    old_func = n["func"]

    needle = """return [
    msg,
    msg
];"""

    replacement = """return [
    msg,
    null
];"""

    if old_func.count(needle) != 1:
        raise RuntimeError(
            f"{scenario}: unable to identify "
            "Candidate-V4 finalizer return block"
        )

    n["func"] = old_func.replace(
        needle,
        replacement
    )

    n = nodes["7000000000000033"]
    n["name"] = (
        f"{scenario} Matched Experiment CSV"
    )
    n["filename"] = str(logfile)

    # Explicitly disable periodic probing.
    n = nodes["7000000000000040"]
    n["name"] = (
        f"{scenario} Probe Router DISABLED"
    )
    n["func"] = """/*
 * Static S5/S6 baseline:
 * non-selected-site probing is intentionally disabled.
 */
return [null, null];"""

    # Safety: even if an accidental message reaches the
    # unused probe path, it cannot contaminate S7 evidence.
    n = nodes["7000000000000042"]
    n["name"] = (
        f"{scenario} UNUSED Probe CSV"
    )
    n["filename"] = str(unused_probe_log)

    n = nodes["7000000000000051"]
    n["name"] = (
        f"{scenario} Serialization Release"
    )
    n["func"] = (
        RELEASE_TEMPLATE
        .replace("__SCENARIO__", scenario)
        .strip()
    )

    outfile.write_text(
        json.dumps(
            flow,
            indent=2,
            ensure_ascii=False
        ) + "\n"
    )

    return flow


s5_file = FLOWDIR / "flows_S5_matched_v1.json"
s6_file = FLOWDIR / "flows_S6_matched_v1.json"

s5 = build_static(
    "S5",
    "edge",
    s5_file,
    PROJECT / "logs/S5_matched/s5_experiment_log.csv",
    PROJECT / "logs/S5_matched/s5_probe_UNEXPECTED.csv",
)

s6 = build_static(
    "S6",
    "cloud",
    s6_file,
    PROJECT / "logs/S6_matched/s6_experiment_log.csv",
    PROJECT / "logs/S6_matched/s6_probe_UNEXPECTED.csv",
)


# ================================================================
# Structural verification
# ================================================================

ALLOWED_CHANGED_IDS = {
    "7000000000000001",
    "7000000000000010",
    "7000000000000020",
    "7000000000000030",
    "7000000000000031",
    "7000000000000033",
    "7000000000000040",
    "7000000000000042",
    "7000000000000051",
}


def changed_ids(base, derived):
    b = index_nodes(base)
    d = index_nodes(derived)

    all_ids = set(b) | set(d)

    return sorted(
        nid
        for nid in all_ids
        if b.get(nid) != d.get(nid)
    )


def verify_static(flow, scenario, forced_site, expected_log):

    n = index_nodes(flow)

    changes = set(
        changed_ids(
            source,
            flow
        )
    )

    if changes != ALLOWED_CHANGED_IDS:
        raise RuntimeError(
            f"{scenario}: unexpected changed node IDs\n"
            f"Expected: {sorted(ALLOWED_CHANGED_IDS)}\n"
            f"Actual  : {sorted(changes)}"
        )

    # Same load-controller path
    assert (
        n["7000000000000011"]["url"]
        == "http://127.0.0.1:8002/decision"
    )

    # Same edge inference endpoint
    assert (
        n["7000000000000021"]["url"]
        == "http://127.0.0.1:8001/predict_named"
    )

    # Same remote cloud endpoint
    assert (
        n["7000000000000022"]["url"]
        == "https://smart-parking-s6-cloud.onrender.com/predict_named"
    )

    # Correct static log path
    assert (
        n["7000000000000033"]["filename"]
        == str(expected_log)
    )

    policy = n["7000000000000020"]["func"]

    assert (
        f'const FORCED_SITE = "{forced_site}";'
        in policy
    )

    # No adaptive decision equations in static policy
    assert "cost_hysteresis" not in policy
    assert "e2c >=" not in policy
    assert "c2e >=" not in policy

    # No fallback in static response path
    response = n["7000000000000030"]["func"]
    assert "prepareAttempt(" not in response
    assert "fallback_used =\n    false" in response

    # Probe disabled
    probe = n["7000000000000040"]["func"]
    assert "return [null, null]" in probe

    # Finalizer must not emit a probe
    finalizer = n["7000000000000031"]["func"]
    assert """return [
    msg,
    null
];""" in finalizer

    # Every static actual row releases immediately.
    release = n["7000000000000051"]["func"]
    assert "n % 10" not in release
    assert 'release_after:\n        "actual_csv"' in release

    return sorted(changes)


s5_changes = verify_static(
    s5,
    "S5",
    "edge",
    PROJECT / "logs/S5_matched/s5_experiment_log.csv",
)

s6_changes = verify_static(
    s6,
    "S6",
    "cloud",
    PROJECT / "logs/S6_matched/s6_experiment_log.csv",
)


# Original V4 must still be immutable.
if sha256(V4) != EXPECTED_V4:
    raise RuntimeError(
        "FATAL: original Candidate V4 changed"
    )


manifest = AUDIT / "matched_flow_generation_manifest.txt"

with manifest.open("w") as f:

    f.write(
        f"source_v4_sha256={sha256(V4)}\n"
    )

    f.write(
        f"s5_matched_sha256={sha256(s5_file)}\n"
    )

    f.write(
        f"s6_matched_sha256={sha256(s6_file)}\n"
    )

    f.write(
        "s5_forced_site=edge\n"
    )

    f.write(
        "s6_forced_site=cloud\n"
    )

    f.write(
        "s5_probe_enabled=NO\n"
    )

    f.write(
        "s6_probe_enabled=NO\n"
    )

    f.write(
        "s5_fallback_enabled=NO\n"
    )

    f.write(
        "s6_fallback_enabled=NO\n"
    )

    f.write(
        "load_controller_path_preserved=YES\n"
    )

    f.write(
        "serialization_gate_preserved=YES\n"
    )

    f.write(
        "performance_t0_boundary_preserved=YES\n"
    )

    f.write(
        "candidate_v4_modified=NO\n"
    )

    f.write(
        "changed_node_ids="
        + ",".join(s5_changes)
        + "\n"
    )


print("================================================")
print("MATCHED STATIC FLOW GENERATION: PASS")
print("================================================")
print()
print("Source V4:")
print(V4)
print(sha256(V4))
print()
print("S5 matched:")
print(s5_file)
print(sha256(s5_file))
print()
print("S6 matched:")
print(s6_file)
print(sha256(s6_file))
print()
print("Changed node IDs:")
for x in s5_changes:
    print(" ", x)
print()
print("S5 forced placement : edge")
print("S6 forced placement : cloud")
print("S5/S6 probes        : DISABLED")
print("S5/S6 fallbacks     : DISABLED")
print("Load controller     : PRESERVED")
print("CPU sampler         : PRESERVED")
print("Serialization gate  : PRESERVED")
print("Timing boundary     : PRESERVED")
print("Candidate V4        : UNCHANGED")
print()
print("Manifest:")
print(manifest)
