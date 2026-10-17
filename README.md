# Lightweight Adaptive Edge–Cloud Inference Placement — Reproducibility Package

Research artifacts accompanying the manuscript **“Lightweight Adaptive Edge–Cloud Inference Placement for IoT Systems under Dynamic Runtime Conditions: A Smart Parking Case Study”**, prepared for *Cluster Computing*.

This repository concerns the **extended S5–S7 controlled-Linux evaluation** of inference-placement policies. The earlier physical VisionFive 2 smart-parking prototype (S1–S4) is documented separately at [Zenodo DOI 10.5281/zenodo.21010731](https://doi.org/10.5281/zenodo.21010731). **S5–S7 results are not measurements of the physical VisionFive 2 board.**

## Study design

Each strategy processed the same frozen S4-derived workload, with 289 logical inference decisions per run, under the same seven-phase background CPU-load schedule. The campaign comprised 10 repetitions each for three strategies (30 runs; 8,670 logical inference decisions):

| Strategy | Placement policy | Successful logical requests |
| --- | --- | ---: |
| S5 | Edge-only | 2,890 / 2,890 |
| S6 | Cloud-only | 2,814 / 2,890 |
| S7 | Adaptive edge–cloud | 2,890 / 2,890 |

The adaptive policy combines smoothed latency and CPU measurements, an edge-load penalty, hysteresis, consecutive-superiority testing, minimum dwell time, alternative-site probing, and failure-aware fallback. The evaluation used a frozen Random Forest model with an identical artifact at the local and remote inference sites.

## Repository layout

| Location | Contents |
| --- | --- |
| `data/frozen_workload/` | S4-derived source trace, fixed inference reference, and seven-phase load schedule |
| `data/raw/S5_edge_only/rep01`–`rep10/` | S5 run-level inference, load, and validation records |
| `data/raw/S6_cloud_only/rep01`–`rep10/` | S6 run-level records, including unsuccessful requests |
| `data/raw/S7_adaptive/rep01`–`rep10/` | S7 decisions, fallback and probing records |
| `analysis/final_analysis_v1/` | Final campaign aggregation, statistical outputs, and analysis code |
| `results/derived/` | Derived tabular results used for manuscript analyses |
| `nodered/` | Node-RED execution-flow exports; inspect environment-specific settings before reuse |
| `services/` | Local and remote inference-service sources and dependencies |
| `model/` | Frozen model artifact and feature-order metadata |
| `scripts/` | Replay, controlled load, flow construction, and publication analysis scripts |
| `protocol/` | Campaign order and validation materials |
| `docs/` | Additional methodology and reuse notes |

Publication figure image files are intentionally omitted; the manuscript contains the figures. Their underlying results data and supporting scripts are retained where available.

## Key parameters

- EWMA smoothing coefficients: `alpha_L = alpha_U = 0.20`.
- Edge CPU utilization thresholds: 55% and 80%; latency penalty scale: 295.659 ms.
- Switching hysteresis: 45 ms; consecutive-superiority requirement: 3 decisions.
- Minimum dwell interval: 10 decisions; alternative-site probing interval: 10 decisions.
- Statistical uncertainty: paired two-stage circular moving-block bootstrap, block length 10 decisions, 5,000 resamples, random seed 519. Interpret intervals conditional on the tested seven-phase workload (not strictly stationary).

## Reported pooled latency results

Application-level latency includes measured durations for unsuccessful inference decisions where applicable; service-only decompositions use successful requests.

| Strategy | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| S5 | 128.22 | 85.27 | 291.46 | 458.38 |
| S6 | 474.26 | 401.36 | 902.54 | 1532.20 |
| S7 | 188.60 | 92.00 | 487.78 | 643.88 |

S7 reduced median and P95 latency relative to S6 by 77.08% and 45.96%, respectively, but S5 remained faster than S7 on these pooled latency metrics. During S7, the controller recorded 19 normal edge-to-cloud transitions, 15 normal cloud-to-edge transitions, and four successfully recovered cloud-primary failures. These observations do not establish general fault tolerance or superior latency relative to edge-only operation.

## Model integrity

The evaluated frozen model's SHA-256 is:

```text
f400fb75d4ef47f0359129e4ac96cad49ae4c1694f33326014d54b977fe2855e
```

Check the checksum of `model/parking_status_model.joblib` after download before running experiments. Prediction equivalence is evaluated only for successfully returned inference requests; it is not a measure of accuracy against independent ground-truth labels.

## Reusing the package

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). The original experiments involved environment-specific Node-RED, MQTT/TLS and remote-service settings. Do not assume that the flow exports will execute unchanged on another host. Credentials, local user paths, and remote service addresses must be supplied by each reproducing researcher; **no live secrets should be committed**.

## Relationship to the earlier repository

- Historical S1–S4 source and physical-prototype evidence: [original GitHub repository](https://github.com/mohammedsaadalyasari-afk/secure-edge-cloud-smart-parking-jnca-reproducibility) and [Zenodo archive](https://doi.org/10.5281/zenodo.21010731).
- This repository: S5–S7 controlled-Linux adaptive-placement experiment and final analytical campaign.

## Citation and reuse

Please cite this reproducibility package using [`CITATION.cff`](CITATION.cff). A permanent archive DOI will be added here after the first public release is deposited in Zenodo. **License terms and contributor metadata must be confirmed before making this repository public.**
