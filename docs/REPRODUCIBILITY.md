# Reproducibility and scope notes

## Provenance

The S5–S7 experiment used a deterministic replay derived from an S4 secure-hardware telemetry segment (1,396 source messages spanning approximately 288.4 seconds) and generated 289 logical inference inputs per run. Original S1–S4 datasets and physical-hardware code are in the separate historical archive: https://doi.org/10.5281/zenodo.21010731.

The final S5–S7 campaign used a controlled Linux execution environment with Node-RED 4.1.1, Node.js 18.19.1, Mosquitto and Python/FastAPI inference endpoints. Reported inference dependencies include NumPy 1.23.5, pandas 1.5.3, scikit-learn 1.1.2, and joblib 1.2.0. The cloud service was a remote inference endpoint; neither the historical cloud provider nor the original physical VisionFive 2 device should be inferred from these measurements.

## Data organization and counting

Each of `data/raw/S5_edge_only/`, `data/raw/S6_cloud_only/`, and `data/raw/S7_adaptive/` should contain directories `rep01` through `rep10`. Each repetition's primary file (`S5_experiment_log.csv`, `S6_experiment_log.csv`, or `S7_experiment_log.csv`) contains 289 logical-decision records. Unsuccessful S6 requests must be retained in application-level latency calculations, not silently removed. Probes are recorded separately and excluded from the main logical-request latency dataset.

The files in `analysis/final_analysis_v1/` and `results/derived/` contain the final analytical outputs. See the actual script headers and CLI definitions before executing scripts; this package has not been independently run on a clean third-party host.

## Controller parameters

The S7 controller uses two same-model inference locations. Parameters reported in the manuscript: EWMA 0.20 for latency and CPU; CPU lower/upper 55%/80%; penalty scale 295.659 ms; hysteresis margin 45 ms; consecutive superiority 3 decisions; minimum dwell 10 decisions; periodic probe interval 10 decisions. Decisions are serialized until request and probe results are committed.

## Evaluation and statistical methods

For each strategy, the same seven-phase worker schedule applies to the same 289 logical inputs: L0 (1–40, 0 workers); L1 (41–80, 4); L2 (81–120, 6); L3 (121–170, 8); L4 (171–210, 4); L5 (211–250, 8); L6 (251–289, 0). Worker counts are controlled inputs, not measured CPU percentages. The environment and remote path can vary across the separately executed runs.

The ten S6/S7 run pairs are analyzed using paired two-stage circular moving-block bootstrap with block length 10, 5,000 resamples, seed 519; matched decision indices are used within each selected pair. The imposed seven-phase workload is nonstationary, so confidence intervals are empirical uncertainty summaries conditional on this campaign, not general coverage guarantees.

## Safe reuse and public-release checks

Before a public release, review flow exports, scripts, YAML, metadata, logs and checksum manifests for API tokens, passwords, MQTT credentials, private keys, host IPs, email addresses, absolute local paths and unintended personal data. Use harmless placeholders for connection values only in clearly labeled **publication copies**; retain original experimental logs and model artifacts unchanged. In particular, treat historical `RUN_SHA256SUMS.txt` and other integrity manifests as evidence of the original run, not portable checksum files if they contain absolute local paths.

Do not publish virtual environments, `node_modules`, device images, application secrets, service-account credentials, TLS private keys, or unrelated development snapshots. Review file sizes before uploading to GitHub.

## Software use

- `nodered/`: inspect exported flows and update configuration nodes for the reproducing environment; confirm which flow export corresponds to the final run.
- `model/`: verify the model hash documented in the README.
- `services/`: review runtime requirements and provide external endpoint configuration.
- `scripts/`: inspect arguments and path dependencies before execution; code was developed in a specific runtime environment.
- `analysis/final_analysis_v1/`: preserve the final analysis code and supporting tabular outputs; compare generated statistics with published tables before claiming independent reproduction.

**Publication readiness is conditional on a complete file-content review.** The existence of the experimental files and model checksum match do not alone constitute a clean-install software reproducibility test.
