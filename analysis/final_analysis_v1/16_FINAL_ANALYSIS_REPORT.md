# Final matched campaign analysis

Inferential unit: matched experimental block/run (n=10 per strategy).
Pooled decision-level results are descriptive only.
The primary latency estimand retains observed failure/timeout elapsed latency.

## Controller-order integrity

All 30 final runs passed exact ordered controller decision_number and replay_seq verification against the frozen 289-row schedule.

## Strategy results
- S5: mean=128.219±12.521 ms; median=86.349±8.536 ms; P95=291.227±41.472 ms; P99=457.518±145.613 ms; success=100.000±0.000%; final failures=0.
- S6: mean=474.255±78.576 ms; median=407.308±30.524 ms; P95=859.228±406.304 ms; P99=1376.243±504.722 ms; success=97.370±4.332%; final failures=76.
- S7: mean=188.604±23.508 ms; median=92.921±10.921 ms; P95=470.617±69.083 ms; P99=636.452±144.071 ms; success=100.000±0.000%; final failures=0.

## Primary paired comparison: S7 versus S6
- total_mean_ms: paired reduction=59.385% (95% bootstrap CI 54.668% to 63.916%); S7 lower in 10/10 blocks.
- total_median_ms: paired reduction=77.079% (95% bootstrap CI 75.175% to 78.799%); S7 lower in 10/10 blocks.
- total_p95_ms: paired reduction=35.413% (95% bootstrap CI 20.030% to 50.625%); S7 lower in 9/10 blocks.
- total_p99_ms: paired reduction=48.171% (95% bootstrap CI 35.984% to 59.342%); S7 lower in 10/10 blocks.
- Success-rate difference S7-S6=2.630 percentage points (95% bootstrap CI 0.692 to 5.536).

## Secondary paired comparison: S7 versus S5
- total_mean_ms: paired reduction=-48.055% (95% bootstrap CI -59.267% to -35.195%); S7 lower in 0/10 blocks.
- total_median_ms: paired reduction=-7.722% (95% bootstrap CI -12.474% to -3.278%); S7 lower in 1/10 blocks.
- total_p95_ms: paired reduction=-64.812% (95% bootstrap CI -82.608% to -44.455%); S7 lower in 1/10 blocks.
- total_p99_ms: paired reduction=-49.762% (95% bootstrap CI -79.138% to -20.895%); S7 lower in 2/10 blocks.

## S7 adaptive behavior
- Edge selected 2176/2890 (75.294%); cloud selected 714/2890 (24.706%).
- Switches=38; cost E->C=19; cost C->E=15; failure-related=4.
- Primary failures=4; fallbacks=4; fallback successes=4; final failures=0.
- Fallback recovery among primary failures=100.000%.
- S7 probes=266/280 successful; all excluded from primary performance latency.
