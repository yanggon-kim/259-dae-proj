# WS_BASE vs DUAL_ISSUE_PC Reference Results

These are the reference GPGPU-Sim results packaged with the consolidated
`01_gpgpusim_design/` directory. The full run uses:

```text
--n 524288 --iters 128 --memory-iters 4 --warmup 0 --repeats 1
```

Speedup is computed as:

```text
speedup = cycles(WS_BASE) / cycles(DUAL_ISSUE_PC)
```

## Full-Scale Result

| Implementation | Cycles | Instructions | Pair successes | Correct | Speedup vs `WS_BASE` |
|---|---:|---:|---:|---|---:|
| `WS_BASE` | 54,110 | 281,149,440 | 0 | PASS | 1.000x |
| `DUAL_ISSUE_PC` | 59,165 | 281,149,440 | 188,694 | PASS | 0.915x |

The dual-issue design issues producer-consumer pairs, but this kernel does not
expose enough simultaneous legal producer and consumer instructions to beat the
warp-specialized baseline at the full 512-CTA scale.

## Pair Diagnostics For `DUAL_ISSUE_PC`

| Metric | Count |
|---|---:|
| Pair attempts | 25,507,328 |
| Pair successes | 188,694 |
| Scoreboard failures | 17,707,793 |
| No-producer windows | 6,611,605 |
| Register-set busy failures | 89,731 |
| Both-ready cycles | 238,404 |

The main limiter is readiness overlap: most attempted pairs fail because one
side is not scoreboard-ready or because the scheduler does not see a useful
producer candidate in that cycle.

## Scout Result

The smaller scout case uses `--n 16384` with the same compute and memory
parameters.

| Implementation | Cycles | Pair successes | Correct | Speedup vs `WS_BASE` |
|---|---:|---:|---|---:|
| `WS_BASE` | 28,201 | 0 | PASS | 1.000x |
| `DUAL_ISSUE_PC` | 28,101 | 2,620 | PASS | 1.004x |

This shows the mechanism can produce a small win when the workload is more
underfilled, but the win does not carry to the full-scale run.
