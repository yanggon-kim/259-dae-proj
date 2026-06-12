# SMEMQ Benefit Search

Goal: find SMEMQ-based workloads where higher CTA/SM at the same queue depth
beats the RFQ-based implementation.

Kernel family: generated RFQ-based and SMEMQ-based stream-FMA
producer/consumer kernels with identical host geometry. The search varies
memory iterations, compute iterations, queue depth, and optional dead
consumer-side FMA padding; simulator design stays fixed.

Rows shown here are completed RFQ/SMEMQ pairs only; incomplete long candidates
are omitted from the aggregate.

| Rank | CTA Count | Memory Iters | Compute Iters | Padding FMAs | Depth | RFQ Cycles | SMEMQ Cycles | SMEMQ Speedup | RFQ CTA/SM | SMEMQ CTA/SM | RFQ Elig. % | SMEMQ Elig. % | RFQ L1D Fails | SMEMQ L1D Fails |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 4 | 128 | 0 | 32 | 80183 | 63937 | 1.2541 | 3 | 4 | 11.51 | 18.33 | 4645704 | 5743289 |
| 2 | 448 | 4 | 128 | 0 | 32 | 76762 | 61326 | 1.2517 | 3 | 4 | 12.27 | 16.12 | 4870683 | 5390373 |
| 3 | 528 | 4 | 128 | 0 | 32 | 96113 | 77575 | 1.2390 | 3 | 4 | 15.04 | 25.74 | 8990964 | 11273837 |
| 4 | 397 | 4 | 128 | 0 | 32 | 71194 | 57550 | 1.2371 | 3 | 4 | 13.33 | 12.78 | 4885831 | 4560219 |
| 5 | 529 | 4 | 128 | 0 | 32 | 85534 | 75286 | 1.1361 | 3 | 4 | 12.83 | 19.23 | 5616456 | 6089911 |
| 6 | 384 | 4 | 128 | 0 | 32 | 63942 | 63942 | 1.0000 | 3 | 4 | 16.23 | 16.23 | 8258705 | 8258705 |
| 7 | 396 | 4 | 128 | 0 | 32 | 67454 | 67454 | 1.0000 | 3 | 4 | 17.54 | 17.54 | 9895307 | 9895307 |
| 8 | 640 | 4 | 128 | 0 | 32 | 87171 | 88721 | 0.9825 | 3 | 4 | 11.30 | 19.11 | 5061981 | 7041155 |

## Interpretation

Best passing candidate: `memory_iters=4`, `compute_iters=128`,
`consumer_padding_fmas=0`, depth `32` with SMEMQ-based speedup `1.2541`.

A candidate is useful only when SMEMQ has higher CTA/SM than RFQ and the extra
resident CTAs improve cycles instead of only increasing L1/MSHR pressure.

For the best row, the extra SMEMQ CTA/SM raises eligible occupancy enough to
beat the additional shared-memory-queue traffic, while the lighter compute
cases are still dominated by memory-system pressure.
