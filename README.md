<!-- PAI-SG-LINKS:BEGIN -->
## Project Links

- [Project Portal](https://statera-workshop.github.io/#repo-lm_dec_base_gru)
- [Statera Workshop Portal](https://statera-workshop.github.io/)
<!-- PAI-SG-LINKS:END -->

# LM_Dec_Base_GRU

GRU - Base family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked (3 seed-repeat runs, see results)**
- Failure class: Note: tracks recently updated values at any distance, reproduced in 5 runs; no distant recall

## Provenance

- Source repository: (new) Legacy baseline, PyTorch nn.GRU
- Upstream / license: torch.nn.GRU (cuDNN); classic baseline requested 2026-10-22

## Differences from the source ([Unified] protocol)

- 4 layers, hidden 896 (to match LSTM size), input dim 512

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 18.62 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.864 -> 0.002 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.582 / test 1.587 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.521 / test 1.537 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 228 MB, inference 86 MB
- Peak GPU memory at length 8192 (batch 1): train 590 MB, inference 131 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.24 (keys removed 0.16); 4 pairs @ 2048: 0.23 (keys removed 0.14); 16 pairs @ 512: 0.06 (keys removed 0.07); 16 pairs @ 2048: 0.07 (keys removed 0.07)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 0.24 (keys removed 0.02); 4 pairs @ 2048: 0.25 (keys removed 0.02); 4 pairs @ 8192: 0.25 (keys removed 0.02); 16 pairs @ 512: 0.07 (keys removed 0.07); 16 pairs @ 2048: 0.07 (keys removed 0.07); 16 pairs @ 8192: 0.07 (keys removed 0.07)
- LM-06 memory update `lm06`: @512: latest 0.75, unchanged 0.04, stale 0.00; @2048: latest 0.71, unchanged 0.07, stale 0.01
- LM-06 memory update `lm06long`: @512: latest 0.76, unchanged 0.04, stale 0.00; @2048: latest 0.73, unchanged 0.05, stale 0.00; @8192: latest 0.73, unchanged 0.07, stale 0.01
- LM-06 memory update `lm06s2`: @512: latest 0.75, unchanged 0.06, stale 0.01; @2048: latest 0.72, unchanged 0.06, stale 0.01
- LM-06 memory update `lm06s3`: @512: latest 0.68, unchanged 0.05, stale 0.01; @2048: latest 0.69, unchanged 0.07, stale 0.02
- LM-06 memory update `lm06longs2`: @512: latest 0.74, unchanged 0.05, stale 0.00; @2048: latest 0.75, unchanged 0.05, stale 0.00; @8192: latest 0.71, unchanged 0.08, stale 0.01

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check gru
python run.py lm02 gru      # enwik8 is downloaded to data/ on first use
```
