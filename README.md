# ⚡ FlashAttention Forward Pass in Triton

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AvinashQtC/flash-attention-triton/blob/main/flash_attention_forward_triton.ipynb)

A from-scratch **causal FlashAttention forward kernel** written in [Triton](https://triton-lang.org), in two versions that compute the same thing but handle the causal mask differently:

| | Kernel | How the causal mask is handled |
|---|---|---|
| **V1** | `_fwd_v1_branch` | One loop over all KV tiles. A **runtime `if`** inside the loop masks tiles that touch the diagonal, and every tile pays for boundary masks. |
| **V2** | `_fwd_v2_staged` | The loop is split into **two compile-time stages** (`STAGE: tl.constexpr`). Off-diagonal tiles run with **no masks**, and only the diagonal tile is masked. |

Everything is in one notebook: [`flash_attention_forward_triton.ipynb`](flash_attention_forward_triton.ipynb).

## What's inside

1. **Algorithm:** tiling plus the online softmax, and how few tiles actually need a mask.
2. **NumPy reference:** the tiled algorithm on CPU, checked against the naive N×N formula (error about 1e-16).
3. **Triton kernels:** V1 (branch) and V2 (staged), autotuned, with stride-aware pointers so non-contiguous inputs work without a copy.
4. **Correctness tests:** 22 checks of the output `O` against `torch.nn.functional.scaled_dot_product_attention` and of the log-sum-exp `L` against `torch.logsumexp`. They cover fp32/fp16, sequence lengths that aren't multiples of the block size (77, 300, 1000), and a transposed layout. **All 22 checks pass.**
5. **Benchmark:** compared against PyTorch SDPA and a naive PyTorch implementation.

## Results (Tesla T4, `sm_75`, fp16, B=4, H=16, d=64, causal)

![benchmark](assets/benchmark.png)

| SEQ_LEN | SDPA (ms) | Naive (ms) | Triton V1 (ms) | Triton V2 (ms) | V2 vs V1 |
|---:|---:|---:|---:|---:|---:|
| 512  | 0.34  | 2.84   | 5.10   | 5.19   | 0.98× |
| 1024 | 0.69  | 11.24  | 7.84   | 8.06   | 0.97× |
| 2048 | 3.00  | 45.43  | 30.40  | 31.28  | 0.97× |
| 4096 | 12.20 | 187.43 | 123.33 | 128.45 | 0.96× |
| 8192 | 47.33 | —      | 493.46 | 512.81 | 0.96× |

**Takeaways**
- Both Triton kernels beat the naive implementation by about 1.5× from N=1024 up, and they run at N=8192, where the naive version was skipped because its N×N score matrix alone needs about 8.6 GB.
- On the T4 the staged V2 kernel is **slightly slower** (2–4%) than V1. The masking work it removes is small, and the branch in V1 doesn't hurt much on this hardware.
- Both kernels reach about 1.1 TFLOP/s, compared with about 11–12 TFLOP/s for PyTorch SDPA, so there is plenty of room left. Turing has no `cp.async`, so software pipelining (`num_stages > 1`) does little here.

## Run it

Open the notebook in Colab with the badge above, then choose **Runtime → Change runtime type → T4 GPU** and **Runtime → Run all**.

Locally you need an NVIDIA GPU and these packages:

```bash
pip install torch triton numpy pandas matplotlib
jupyter notebook flash_attention_forward_triton.ipynb
```

## Next steps

- [ ] Backward pass. `L` is already returned, which is what the FlashAttention backward needs.
- [ ] `exp2` with log₂e folded into the softmax scale.
- [ ] Non-causal mode and variable-length (packed) sequences.
- [ ] GQA / MQA.
- [ ] A `torch.autograd.Function` wrapper.

## References

1. T. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
2. T. Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
3. M. Milakov, N. Gimelshein, *Online normalizer calculation for softmax*, 2018. [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
4. Triton tutorial: [Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
