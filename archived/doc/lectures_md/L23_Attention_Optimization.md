# L23 — Advanced Optimizations: Improving Attention

## Objectives

- Understand why attention is a major bottleneck in LLM inference.
- Learn the naive three-kernel attention implementation and its memory problems.
- Understand kernel fusion challenges (shared memory limits, Softmax dependency).
- Learn **Online Softmax** and how it enables tiled, fused attention.
- Understand **Flash Attention** (versions 1–4) and its significance.
- Learn attention-level optimizations: **Local/Windowed Attention**.
- Understand next-token generation and the **KV Cache** optimization.
- Learn **PagedAttention** for efficient KV cache memory management.
- Understand **Quantization** for reducing KV cache size.
- Learn the **Perplexity (PPL)** scoring system for evaluating quantized models.
- Distinguish the two phases of LLM inference: **Prefill** vs **Decode**.

---

## Review: What is Attention?

- Main operation in modern LLMs that allows tokens to attend to each other, providing context across the text.
- Attention is one of the major bottlenecks in LLM serving — significantly higher latency than all other operations (except matmul) even in optimized frameworks.

---

## Basic Attention Design

- Attention itself is just **two matrix multiplications and a Softmax**.
- To obtain Q, K, V vectors, each input token is multiplied by pretrained weight matrices (plus biases).
- Steps: Q/K/V projection (matmul) → QK^T matmul → Softmax → PV matmul.

---

## Naive Implementation — Three Separate Kernels

1. **Kernel 1**: Load Q (T×C) and K^T (C×T) from global memory, matmul → store T×T result to global memory.
2. **Kernel 2**: Load T×T result, apply Softmax, store T×T result back.
3. **Kernel 3**: Load T×T Softmax result and V, matmul → final output.

**Problem**: T is the sequence/context length — in modern models this can be tens of thousands to millions. Moving a 100K×100K matrix back and forth between global and on-chip memory is terrible for performance.

---

## Kernel Fusion — Why It's Hard

- Fusing all three kernels into one would avoid moving the T×T matrix through global memory.
- **But**: shared memory is tiny (~100KB per SM on A40, ~140KB on B200) — far too small for a 100K×100K matrix.
- **Tiling attempt**: compute small tiles of the T×T matrix, do Softmax and PV matmul on partial results?
- **Blocker**: Softmax requires the **max and sum of each full row** of the T×T matrix. You can't do Softmax (or the PV matmul) until you have the full QK result.

---

## Online Softmax — The Breakthrough

Standard Softmax formula requires two passes: first find the global row max, then compute exponentials and normalize.

**Online Softmax** keeps a **running max** `m` and **normalization term** `d` as it iterates over the input, continuously rescaling earlier values when a new max appears. This eliminates the need to compute the global maximum upfront, allowing Softmax to be computed **iteratively**.

- Originally developed for next-token generation in language models with extremely large vocabularies (pre-Transformer era).
- It took ~4 years for people to apply this algorithm to Transformers.

---

## Flash Attention

In May 2022, a Stanford research group connected Online Softmax with tiled attention.

### Key Idea

Using Online Softmax, Flash Attention **fuses all three attention steps** (QK matmul, Softmax, PV matmul) into **one kernel**.

- In a **tiling fashion**, loads tiles of Q, K, V into fast on-chip memory (shared memory).
- Accumulates partial output values for a tile of Queries using each tile of Keys and Values.
- Despite having the **same FLOPs** as basic attention, it achieves **3×–10× speedup** by eliminating memory transfer overhead.

### Versions

| Version | Focus |
|---|---|
| FlashAttention 1 | Introduced tiled attention using Online Softmax |
| FlashAttention 2 | Algorithmic tweaks + different work partitioning among warps |
| FlashAttention 3 | Hopper+ hardware (WGMMA, TMA, FP8, etc.) |
| FlashAttention 4 | Blackwell hardware (CTA scheduling) |

---

## Full Sequence Attention Is Still Costly

Even with FlashAttention, attention scales **O(T²)** in time complexity.

Many tasks don't require long-range context (translation, normal conversation). In resource-constrained cases (e.g., smart watch), full-context attention is impractical.

### Local/Windowed Attention

- Reduces attention context to a **fixed local window** of W tokens.
- Complexity drops from O(T²) to **O(T·W)**.
- If sequence is longer than W, attend only to the most recent W tokens.
- **Pro**: easy to implement, quick performance gain.
- **Con**: loses long-range context — chatbot forgets your name, code generation becomes redundant.

### Advanced Strategies

- Periodically attend to full context to recover accuracy.
- Dynamic window size for different prompt types.
- Strided windows with increasing jumps to reach earliest tokens.
- Tricky to tune; hard to recover accuracy for all use cases.

---

## Next Token Generation & KV Cache

### Autoregressive Generation

Each new token computes attention over **all previous tokens**. The Query for the new token is multiplied with all previous Keys and Values.

**Problem**: Key and Value vectors of previous tokens **don't change** when generating new tokens, yet the naive approach recalculates them every time.

- At the 50th token: 50 K/V computations needed.
- At the 500th token: 500 computations.
- Total operations grow **quadratically** — ~500,000 redundant calculations for 1,000 tokens.

### KV Cache

- **Saves and reuses** previously computed K and V vectors.
- Per-token generation cost drops from **O(n²) to O(n)**.
- Attention shifts from compute-bound to **memory-bound** (reads replace computation).

### Implementation Considerations

- Despite the name, KV cache is stored in **global memory** (shared memory/registers are too small).
- Requires persistent state between forward pass calls across batches.
- Many kernels need redesign.
- Position embedding logic must stay consistent between prefill and decode.
- Large KV cache memory needs proper management.

---

## KV Cache Memory Management

### The Scale Problem

- A100 GPU: ~80GB global memory.
- 70B LLaMA2 model: ~32KB of KV cache per token per layer.
- Single user prompt: ~10GB from KV cache alone.

### Memory Fragmentation

Naive KV cache management causes both internal and external fragmentation:

- **Internal fragmentation**: model stops generating early → rest of allocation wasted.
- **External fragmentation**: allocation sizes vary → gaps between chunks.
- **Reservation waste**: constant-size allocations locked for entire request duration.

### PagedAttention (2023)

Inspired by OS Virtual Memory Paging:

- Uses **fixed-size pages (blocks)** and **page tables** to decouple logical KV cache positions from physical memory locations.
- Each request's KV cache is broken into smaller pages (e.g., 16 tokens per block).
- New physical blocks allocated as new tokens are generated.
- When GPU allocator runs out of space, entire KV cache can be swapped to CPU until other requests complete.
- Model views cache as contiguous logical space while physical storage is scattered.

---

## Quantization

- Reduce KV cache size by lowering the precision of stored K/V vectors.
- FP32 → FP16: 2× cache size. FP32 → FP8: 4× cache size.
- **Tradeoff**: significant precision loss. Need sophisticated algorithms for overflow and scaling.
- Not "free" — must evaluate actual impact on model quality.

### Model Evaluation — Perplexity (PPL)

- Most common metric for evaluating autoregressive language models.
- Measures how well a model predicts the next token.
- Defined as the exponentiated average negative log-likelihood.
- **Low PPL** = model predicts well. **High PPL** = model is consistently "surprised."
- Dataset choice matters — PPL measures fit to a specific distribution.
- Notable evaluation datasets: Wikitext-2, C4, gsm8k.

---

## Two Phases of LLM Inference

| Phase | Name | Description | Profile |
|---|---|---|---|
| 1 | **Prefill** | Processes entire user prompt in parallel to compute KV cache | **Compute-bound** — highly parallel matmuls |
| 2 | **Decode** | Generates response autoregressively, one token at a time | **Memory-bandwidth bound** — constantly reading KV cache |

---

## Key Takeaways

- Naive attention uses three kernels with massive T×T intermediate matrices in global memory.
- **Online Softmax** removes the Softmax barrier to kernel fusion by maintaining running statistics.
- **Flash Attention** fuses QK-matmul, Softmax, and PV-matmul into one tiled kernel — same FLOPs, 3–10× faster.
- **Local/Windowed Attention** reduces O(T²) to O(T·W) by limiting context window.
- **KV Cache** eliminates redundant K/V recomputation in autoregressive decoding — shifts attention from compute-bound to memory-bound.
- **PagedAttention** applies OS paging concepts to KV cache, solving fragmentation.
- **Quantization** trades precision for memory capacity — evaluated via **Perplexity (PPL)**.
- LLM inference has two distinct phases: **Prefill** (compute-bound) and **Decode** (memory-bound).
