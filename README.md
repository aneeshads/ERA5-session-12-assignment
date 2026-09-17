# ZeRO-1, ZeRO-2 and ZeRO-3 on 32 Virtual GPUs

**ERA V5 — Session 12 assignment.** This project constructs thirty-two virtual
GPUs from CPU threads, trains a small model on them under four data-parallel
strategies — standard data parallelism (DDP) and ZeRO stages 1, 2 and 3 — and
measures, at every step, the memory held on each GPU, the bytes moved over the
interconnect, and the computation performed per GPU.

* [`assignment.ipynb`](assignment.ipynb) — the executed notebook; runs unchanged on Google Colab with no setup
* [`zero_sim.py`](zero_sim.py) — the simulator: virtual GPUs, the interconnect, the hand-written model, and the four strategies
* [`figures/`](figures/) — the plots reproduced below

```bash
python3 zero_sim.py        # trains all four stages (~2 min on a laptop) and prints the comparison
```

---

## Summary

Under data parallelism on N GPUs, every GPU holds a complete copy of the
model, its gradients, and its optimizer state. With mixed-precision Adam this
amounts to **16 bytes per parameter on every GPU**; only the activations
differ from one GPU to the next. ZeRO (*Zero Redundancy Optimizer*,
Rajbhandari et al., 2020) removes this replication in three stages: the
optimizer state is partitioned (**ZeRO-1**), then the gradients (**ZeRO-2**),
then the parameters themselves (**ZeRO-3**). Each GPU owns 1/N of every
tensor and obtains the remainder on demand. The first two stages add no
communication; the third increases it by 1.5×. None of the stages alters the
arithmetic — the simulation verifies that all four strategies produce
bit-identical weights.

---

## 1. Design of the Simulator

The simulator is constructed so that the memory figures are real byte counts
and the communication figures are real collective calls, while the hardware
itself is emulated.

| component | description | real or emulated |
|---|---|---|
| `VirtualGPU` | An object with a rank and a private ledger. `hold(name, category, tensor)` places a tensor on the GPU and charges the ledger; `drop(name)` releases it. The ledger tracks current bytes per category and the peak, with a breakdown at the moment of the peak. | Memory is **real**: the bytes of the tensors the object currently holds. |
| worker threads | Each virtual GPU's forward, backward and optimizer work runs on its own thread, 32 concurrently. | Compute is real; wall-clock time is not meaningful (a single Python interpreter lock). |
| `Fabric` | Performs `all_reduce`, `reduce_scatter` and `all_gather` across the 32 GPUs and charges each GPU the bytes a ring implementation would move. | The arithmetic is performed once in-process; the **byte accounting is real**. |
| model | A four-layer MLP, 256→1024→1024→1024→10, Ψ = 2,372,618 parameters, with **forward and backward passes written by hand**. | Real; no autograd, so the exact moment each weight is required is explicit. |
| mixed precision | bf16 parameters and gradients; fp32 master weights and Adam moments *m*, *v*. | Real data types and sizes. |
| bf16 matmul | The host CPU has no fast bf16 kernel, so `mm_bf16` multiplies in fp32 and rounds the result to bf16. | Equivalent to a tensor core: bf16 in, fp32 accumulation, bf16 out. |

The backward pass is written by hand because ZeRO-3 is fundamentally a
question of *when a weight exists*. Under autograd, the weight is silently
kept alive between forward and backward. An explicit `layer_backward` forces
the question "where does `W` come from at this point?" — and for ZeRO-3 the
answer is that it does not exist and must be gathered again.

### A brief description of matmul

Every `mm_bf16(...)` call in the simulator is a **matrix multiplication**.
It accounts for almost all of a network's compute, so the operation is worth
stating precisely.

A single neuron holds a list of weights and receives a list of inputs; its
output is each input multiplied by its weight, summed — a **dot product**:

```
inputs   x = [2, 1, 3]
weights  w = [1, 0, 2]
output     = 2·1 + 1·0 + 3·2 = 8
```

A layer contains many neurons and receives a *batch* of inputs, so every
input must be dotted with every neuron. With the inputs as rows of one matrix
and the neurons as columns of another, matrix multiplication performs all of
those dot products at once:

```
        X (batch × in)          W (in × out)            Y = X @ W (batch × out)
   ┌ 2  1  3 ┐  input #1     ┌ 1  5 ┐  neuron A, B     ┌ 8  11 ┐  input #1 → A, B
   └ 0  4  1 ┘  input #2  @  │ 0  1 │             =    └ 2   4 ┘  input #2 → A, B
                             └ 2  0 ┘
```

Cell *(i, j)* of the result is *row i of X · column j of W*. The top-left
entry is the single neuron above: `2·1 + 1·0 + 3·2 = 8`. The top-right entry
is `2·5 + 1·1 + 3·0 = 11`.

**Shape rule:** `(batch × in) @ (in × out) → (batch × out)`. The inner
dimension must match — it is the length of the vectors being dotted — and it
is eliminated.

In `layer_forward`, `x` is 64 samples × 256 features and `W` is stored as
`(out, in)` = 1024 × 256, so `W.T` is 256 × 1024 and `x @ W.T` is 64 × 1024:
all 64 samples pass through all 1024 neurons in a single call. The backward
pass consists of two further matmuls over the same tensors, rearranged:
`dW = dout.T @ x` (each weight's contribution to the error) and
`dx = dout @ W` (each input's contribution, passed to the layer below). One
matmul forward and two backward, each a multiply and an add per weight per
sample, gives the `6·Ψ·batch` FLOP count used in §3.

A `(a×b) @ (b×c)` matmul costs `a·b·c` multiply-adds — 67 million for one
1024×1024 layer at batch 64 — and GPUs contain dedicated hardware ("tensor
cores") for exactly this operation. This is also the origin of the bf16 note
in the table above: a tensor core accepts bf16, accumulates in fp32, and
returns bf16, which is precisely what `mm_bf16` reproduces.

### The Origin of 16 Bytes per Parameter

| tensor | dtype | bytes | consumer |
|---|---|---|---|
| working weight | bf16 | 2 | forward and backward |
| gradient | bf16 | 2 | optimizer |
| master weight | fp32 | 4 | optimizer only |
| Adam *m* | fp32 | 4 | optimizer only |
| Adam *v* | fp32 | 4 | optimizer only |
| | | **16** | |

The final three rows — 12 of the 16 bytes, the paper's *K = 12* — are read
by the optimizer alone. That observation is the entire opening for ZeRO-1.

### The Three Collectives

| collective | semantics | bytes per GPU (ring algorithm) |
|---|---|---|
| all-reduce | every GPU contributes a full tensor; every GPU receives the mean | 2·(N−1)/N · size |
| reduce-scatter | every GPU contributes a full tensor; GPU *i* receives only chunk *i* of the mean | (N−1)/N · size |
| all-gather | GPU *i* contributes chunk *i*; every GPU receives the full tensor | (N−1)/N · size |

The identity on which the analysis rests: **an all-reduce is a reduce-scatter
followed by an all-gather**, at the same byte cost. This is why the first two
ZeRO stages are free in communication.

## 2. The Four Strategies as Implemented

One training step proceeds in lockstep across the GPUs. Each GPU processes
its own micro-batch of 64 samples.

**DDP (stage 0).** Every GPU holds all 16Ψ bytes. After forward and backward,
every gradient is all-reduced so that all GPUs hold the mean; every GPU then
runs the full Adam update on its full fp32 copy and refreshes its bf16 working
weights. Thirty-two GPUs perform thirty-two identical optimizer steps.

**ZeRO-1 — partition the optimizer state.** Each GPU retains only its 1/32
slice of the master weights, *m* and *v*. Forward and backward are unchanged
(full bf16 weights; full bf16 gradients materialize). In place of the
all-reduce, the gradients are **reduce-scattered** so that GPU *i* receives
only the mean of slice *i*; each GPU updates its slice; the updated bf16
slices are **all-gathered** so that every GPU holds the new full weights.
Reduce-scatter plus all-gather equals the byte cost of one all-reduce: nothing
additional crosses the interconnect, and no GPU ever requires the full
averaged gradient.

**ZeRO-2 — partition the gradients as well.** The same two collectives as
ZeRO-1, but the reduce-scatter is issued *inside* backward: the moment layer
*i*'s gradient is computed, it is scattered and only the owning 1/32 slice is
retained. The full gradient never resides in memory. The difference between
ZeRO-1 and ZeRO-2 is solely *when* one call is made.

**ZeRO-3 — partition the parameters.** Each GPU holds only its 1/32 slice of
the bf16 weights. Before layer *i*'s forward pass, its weights are
**all-gathered** into a temporary buffer, used, and freed. Backward requires
them again, so they are **gathered a second time**, the gradient is computed,
the buffer is freed, and the gradient is reduce-scattered as in ZeRO-2. After
the optimizer updates its fp32 slice, each GPU refreshes only its bf16 slice —
there is no final all-gather; the next forward pass gathers on demand. Three
collectives per layer rather than two per step: an additional Ψ on the
interconnect.

## 3. Measurements

### Memory per GPU

![peak memory](figures/peak_memory.png)

| strategy | peak per GPU | model states (measured) | paper formula | = | reduction vs DDP |
|---|---|---|---|---|---|
| DDP | 36.20 MiB | 36.20 MiB | 16Ψ | 36.20 | 1× |
| ZeRO-1 | 9.90 MiB | 9.90 MiB | 4Ψ + 12Ψ/N | 9.90 | 3.7× |
| ZeRO-2 | 5.97 MiB | 5.52 MiB | 2Ψ + 14Ψ/N | 5.52 | 6.1× |
| ZeRO-3 | 3.46 MiB | 1.13 MiB | 16Ψ/N | 1.13 | 10.5× |

The measured model-state bytes match Figure 1 of the paper to two decimal
places, which indicates that the simulation implements the mechanism the paper
describes rather than an approximation of it.

The ledger exposes two effects the paper's formula does not:

* **ZeRO-2's peak (5.97 MiB) exceeds its model-state figure (5.52 MiB)**
  because the peak now occurs at the *end of the forward pass*, where
  activations are largest. Once gradients cease to dominate, activations
  become the next visible term.
* **ZeRO-3's peak (3.46 MiB) is three times its model-state figure
  (1.13 MiB).** The additional 2.0 MiB is a single 1024×1024 layer gathered in
  full. ZeRO-3 partitions the *storage* of a layer, but the layer must still
  exist in its entirety on every GPU while in use. The floor of ZeRO-3 is the
  largest layer, not Ψ/N.

### Memory Within a Step

![timeline](figures/timeline.png)

The plot shows GPU 0's ledger after each phase of a single step and makes the
mechanism of each stage visible:

* **DDP and ZeRO-1** rise through backward as full gradients accumulate
  (+4.5 MiB), then fall when the optimizer releases them. ZeRO-1 has the same
  profile 26 MiB lower, having discarded 31/32 of the fp32 states before the
  step begins.
* **ZeRO-2 falls through backward.** Each layer's gradient is scattered as
  soon as it exists, so memory decreases layer by layer rather than
  increasing.
* **ZeRO-3 is a sawtooth.** Each tooth is a layer being gathered and freed.
  Teeth appear in backward as well — the second gather, which is the source of
  the additional communication.

### Communication per GPU

![communication](figures/comm.png)

| strategy | measured per step | paper | calls per step |
|---|---|---|---|
| DDP | 8.77 MiB | 2Ψ elements | 8 all-reduce |
| ZeRO-1 | 8.77 MiB | 2Ψ | 8 reduce-scatter + 8 all-gather |
| ZeRO-2 | 8.77 MiB | 2Ψ | 8 reduce-scatter + 8 all-gather |
| ZeRO-3 | 13.15 MiB | 3Ψ | 16 all-gather + 8 reduce-scatter |

(The count of 8 corresponds to the model's eight parameter tensors; a
production implementation would bucket them.) ZeRO-1 and ZeRO-2 cost exactly
what DDP costs. ZeRO-3 costs 1.5×, entirely from gathering every layer a
second time during backward. The gathered layer cannot be retained between
forward and backward, as that would reinstate 2Ψ of memory and defeat the
purpose of the stage.

### Computation per GPU

Three distinct quantities are reported, since they behave differently.

| | DDP | ZeRO-1/2/3 |
|---|---|---|
| forward + backward FLOPs per GPU | 0.911 GFLOP | **identical** |
| parameters updated by the optimizer per GPU | 2,372,640 | **74,145** (÷32) |
| Adam step, one GPU in isolation, single thread | 9.1 ms | **0.14 ms** (63×) |
| synchronisation points per step | 8 | 16 / 16 / 24 |

* **The matrix multiplications are unchanged.** Each GPU still processes its
  own micro-batch through the full model. ZeRO changes which GPU *stores* a
  weight, not which GPU *uses* it. The FLOP count is identical and the loss
  curves coincide.
* **The optimizer becomes at least N× cheaper per GPU** under any ZeRO stage,
  since each GPU updates only what it owns — 32× fewer elements, and 63× less
  measured time, because a 300 KB shard fits in cache while the 9.5 MB full
  tensors do not. Under DDP, thirty-two GPUs performing the same full Adam
  update constitutes redundant computation as well as redundant memory.
* **ZeRO-3 adds synchronisation points, not FLOPs.** Each additional
  all-gather is a point at which the GPU waits on the interconnect.
  Production systems conceal this by prefetching layer *i+1* while computing
  layer *i*, which is why ZeRO-3 requires a fast interconnect to match ZeRO-2's
  throughput.

The simulation's wall-clock time is not reported as a compute measurement.
Thirty-two Python threads share a single interpreter lock, so per-thread
timings reflect the interpreter rather than the algorithm; the notebook prints
them with that caveat attached.

### Bit-Identical Weights

```
ZeRO-1 vs DDP after 60 steps:  max |Δw| = 0.0
ZeRO-2 vs DDP after 60 steps:  max |Δw| = 0.0
ZeRO-3 vs DDP after 60 steps:  max |Δw| = 0.0
```

![loss](figures/loss.png)

This is the central correctness check. ZeRO is neither an approximation nor a
different optimizer; it is solely a decision about *which GPU holds which
bytes*. The fabric sums gradients in fp32 in a fixed order, and Adam's update
is element-wise, so a shard of the update contains exactly the same values as
the corresponding slice of the full update.

### Scaling with N

![scan](figures/scan_n.png)

Same model, N = 1, 2, 4, 8, 16, 32. DDP is flat: additional GPUs do not
reduce per-GPU memory. ZeRO-1/2/3 decrease as 1/N. ZeRO-3 bends toward the
dashed line — the largest gathered layer — which is the floor beneath which it
cannot go. (At N = 1 all four strategies are the same: one GPU holding
everything.)

## 4. Implications for Production-Scale Models

Because the simulator reproduces the paper's formulas exactly, those formulas
can be applied with confidence at sizes that cannot be simulated. Model states
only, on 32 × 80 GB GPUs:

| model | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| 1 B | 16 GB | 4.4 GB | 2.4 GB | 0.5 GB |
| 7 B | 112 GB ✗ | 30.6 GB | 17.1 GB | 3.5 GB |
| 13 B | 208 GB ✗ | 56.9 GB | 31.7 GB | 6.5 GB |
| 70 B | 1120 GB ✗ | 306 GB ✗ | 171 GB ✗ | 35 GB |
| 175 B | 2800 GB ✗ | 766 GB ✗ | 428 GB ✗ | 87.5 GB ✗ |

A 7B model does not fit under DDP at all, fits comfortably under ZeRO-1, and
under ZeRO-3 is small enough that activations become the dominant concern. A
70B model requires ZeRO-3 on 32 GPUs; a 175B model requires more GPUs or
techniques beyond ZeRO.

## 5. Conclusions

1. **The cost of data parallelism is replication.** Thirty-two GPUs hold
   thirty-two identical copies of 16 bytes per parameter; only the
   activations ever differ.
2. **ZeRO-1 removes the largest component at no cost.** Twelve of the sixteen
   bytes are read only by the optimizer, and the optimizer is element-wise, so
   each GPU can own 1/N and update only that portion. The reduce-scatter and
   all-gather it requires are the two halves of the all-reduce that DDP was
   already performing.
3. **ZeRO-2 is ZeRO-1 with improved timing.** Scattering each gradient *as it
   is produced* means the full gradient never exists. The bytes on the
   interconnect are unchanged.
4. **ZeRO-3 is the first stage with a cost.** Partitioning the weights
   requires gathering every layer twice per step: memory falls to 16Ψ/N plus
   one layer, and communication rises by 1.5×. This is justified when the
   model does not otherwise fit; ZeRO-2 is otherwise the preferred operating
   point.
5. **Compute is unaffected.** Identical FLOPs, identical loss, bit-identical
   weights. The optimizer step becomes N× cheaper.
6. **Two quantities ZeRO does not reduce:** activations (per-GPU work is
   unchanged) and the largest single layer (which must exist in full while in
   use). These require activation checkpointing and tensor or pipeline
   parallelism respectively, which is why ZeRO is typically combined with
   them.

## Limitations

* The fabric performs its arithmetic once and distributes copies; bandwidth
  and latency are not simulated, so bytes and call counts are reported rather
  than seconds. A real ring all-reduce also carries an N-dependent latency
  term that is not modelled.
* No bucketing, prefetching, or overlap of communication with compute is
  implemented — the areas in which DeepSpeed and FSDP concentrate most of
  their engineering. The simulator therefore shows the *floor* of each stage,
  not its tuned performance.
* One micro-batch per GPU per step; no gradient accumulation.
* Wall-clock time in the notebook reflects the Python interpreter scheduling
  32 threads, not GPU time, and is labelled accordingly wherever it appears.

## References

* Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. *ZeRO: Memory
  Optimizations Toward Training Trillion Parameter Models*, 2020. Figure 1
  provides the memory table and §7 the communication analysis; every figure
  in this document is checked against it.
