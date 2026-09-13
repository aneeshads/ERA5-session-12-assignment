"""Generates assignment.ipynb.  Run:  python3 build_notebook.py && jupyter nbconvert --execute --to notebook --inplace assignment.ipynb"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
SIM_SRC = (HERE / "zero_sim.py").read_text()

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# ZeRO-1, ZeRO-2, ZeRO-3 on 32 virtual GPUs

**ERA V5 — Session 12 assignment.**

I build 32 virtual GPUs out of CPU threads, train one small model on them with
plain data parallelism (DDP), then with ZeRO stages 1, 2 and 3, and measure
what changes: **memory per GPU**, **bytes moved over the interconnect**, and
**work per GPU**.  Every stage is checked to produce the *same* weights,
bit for bit — ZeRO is a memory trick, not a different optimizer.

The simulator is `zero_sim.py` (written next to this notebook by the first
cell if it is missing, so this runs on Colab with no setup).

**What is real and what is pretend**

| | |
|---|---|
| a *virtual GPU* | a Python object with a private memory ledger and its own thread. "Memory" = bytes of tensors currently held on it — measured, not estimated |
| the *fabric* | one object that performs all-reduce / reduce-scatter / all-gather and charges each GPU what a ring implementation would move |
| the model | a 4-layer MLP, forward **and backward written by hand**, so the ZeRO-3 gather / free / re-gather dance is visible line by line |
| mixed precision | real: bf16 params and grads, fp32 master weights and Adam moments — that is the famous 16 bytes/parameter |
| bf16 matmul | this CPU has no fast bf16 kernel, so the multiply runs in fp32 and rounds to bf16 — which is exactly what a tensor core does |
| wall-clock time | **not** a proxy for GPU time — 32 Python threads share one GIL. I report analytic counts and an isolated micro-benchmark instead |
""")

code(f'''
# Self-contained: write the simulator next to the notebook if it is not there (e.g. on Colab).
import os, textwrap
if not os.path.exists("zero_sim.py"):
    open("zero_sim.py", "w").write({SIM_SRC!r})

import math, time, torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from zero_sim import *

MiB = 2 ** 20
STAGE_NAMES = {{0: "DDP (no ZeRO)", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}}
STAGE_COLOR = {{0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#eda100"}}
CAT_ORDER = [CAT_P, CAT_G, CAT_M, CAT_A1, CAT_A2, CAT_ACT, CAT_TMP]
CAT_COLOR = dict(zip(CAT_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]))
mpl.rcParams.update({{"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10}})
os.makedirs("figures", exist_ok=True)
print("torch", torch.__version__, "| cpu threads available:", os.cpu_count())
''')

md(r"""
## 1. The virtual hardware

A GPU, for the purpose of this assignment, is *a box of memory with a rank
number*.  `hold()` puts a tensor in the box and charges the ledger;
`drop()` frees it.  The fabric between the boxes performs the three
collectives ZeRO is built from and bills each GPU the ring cost:

| collective | what happens | bytes each GPU sends+receives |
|---|---|---|
| **all-reduce** | everyone contributes a full tensor, everyone gets the mean | `2·(N−1)/N · size` |
| **reduce-scatter** | everyone contributes a full tensor, GPU *i* gets only chunk *i* of the mean | `(N−1)/N · size` |
| **all-gather** | GPU *i* contributes chunk *i*, everyone gets the full tensor | `(N−1)/N · size` |

An all-reduce *is* a reduce-scatter followed by an all-gather — that identity is
the whole reason ZeRO-1 and ZeRO-2 cost no extra communication.
""")

code(r'''
N = 32
gpus = [VirtualGPU(r) for r in range(N)]
fabric = Fabric(N)

# every GPU holds a different 1 MiB-ish bf16 tensor ...
for g in gpus:
    g.hold("demo", CAT_G, torch.full((512 * 1024,), float(g.rank), dtype=BF16))
print(f"GPU 0 holds {gpus[0].total()/MiB:.2f} MiB   ({gpus[0].snapshot()})")

# ... reduce-scatter: everyone hands in the full tensor, gets back 1/32 of the mean
shards = fabric.reduce_scatter([g.drop("demo") for g in gpus])
for g, s in zip(gpus, shards):
    g.hold("demo_shard", CAT_G, s)
print(f"after reduce-scatter GPU 0 holds {gpus[0].total()/MiB:.3f} MiB, values all = {shards[0][0].item()} (mean of 0..31)")
print(f"fabric charged each GPU {fabric.bytes_per_gpu/MiB:.3f} MiB  = (N-1)/N x 1 MiB")

# ... all-gather: everyone hands in their shard, gets the full tensor back
full = fabric.all_gather([g.drop("demo_shard") for g in gpus])
print(f"after all-gather each GPU has {full[0].numel()*2/MiB:.2f} MiB again; fabric total {fabric.bytes_per_gpu/MiB:.3f} MiB per GPU = one all-reduce")
''')

md(r"""
## 2. The demo model and where 16 bytes per parameter comes from

A 4-layer MLP, 256 → 1024 → 1024 → 1024 → 10, about **2.37 M parameters (Ψ)**,
trained on a synthetic 10-class task labelled by a fixed random "teacher"
network.  Small on purpose: 32 full copies must fit in laptop RAM.

With mixed-precision Adam, every parameter costs on every GPU:

| what | dtype | bytes |
|---|---|---|
| working copy of the weight (used in forward/backward) | bf16 | 2 |
| its gradient | bf16 | 2 |
| master copy of the weight (updated by the optimizer) | fp32 | 4 |
| Adam first moment *m* | fp32 | 4 |
| Adam second moment *v* | fp32 | 4 |
| | | **16** |

The ZeRO paper calls the last three (12 bytes, "K = 12") the *optimizer
states*.  Plain data parallelism replicates all 16 bytes on all N GPUs.
""")

code(r'''
spec = ModelSpec.build(DEFAULT_DIMS, 32)
print(f"{'param':8s} {'shape':>14s} {'count':>10s} {'per-GPU shard':>14s}")
for p in spec.params:
    print(f"{p.name:8s} {str(p.shape):>14s} {p.numel:>10,d} {p.shard:>14,d}")
psi = spec.total_params
print(f"\nΨ = {psi:,} parameters -> 16 Ψ = {16*psi/MiB:.2f} MiB of model states per GPU under DDP")
print(f"largest layer = {spec.largest_layer_params:,} params = {2*spec.largest_layer_params/MiB:.2f} MiB in bf16  (matters for ZeRO-3, see §5)")
''')

md(r"""
## 3. Train with all four strategies

One training step, in lockstep across the 32 GPUs.  What each stage does differently:

| | params (bf16) | grads (bf16) | master + Adam (fp32) | collectives per step |
|---|---|---|---|---|
| **DDP** | full copy | full copy | full copy | all-reduce grads after backward |
| **ZeRO-1** | full copy | full copy *during* backward | **my 1/N** | reduce-scatter grads after backward → update my shard → all-gather params |
| **ZeRO-2** | full copy | **my 1/N** — each layer's grad is reduce-scattered the moment backward produces it | my 1/N | (same two collectives, different timing) |
| **ZeRO-3** | **my 1/N** — a layer is all-gathered just before use and freed right after, in forward *and again* in backward | my 1/N | my 1/N | all-gather ×2 per layer + reduce-scatter |
""")

code(r'''
STEPS = 60
results = {}
for stage in range(4):
    results[stage] = run_stage(stage, n_gpus=32, steps=STEPS)
''')

md(r"""
### 3.1 Same weights, same loss

The point of the check: ZeRO changes *where* the numbers live, not *what*
they are.  Every stage must end with identical weights.
""")

code(r'''
ref = results[0]["final_params"].float()
for s in (1, 2, 3):
    d = (results[s]["final_params"].float() - ref).abs().max().item()
    print(f"{STAGE_NAMES[s]:14s} vs DDP after {STEPS} steps:  max |Δw| = {d}")

fig, ax = plt.subplots(figsize=(7, 3.4))
for s in range(4):
    ax.plot(range(1, STEPS + 1), results[s]["losses"], color=STAGE_COLOR[s], lw=2 if s == 0 else 1.2,
            ls=["-", "--", "-.", ":"][s], label=STAGE_NAMES[s])
ax.set(xlabel="step", ylabel="cross-entropy loss", title="All four strategies trace the identical loss curve")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/loss.png"); plt.show()
''')

md(r"""
## 4. Memory per GPU

Peak memory on a GPU, split by what is in it, and next to it the paper's
formula (Rajbhandari et al. 2020, Figure 1).  The paper counts only *model
states* (params + grads + optimizer); the simulator also sees activations and
ZeRO-3's temporarily gathered layer.
""")

code(r'''
print(f"{'strategy':14s} {'peak / GPU':>11s} {'model states':>13s} {'paper':>8s}   {'formula':18s} {'vs DDP':>7s}")
formulas = {0: "16Ψ", 1: "4Ψ + 12Ψ/N", 2: "2Ψ + 14Ψ/N", 3: "16Ψ/N"}
for s in range(4):
    r = results[s]
    print(f"{STAGE_NAMES[s]:14s} {r['peak_bytes']/MiB:8.2f} MiB {r['peak_model_state_bytes']/MiB:9.2f} MiB "
          f"{r['paper_model_state_bytes']/MiB:8.2f}   {formulas[s]:18s} {results[0]['peak_bytes']/r['peak_bytes']:6.1f}x")

fig, ax = plt.subplots(figsize=(8, 3.8))
for s in range(4):
    bottom = 0
    for cat in CAT_ORDER:
        v = results[s]["peak_breakdown"].get(cat, 0) / MiB
        if v:
            ax.bar(s, v, bottom=bottom, color=CAT_COLOR[cat], width=0.6, edgecolor="white", linewidth=1.5,
                   label=cat if s == 0 or cat in (CAT_ACT, CAT_TMP) and s == 3 else None)
            bottom += v
    ax.text(s, bottom + 0.6, f"{bottom:.1f} MiB", ha="center", fontsize=9)
ax.set(xticks=range(4), xticklabels=[STAGE_NAMES[s] for s in range(4)], ylabel="MiB on one GPU (at its peak)",
       title=f"Peak memory per GPU, Ψ = {psi/1e6:.2f} M params, N = 32")
h, l = ax.get_legend_handles_labels(); ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/peak_memory.png"); plt.show()
''')

md(r"""
### 4.1 Memory *during* one step

The peak number hides the mechanism.  Here is GPU 0's ledger after every
phase of one step, for each strategy.  Read left to right:

* **DDP / ZeRO-1** — memory *rises* through backward as full gradients pile
  up, then drops after the optimizer frees them.  ZeRO-1 starts 26 MiB lower
  because it owns only 1/32 of the fp32 states.
* **ZeRO-2** — memory *falls* through backward: each layer's gradient is
  reduce-scattered the moment it exists and only 1/32 of it is kept.  The full
  gradient never sits in memory.  Same collectives as ZeRO-1, different
  timing — that is the entire difference between the two stages.
* **ZeRO-3** — a sawtooth.  Each tooth is one layer's weights being gathered
  (2 MiB for a 1024×1024 layer) and freed.  Notice the teeth in backward too:
  the weights are gathered a *second* time, which is where the extra
  communication comes from.
""")

code(r'''
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=False)
for s, ax in zip(range(4), axes):
    tl = results[s]["timeline"]
    labels = [l for l, _ in tl]
    x = range(len(tl))
    bottom = [0.0] * len(tl)
    for cat in CAT_ORDER:
        vals = [b.get(cat, 0) / MiB for _, b in tl]
        if any(vals):
            ax.bar(x, vals, bottom=bottom, color=CAT_COLOR[cat], width=0.85, label=cat, edgecolor="white", linewidth=0.8)
            bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set(title=STAGE_NAMES[s], xticks=list(x))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(0, max(bottom) * 1.15)
    ax.text(0.02, 0.97, f"peak {max(bottom):.1f} MiB", transform=ax.transAxes, va="top", fontsize=9)
axes[0].set_ylabel("MiB on GPU 0")
h, l = axes[3].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.12))
fig.suptitle("Memory on one GPU through one training step", y=1.02)
fig.tight_layout(); fig.savefig("figures/timeline.png", bbox_inches="tight"); plt.show()
''')

md(r"""
## 5. Communication per GPU

Bytes each GPU sends and receives per step, measured by the fabric, next to the
paper's claim: DDP, ZeRO-1 and ZeRO-2 all move **2Ψ elements** per step;
ZeRO-3 moves **3Ψ** — 1.5×.

Why: an all-reduce of the gradient (2Ψ) is *exactly* a reduce-scatter (Ψ) plus
an all-gather (Ψ).  ZeRO-1/2 do the reduce-scatter on the gradient and the
all-gather on the *updated parameters* instead — same bytes, but the
parameters come back already updated, so nobody needs the full gradient.
ZeRO-3 cannot keep the gathered parameters between forward and backward
(that would cost 2Ψ of memory again), so it gathers them twice: +Ψ.
""")

code(r'''
print(f"{'strategy':14s} {'measured':>10s} {'paper':>8s}  collectives per step")
for s in range(4):
    r = results[s]
    calls = ", ".join(f"{int(v)} {k.replace('_', '-')}" for k, v in r["collective_calls_per_step"].items())
    print(f"{STAGE_NAMES[s]:14s} {r['comm_bytes_per_gpu_per_step']/MiB:7.2f} MiB {r['paper_comm_bytes']/MiB:5.2f} MiB  {calls}")

fig, ax = plt.subplots(figsize=(7, 3.4))
meas = [results[s]["comm_bytes_per_gpu_per_step"] / MiB for s in range(4)]
paper = [results[s]["paper_comm_bytes"] / MiB for s in range(4)]
ax.bar([s - 0.18 for s in range(4)], meas, width=0.34, color=[STAGE_COLOR[s] for s in range(4)], label="measured by the fabric")
ax.bar([s + 0.18 for s in range(4)], paper, width=0.34, color="none", edgecolor="#555", hatch="///", label="paper formula")
for s in range(4):
    ax.text(s - 0.18, meas[s] + 0.2, f"{meas[s]:.1f}", ha="center", fontsize=8)
ax.set(xticks=range(4), xticklabels=[STAGE_NAMES[s] for s in range(4)], ylabel="MiB per GPU per step",
       title="Bytes moved per GPU per step (bf16 gradients / parameters)")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/comm.png"); plt.show()
''')

md(r"""
## 6. Computation per GPU

Three separate things, because they behave differently:

1. **Forward + backward FLOPs** — identical in every stage.  Each GPU still
   multiplies its own micro-batch through the *full* model; ZeRO changes
   who *stores* a weight, not who *uses* it.
2. **Optimizer work** — DDP updates all Ψ parameters on every GPU (32× redundant).
   ZeRO-1/2/3 update Ψ/N each.  Measured below with one GPU alone on one
   thread, because per-thread timings inside the 32-thread run are dominated
   by GIL contention and would be misleading.
3. **Waiting on the fabric** — the extra all-gathers of ZeRO-3 are extra
   synchronisation points; real systems hide them by prefetching the next
   layer while computing the current one.
""")

code(r'''
r0 = results[0]
print(f"forward+backward FLOPs per GPU per step (6·Ψ·batch):  {r0['flops_per_gpu_per_step']/1e9:.3f} GFLOP  -- same for every stage")
print(f"optimizer elements updated per GPU per step: DDP {results[0]['optimizer_elements_per_gpu']:,}  |  "
      f"ZeRO-1/2/3 {results[1]['optimizer_elements_per_gpu']:,}  (÷{results[0]['optimizer_elements_per_gpu']//results[1]['optimizer_elements_per_gpu']})")

bench = optimizer_microbench(sum(p.padded for p in spec.params), 32)
print("\nAdam update, one GPU alone, single thread:")
for k, v in bench.items():
    print(f"  {k:14s} {v*1e3:7.3f} ms")
print(f"  -> {list(bench.values())[0]/list(bench.values())[1]:.0f}x less optimizer time per GPU with any ZeRO stage")

print("\ncollective calls per step (synchronisation points):")
for s in range(4):
    print(f"  {STAGE_NAMES[s]:14s} {sum(results[s]['collective_calls_per_step'].values()):.0f}")

print("\nwall-clock of the whole 32-thread simulation per step (ms) — NOT a GPU-time proxy, shown for honesty:")
for s in range(4):
    w = results[s]["wall_per_step"]
    print(f"  {STAGE_NAMES[s]:14s} total {results[s]['wall_total_per_step']*1e3:5.0f}  " +
          "  ".join(f"{k} {v*1e3:4.0f}" for k, v in w.items()))
''')

md(r"""
## 7. How the saving scales with the number of GPUs

Same model, same stage, N = 1, 2, 4, 8, 16, 32.  DDP is flat (every GPU
holds everything, no matter how many there are).  ZeRO-3 falls as 1/N …
until it hits the floor set by the largest single layer that must be
gathered in full (the dashed line).  Activations are also flat: ZeRO does
nothing for them — that is what activation checkpointing is for.
""")

code(r'''
Ns = [1, 2, 4, 8, 16, 32]
scan = {s: [run_stage(s, n_gpus=n, steps=3, verbose=False)["peak_bytes"] / MiB for n in Ns] for s in range(4)}

fig, ax = plt.subplots(figsize=(7, 3.8))
for s in range(4):
    ax.plot(Ns, scan[s], marker="o", ms=6, color=STAGE_COLOR[s], label=STAGE_NAMES[s])
ax.axhline(2 * spec.largest_layer_params / MiB, ls="--", color="#4a3aa7", lw=1)
ax.text(1.05, 2 * spec.largest_layer_params / MiB * 1.1, "largest layer, gathered (ZeRO-3 floor)", fontsize=8, color="#4a3aa7")
ax.set(xscale="log", yscale="log", xticks=Ns, xticklabels=Ns, xlabel="number of GPUs (N)", ylabel="peak MiB per GPU",
       title="Peak memory per GPU vs cluster size")
ax.set_yticks([2, 4, 8, 16, 32]); ax.set_yticklabels(["2", "4", "8", "16", "32"]); ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/scan_n.png"); plt.show()
for s in range(4):
    print(f"{STAGE_NAMES[s]:14s} " + "  ".join(f"N={n}: {v:6.2f}" for n, v in zip(Ns, scan[s])))
''')

md(r"""
## 8. What this means for a real model

The simulator confirmed the paper's formulas to two decimals, so I can trust
them at sizes I cannot simulate.  Model states only (no activations), 32 GPUs
with 80 GB each:
""")

code(r'''
GB = 1e9
print(f"{'model':>8s} | " + " | ".join(f"{STAGE_NAMES[s]:>14s}" for s in range(4)) + "   (per-GPU model states, 32 x 80 GB GPUs)")
for name, P in (("1 B", 1e9), ("7 B", 7e9), ("13 B", 13e9), ("70 B", 70e9), ("175 B", 175e9)):
    row = []
    for s in range(4):
        b = paper_model_state_bytes(P, 32, s) / GB
        row.append(f"{b:8.1f} GB {'ok' if b < 80 else 'NO':>3s}")
    print(f"{name:>8s} | " + " | ".join(row))
''')

md(r"""
## 9. What I take away

1. **Data parallelism wastes memory by replication.** 32 GPUs, 32 identical
   copies of 16 bytes/parameter.  Only the *activations* actually differ
   between GPUs (different micro-batches).
2. **ZeRO-1 removes the biggest redundancy at zero cost.** The 12 bytes/param
   of fp32 master + Adam moments were only ever *read* by the optimizer, and
   the optimizer update is element-wise — so each GPU can own 1/N of them and
   update just those.  The reduce-scatter + all-gather it needs is exactly
   what an all-reduce already was.  3.7× less memory here, and the optimizer
   step gets ~N× cheaper per GPU as a bonus.
3. **ZeRO-2 is ZeRO-1 with better timing.** Reduce-scatter each layer's
   gradient *during* backward instead of after, and the full gradient never
   exists.  Same bytes on the wire (2Ψ), another 2 bytes/param saved.
4. **ZeRO-3 is the first stage that costs something.** Sharding the parameters
   themselves means every layer is all-gathered just-in-time — twice per step,
   because keeping it between forward and backward would undo the saving.
   Memory falls to 16Ψ/N + one gathered layer; communication rises 1.5×.
   This is why ZeRO-3 wants fast interconnects and prefetching.
5. **Compute is untouched.** Same FLOPs, same loss curve, same weights to the
   bit.  ZeRO is purely a question of *where* the 16 bytes live.
6. **Two things ZeRO does not shrink:** activations (the per-GPU work is the
   same), and the largest single layer (must exist in full on one GPU while it
   is used).  Those need activation checkpointing and tensor parallelism
   respectively — which is what the next sessions are for.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, HERE / "assignment.ipynb")
print("wrote assignment.ipynb with", len(cells), "cells")
