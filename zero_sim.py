"""
zero_sim.py — 32 virtual GPUs, one demo model, ZeRO stages 0/1/2/3.

ERA V5, Session 12.

What is real here and what is pretend
-------------------------------------
* A "virtual GPU" is a Python object with a private memory ledger and its own
  worker thread.  Every tensor it holds is registered with the ledger, so
  "memory" means "bytes of tensors currently living on this GPU" — measured,
  not estimated.  Compute really does run on 32 threads at once.
* The "fabric" (interconnect) is one object that performs the collectives —
  all-reduce, reduce-scatter, all-gather — and charges each GPU the bytes a
  ring implementation would move.  The arithmetic is done once in-process;
  the accounting is what we care about.
* The model's forward AND backward are written by hand (no autograd), so the
  ZeRO-3 dance — gather a layer's weights, use them, throw them away, gather
  them again in backward — is visible line by line.
* Mixed precision is real: bf16 working copies of params and grads, fp32
  master weights and Adam moments.  That is where the famous
  "16 bytes per parameter" comes from:  2 + 2 + 4 + 4 + 4.
* This CPU has no fast bf16 matmul, so the multiply runs in fp32 and the
  result is rounded to bf16 — which is exactly what a bf16 tensor core does
  (bf16 in, fp32 accumulate, bf16 out).  Stored tensors stay bf16.

Everything is seeded.  All four strategies produce bit-identical weights.
"""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

torch.set_num_threads(1)  # one intra-op thread per virtual GPU, no oversubscription

BF16 = torch.bfloat16
FP32 = torch.float32

# Memory categories.  The first five are the "model states" of the ZeRO paper.
CAT_P, CAT_G, CAT_M, CAT_A1, CAT_A2 = "params bf16", "grads bf16", "master fp32", "adam m fp32", "adam v fp32"
CAT_ACT, CAT_TMP = "activations", "gathered layer (temp)"
MODEL_STATES = (CAT_P, CAT_G, CAT_M, CAT_A1, CAT_A2)


# ---------------------------------------------------------------------------
# 1. Virtual hardware
# ---------------------------------------------------------------------------
class VirtualGPU:
    """A GPU, for our purposes: a private memory pool with a rank number.

    hold(name, category, tensor) puts a tensor 'on' the GPU;  drop(name) frees
    it.  The ledger tracks current bytes per category, the peak total (with a
    snapshot of the breakdown at that moment) and the peak of model states.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self._held: dict[str, tuple[str, torch.Tensor]] = {}
        self.cur: dict[str, int] = defaultdict(int)
        self.peak = 0
        self.peak_breakdown: dict[str, int] = {}
        self.peak_model_states = 0
        self.busy = defaultdict(float)  # seconds this GPU spent computing, per phase

    def hold(self, name: str, category: str, t: torch.Tensor) -> torch.Tensor:
        assert name not in self._held, f"gpu{self.rank}: {name!r} already held"
        self._held[name] = (category, t)
        self.cur[category] += t.numel() * t.element_size()
        total = self.total()
        if total > self.peak:
            self.peak = total
            self.peak_breakdown = self.snapshot()
        self.peak_model_states = max(self.peak_model_states, self.model_states())
        return t

    def drop(self, name: str) -> torch.Tensor:
        cat, t = self._held.pop(name)
        self.cur[cat] -= t.numel() * t.element_size()
        return t

    def get(self, name: str) -> torch.Tensor:
        return self._held[name][1]

    def total(self) -> int:
        return sum(self.cur.values())

    def model_states(self) -> int:
        return sum(self.cur[c] for c in MODEL_STATES)

    def snapshot(self) -> dict[str, int]:
        return {k: v for k, v in self.cur.items() if v}


class Fabric:
    """The interconnect between the GPUs.

    Every collective is a synchronisation point: nobody continues until all
    N inputs are in.  Bytes charged per GPU follow the ring algorithm:

        all_reduce      2 (N-1)/N  x  size        (= reduce-scatter + all-gather)
        reduce_scatter    (N-1)/N  x  size
        all_gather        (N-1)/N  x  size

    Sums are done in fp32 in a fixed order so every stage sees exactly the
    same reduced gradient.
    """

    def __init__(self, n: int):
        self.n = n
        self.bytes_per_gpu = 0.0
        self.calls: dict[str, int] = defaultdict(int)
        self.seconds = 0.0
        self._lock = threading.Lock()

    def _charge(self, kind: str, full_nbytes: int, factor: float, t0: float):
        with self._lock:
            self.bytes_per_gpu += factor * (self.n - 1) / self.n * full_nbytes
            self.calls[kind] += 1
            self.seconds += time.perf_counter() - t0

    def _mean(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([t.float() for t in tensors]).sum(0).div_(self.n)

    def all_reduce(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Everyone contributes a full tensor, everyone gets back the mean."""
        t0 = time.perf_counter()
        assert len(tensors) == self.n
        mean = self._mean(tensors).to(tensors[0].dtype)
        out = [mean.clone() for _ in range(self.n)]
        self._charge("all_reduce", tensors[0].numel() * tensors[0].element_size(), 2.0, t0)
        return out

    def reduce_scatter(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """Everyone contributes a full tensor; GPU i gets back only chunk i of the mean."""
        t0 = time.perf_counter()
        assert len(tensors) == self.n
        chunks = self._mean(tensors).to(tensors[0].dtype).chunk(self.n)
        out = [c.clone() for c in chunks]
        self._charge("reduce_scatter", tensors[0].numel() * tensors[0].element_size(), 1.0, t0)
        return out

    def all_gather(self, shards: list[torch.Tensor]) -> list[torch.Tensor]:
        """GPU i contributes chunk i; everyone gets back the full tensor."""
        t0 = time.perf_counter()
        assert len(shards) == self.n
        full = torch.cat(shards)
        out = [full.clone() for _ in range(self.n)]
        self._charge("all_gather", full.numel() * full.element_size(), 1.0, t0)
        return out


# ---------------------------------------------------------------------------
# 2. The demo model: a 4-layer MLP with hand-written forward and backward
# ---------------------------------------------------------------------------
@dataclass
class ParamSpec:
    name: str
    shape: tuple[int, ...]
    numel: int
    padded: int          # numel rounded up to a multiple of n_gpus so it shards evenly
    n_gpus: int

    @property
    def shard(self) -> int:
        return self.padded // self.n_gpus


@dataclass
class ModelSpec:
    dims: tuple[int, ...]
    n_gpus: int
    params: list[ParamSpec] = field(default_factory=list)

    @staticmethod
    def build(dims: tuple[int, ...], n_gpus: int) -> "ModelSpec":
        spec = ModelSpec(dims, n_gpus)
        for i in range(len(dims) - 1):
            for nm, shape in ((f"L{i}.W", (dims[i + 1], dims[i])), (f"L{i}.b", (dims[i + 1],))):
                numel = math.prod(shape)
                padded = math.ceil(numel / n_gpus) * n_gpus
                spec.params.append(ParamSpec(nm, shape, numel, padded, n_gpus))
        return spec

    @property
    def n_layers(self) -> int:
        return len(self.dims) - 1

    def layer_params(self, i: int) -> tuple[ParamSpec, ParamSpec]:
        return self.params[2 * i], self.params[2 * i + 1]

    @property
    def total_params(self) -> int:
        return sum(p.numel for p in self.params)

    @property
    def largest_layer_params(self) -> int:
        return max(self.params[2 * i].numel + self.params[2 * i + 1].numel for i in range(self.n_layers))

    def init_master(self, seed: int) -> dict[str, torch.Tensor]:
        """fp32 master weights, flat + padded, uniform(-1/sqrt(fan_in), +1/sqrt(fan_in)) like nn.Linear."""
        g = torch.Generator().manual_seed(seed)
        out = {}
        for i in range(self.n_layers):
            W, b = self.layer_params(i)
            bound = 1.0 / math.sqrt(self.dims[i])
            w = (torch.rand(W.numel, generator=g) * 2 - 1) * bound
            bb = (torch.rand(b.numel, generator=g) * 2 - 1) * bound
            out[W.name] = torch.cat([w, torch.zeros(W.padded - W.numel)])
            out[b.name] = torch.cat([bb, torch.zeros(b.padded - b.numel)])
        return out


def mm_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Emulates a bf16 tensor-core matmul: bf16 in, fp32 accumulate, bf16 out."""
    return (a.float() @ b.float()).to(BF16)


def view_param(flat: torch.Tensor, p: ParamSpec) -> torch.Tensor:
    return flat[: p.numel].view(p.shape)


def layer_forward(gpu: VirtualGPU, i: int, spec: ModelSpec, x: torch.Tensor,
                  W: torch.Tensor, b: torch.Tensor, last: bool) -> torch.Tensor:
    """y = relu(x W^T + b)  (no relu on the last layer).  Saves what backward needs."""
    Wp, _ = spec.layer_params(i)
    gpu.hold(f"act/L{i}.in", CAT_ACT, x)                    # needed for dW
    h = mm_bf16(x, view_param(W, Wp).T) + b[: spec.dims[i + 1]]
    if last:
        return h
    mask = h > 0
    gpu.hold(f"act/L{i}.mask", CAT_ACT, mask)                # needed for relu backward
    return h * mask


def layer_backward(gpu: VirtualGPU, i: int, spec: ModelSpec, dout: torch.Tensor,
                   W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Given dL/dy for layer i, return (dW_flat, db_flat, dL/dx).  Frees saved activations."""
    Wp, bp = spec.layer_params(i)
    x = gpu.drop(f"act/L{i}.in")
    dW = mm_bf16(dout.T, x).reshape(-1)                      # (out,in) -> flat
    db = dout.float().sum(0).to(BF16)
    dW_flat = torch.cat([dW, dW.new_zeros(Wp.padded - Wp.numel)])
    db_flat = torch.cat([db, db.new_zeros(bp.padded - bp.numel)])
    if i == 0:
        dx = None
    else:
        dx = mm_bf16(dout, view_param(W, Wp))                 # (B,out) @ (out,in)
        mask = gpu.drop(f"act/L{i-1}.mask")
        dx = dx * mask
    return dW_flat, db_flat, dx


def loss_and_grad(logits_bf16: torch.Tensor, y: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Cross-entropy in fp32; returns (loss, dL/dlogits in bf16)."""
    logits = logits_bf16.float()
    logp = torch.log_softmax(logits, dim=1)
    loss = -logp.gather(1, y[:, None]).mean()
    d = torch.softmax(logits, dim=1)
    d[torch.arange(len(y)), y] -= 1.0
    d /= len(y)
    return loss.item(), d.to(BF16)


def adam_update(master: torch.Tensor, grad: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                step: int, lr: float, b1=0.9, b2=0.999, eps=1e-8) -> None:
    """Plain Adam, in place, on fp32 tensors.  Same maths whether given a full tensor or a shard."""
    g = grad.float()
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    m_hat = m / (1 - b1 ** step)
    v_hat = v / (1 - b2 ** step)
    master.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


# ---------------------------------------------------------------------------
# 3. Synthetic data: a fixed random "teacher" labels random inputs
# ---------------------------------------------------------------------------
class TeacherData:
    def __init__(self, dims: tuple[int, ...], seed: int):
        g = torch.Generator().manual_seed(seed)
        self.in_dim, self.n_cls = dims[0], dims[-1]
        self.W1 = torch.randn(dims[0], 128, generator=g) / math.sqrt(dims[0])
        self.W2 = torch.randn(128, dims[-1], generator=g) / math.sqrt(128)
        self.seed = seed

    def batch(self, step: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 100_003 + step)
        x = torch.randn(n, self.in_dim, generator=g)
        y = (torch.tanh(x @ self.W1) @ self.W2).argmax(1)
        return x, y


# ---------------------------------------------------------------------------
# 4. The four strategies
# ---------------------------------------------------------------------------
class ZeroSimulator:
    """Trains one model with data parallelism across N virtual GPUs at ZeRO stage 0, 1, 2 or 3.

    stage 0 : classic DDP.  Every GPU: full params, full grads, full optimizer state.
              all-reduce grads at the end of backward.
    stage 1 : optimizer state sharded (each GPU owns 1/N of master, m, v).
              reduce-scatter grads AFTER backward -> update my shard -> all-gather bf16 params.
    stage 2 : + gradients sharded: reduce-scatter each layer's grad DURING backward, the moment it
              exists, so the full gradient never sits in memory.
    stage 3 : + parameters sharded: all-gather a layer just before it is used — in forward AND
              again in backward — and free it right after.
    """

    def __init__(self, stage: int, spec: ModelSpec, seed: int = 0, lr: float = 1e-3,
                 per_gpu_batch: int = 64):
        assert stage in (0, 1, 2, 3)
        self.stage, self.spec, self.n, self.lr = stage, spec, spec.n_gpus, lr
        self.B = per_gpu_batch
        self.gpus = [VirtualGPU(r) for r in range(self.n)]
        self.fabric = Fabric(self.n)
        self.pool = ThreadPoolExecutor(max_workers=self.n)
        self.data = TeacherData(spec.dims, seed + 1)
        self.wall = defaultdict(float)          # wall-clock per phase (the whole cluster)
        self.timeline: list[tuple[str, dict[str, int]]] = []   # GPU-0 memory after each phase, last step
        self.optimizer_elements_per_gpu = 0
        self._load(spec.init_master(seed))

    # -- helpers -------------------------------------------------------------
    def par(self, phase: str, fn, *iters):
        """Run fn(gpu, ...) on every GPU concurrently, one thread each, timing each GPU's own work."""
        def timed(gpu, *args):
            t0 = time.perf_counter()
            out = fn(gpu, *args)
            gpu.busy[phase] += time.perf_counter() - t0
            return out
        return list(self.pool.map(timed, self.gpus, *iters))

    def _mark(self, label: str):
        self.timeline.append((label, self.gpus[0].snapshot()))

    def _load(self, master: dict[str, torch.Tensor]):
        """Initial placement.  Like loading a checkpoint: not charged to the fabric."""
        sharded_opt = self.stage >= 1
        sharded_params = self.stage == 3
        for p in self.spec.params:
            full = master[p.name]
            for gpu in self.gpus:
                r = gpu.rank
                sl = slice(r * p.shard, (r + 1) * p.shard)
                if sharded_opt:
                    gpu.hold(f"{p.name}/master", CAT_M, full[sl].clone())
                    gpu.hold(f"{p.name}/m", CAT_A1, torch.zeros(p.shard))
                    gpu.hold(f"{p.name}/v", CAT_A2, torch.zeros(p.shard))
                else:
                    gpu.hold(f"{p.name}/master", CAT_M, full.clone())
                    gpu.hold(f"{p.name}/m", CAT_A1, torch.zeros(p.padded))
                    gpu.hold(f"{p.name}/v", CAT_A2, torch.zeros(p.padded))
                if sharded_params:
                    gpu.hold(f"{p.name}/bf16", CAT_P, full[sl].to(BF16))
                else:
                    gpu.hold(f"{p.name}/bf16", CAT_P, full.to(BF16))

    # -- one training step ---------------------------------------------------
    def step(self, step_no: int) -> float:
        spec, n = self.spec, self.n
        self.timeline = []
        self._mark("start")
        x_all, y_all = self.data.batch(step_no, self.B * n)
        xs = [x_all[r * self.B:(r + 1) * self.B].to(BF16) for r in range(n)]
        ys = [y_all[r * self.B:(r + 1) * self.B] for r in range(n)]

        # ---- forward, layer by layer (lockstep, so ZeRO-3 can gather between layers)
        t0 = time.perf_counter()
        acts = xs
        for i in range(spec.n_layers):
            Wp, bp = spec.layer_params(i)
            if self.stage == 3:
                self._gather_layer(i)
                self._mark(f"fwd L{i} gathered")
            last = i == spec.n_layers - 1
            acts = self.par("forward", lambda g, a, i=i, last=last: layer_forward(
                g, i, spec, a, g.get(self._wname(Wp)), g.get(self._wname(bp)), last), acts)
            if self.stage == 3:
                self._free_layer(i)
            self._mark(f"fwd L{i}")
        losses_and_d = [loss_and_grad(a, y) for a, y in zip(acts, ys)]
        loss = sum(l for l, _ in losses_and_d) / n
        douts = [d for _, d in losses_and_d]
        self.wall["forward"] += time.perf_counter() - t0

        # ---- backward, layer by layer, last layer first
        t0 = time.perf_counter()
        for i in reversed(range(spec.n_layers)):
            Wp, bp = spec.layer_params(i)
            if self.stage == 3:
                self._gather_layer(i)                                  # 2nd gather of this layer
                self._mark(f"bwd L{i} gathered")
            res = self.par("backward", lambda g, d, i=i: layer_backward(g, i, spec, d, g.get(self._wname(Wp))), douts)
            if self.stage == 3:
                self._free_layer(i)
            douts = [r[2] for r in res]
            for p, k in ((Wp, 0), (bp, 1)):
                grads = [r[k] for r in res]
                if self.stage >= 2:
                    # ZeRO-2/3: reduce-scatter this layer's grad right away; keep only my shard
                    self._collect(lambda: [g.hold(f"{p.name}/grad", CAT_G, s)
                                           for g, s in zip(self.gpus, self.fabric.reduce_scatter(grads))])
                else:
                    for g, gr in zip(self.gpus, grads):
                        g.hold(f"{p.name}/grad", CAT_G, gr)            # full grad, reduced after backward
            self._mark(f"bwd L{i}")
        self.wall["backward"] += time.perf_counter() - t0

        # ---- gradient reduction after backward (stage 0: all-reduce, stage 1: reduce-scatter)
        if self.stage == 0:
            for p in spec.params:
                self._collect(lambda: [g.hold(f"{p.name}/grad", CAT_G, r) for g, r in zip(
                    self.gpus, self.fabric.all_reduce([g.drop(f"{p.name}/grad") for g in self.gpus]))])
            self._mark("all-reduce grads")
        elif self.stage == 1:
            for p in spec.params:
                self._collect(lambda: [g.hold(f"{p.name}/grad", CAT_G, s) for g, s in zip(
                    self.gpus, self.fabric.reduce_scatter([g.drop(f"{p.name}/grad") for g in self.gpus]))])
            self._mark("reduce-scatter grads")

        # ---- optimizer step: every GPU updates what it owns
        t0 = time.perf_counter()

        def opt(gpu: VirtualGPU) -> int:
            count = 0
            for p in spec.params:
                grad = gpu.drop(f"{p.name}/grad")
                master, m, v = gpu.get(f"{p.name}/master"), gpu.get(f"{p.name}/m"), gpu.get(f"{p.name}/v")
                adam_update(master, grad, m, v, step_no, self.lr)
                count += master.numel()
                if self.stage == 0:
                    gpu.get(f"{p.name}/bf16").copy_(master.to(BF16))     # full copy, refreshed locally
                elif self.stage == 3:
                    gpu.get(f"{p.name}/bf16").copy_(master.to(BF16))     # my shard, refreshed locally
            return count

        self.optimizer_elements_per_gpu = self.par("optimizer", opt)[0]
        self.wall["optimizer"] += time.perf_counter() - t0
        self._mark("optimizer")

        # ---- stage 1/2: my updated shard -> everyone's full bf16 params
        if self.stage in (1, 2):
            for p in spec.params:
                shards = [g.get(f"{p.name}/master").to(BF16) for g in self.gpus]
                self._collect(lambda: [g.get(f"{p.name}/bf16").copy_(f)
                                       for g, f in zip(self.gpus, self.fabric.all_gather(shards))])
            self._mark("all-gather params")
        return loss

    # -- helpers for the collectives ----------------------------------------
    def _collect(self, fn):
        t0 = time.perf_counter()
        fn()
        self.wall["collectives"] += time.perf_counter() - t0

    def _wname(self, p: ParamSpec) -> str:
        return f"{p.name}/full" if self.stage == 3 else f"{p.name}/bf16"

    def _gather_layer(self, i: int):
        for p in self.spec.layer_params(i):
            self._collect(lambda: [g.hold(f"{p.name}/full", CAT_TMP, f) for g, f in zip(
                self.gpus, self.fabric.all_gather([g.get(f"{p.name}/bf16") for g in self.gpus]))])

    def _free_layer(self, i: int):
        for p in self.spec.layer_params(i):
            for g in self.gpus:
                g.drop(f"{p.name}/full")

    # -- results -------------------------------------------------------------
    def gathered_params(self) -> torch.Tensor:
        """Full bf16 weights (gathered off the books for stage 3), unpadded, as one flat vector."""
        out = []
        for p in self.spec.params:
            if self.stage == 3:
                out.append(torch.cat([g.get(f"{p.name}/bf16") for g in self.gpus])[: p.numel])
            else:
                out.append(self.gpus[0].get(f"{p.name}/bf16")[: p.numel])
        return torch.cat(out)

    def close(self):
        self.pool.shutdown()


# ---------------------------------------------------------------------------
# 5. The paper's formulas, to compare with what the ledger measured
# ---------------------------------------------------------------------------
def paper_model_state_bytes(psi: int, n: int, stage: int) -> float:
    """Per-GPU bytes of model states (params + grads + optimizer), mixed-precision Adam (K = 12).
    Rajbhandari et al. 2020, Figure 1."""
    return {0: 16 * psi,
            1: 4 * psi + 12 * psi / n,
            2: 2 * psi + 14 * psi / n,
            3: 16 * psi / n}[stage]


def paper_comm_bytes(psi: int, n: int, stage: int) -> float:
    """Per-GPU bytes moved per step with bf16 params/grads and ring collectives:
    2*psi elements (=4*psi bytes) for DDP / ZeRO-1 / ZeRO-2, 3*psi elements for ZeRO-3."""
    return (n - 1) / n * (3 if stage == 3 else 2) * 2 * psi


def optimizer_microbench(psi_padded: int, n_gpus: int, reps: int = 50, trials: int = 5) -> dict[str, float]:
    """One GPU, alone, single thread: seconds for an Adam update over psi elements vs psi/n
    (best of `trials`, each averaged over `reps`).  Isolated on purpose: 32 Python threads
    contending for the GIL make per-thread timings inside the simulation meaningless."""
    out = {}
    for label, size in (("full (DDP)", psi_padded), (f"shard (1/{n_gpus})", psi_padded // n_gpus)):
        master, grad, m, v = torch.randn(size), torch.randn(size).to(BF16), torch.zeros(size), torch.zeros(size)
        best = float("inf")
        for _ in range(trials):
            t0 = time.perf_counter()
            for s in range(1, reps + 1):
                adam_update(master, grad, m, v, s, 1e-3)
            best = min(best, (time.perf_counter() - t0) / reps)
        out[label] = best
    return out


DEFAULT_DIMS = (256, 1024, 1024, 1024, 10)


def run_stage(stage: int, n_gpus: int = 32, dims=DEFAULT_DIMS, steps: int = 100, seed: int = 0,
              per_gpu_batch: int = 64, lr: float = 1e-3, verbose: bool = True) -> dict:
    spec = ModelSpec.build(dims, n_gpus)
    sim = ZeroSimulator(stage, spec, seed=seed, lr=lr, per_gpu_batch=per_gpu_batch)
    losses = []
    wall0 = time.perf_counter()
    for s in range(1, steps + 1):
        losses.append(sim.step(s))
    wall = time.perf_counter() - wall0
    gpu0 = sim.gpus[0]
    busy = {ph: sum(g.busy[ph] for g in sim.gpus) / sim.n / steps for ph in ("forward", "backward", "optimizer")}
    res = dict(
        stage=stage, n_gpus=n_gpus, psi=spec.total_params, largest_layer=spec.largest_layer_params,
        steps=steps, losses=losses, final_params=sim.gathered_params(),
        peak_bytes=max(g.peak for g in sim.gpus),
        peak_breakdown=gpu0.peak_breakdown,
        peak_model_state_bytes=max(g.peak_model_states for g in sim.gpus),
        paper_model_state_bytes=paper_model_state_bytes(spec.total_params, n_gpus, stage),
        timeline=sim.timeline,
        comm_bytes_per_gpu_per_step=sim.fabric.bytes_per_gpu / steps,
        paper_comm_bytes=paper_comm_bytes(spec.total_params, n_gpus, stage),
        collective_calls_per_step={k: v / steps for k, v in sim.fabric.calls.items()},
        optimizer_elements_per_gpu=sim.optimizer_elements_per_gpu,
        flops_per_gpu_per_step=6 * spec.total_params * per_gpu_batch,   # 2 fwd + 4 bwd per param per sample
        busy_per_gpu_per_step=busy,
        wall_per_step={k: v / steps for k, v in sim.wall.items()},
        wall_total_per_step=wall / steps,
    )
    sim.close()
    if verbose:
        MiB = 2 ** 20
        print(f"ZeRO-{stage} on {n_gpus:2d} GPUs | loss {losses[0]:.3f} -> {losses[-1]:.3f}"
              f" | peak/GPU {res['peak_bytes']/MiB:6.2f} MiB"
              f" | model states {res['peak_model_state_bytes']/MiB:6.2f} (paper {res['paper_model_state_bytes']/MiB:6.2f})"
              f" | comm/GPU/step {res['comm_bytes_per_gpu_per_step']/MiB:6.2f} (paper {res['paper_comm_bytes']/MiB:6.2f})"
              f" | {res['wall_total_per_step']*1e3:6.0f} ms/step")
    return res


if __name__ == "__main__":
    results = [run_stage(s) for s in range(4)]
    ref = results[0]["final_params"].float()
    for r in results[1:]:
        print(f"ZeRO-{r['stage']} vs DDP: max |Δw| = {(r['final_params'].float() - ref).abs().max().item()}")
