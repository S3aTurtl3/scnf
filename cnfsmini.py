import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Optional

import numpy as np

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_TANH_NLL, BASELINE_TANH_NFE = 2.1168, 51.0
BASELINE_SIREN_NLL, BASELINE_SIREN_NFE = 2.0127, 127.1
NLL_TARGET, NFE_TARGET = BASELINE_TANH_NLL, BASELINE_TANH_NFE


@dataclass(frozen=True)
class Config:
    width_size: int = 128
    depth: int = 3
    w0_first: float = 30.0
    w0_hidden: float = 1.0

    kinetic_alpha: float = 0.1
    jacobian_alpha: float = 0.0

    mc_penalty: bool = True
    mc_samples: int = 8
    mc_stratified: bool = True

    learning_rate: float = 1e-3
    lr_schedule: str = "constant"
    grad_clip: float = 0.0
    batch_size: int = 50  
    num_iters: int = 2000

    t0: float = 0.0
    t1: float = 0.5
    dt0: float = 0.05
    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 1000
    exact_logp: bool = True

    seed: int = 0
    data_size: int = 2

    def replace(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)

    def summary(self):
        mc = f"mc{self.mc_samples}{'s' if self.mc_stratified else ''}" if self.mc_penalty else "quad"
        return (f"a_ke={self.kinetic_alpha:<8.4g} a_jac={self.jacobian_alpha:<8.4g} "
                f"w0={self.w0_first:<5.4g} w={self.width_size:<4d} d={self.depth} "
                f"lr={self.learning_rate:.2e} {self.lr_schedule} {mc}")


def make_checkerboard(num_samples, rng):
    x1 = rng.uniform(-2.0, 2.0, size=num_samples)
    x2 = rng.uniform(0.0, 1.0, size=num_samples) - rng.integers(0, 2, size=num_samples) * 2.0
    x2 = x2 + np.floor(x1) % 2
    return (np.stack([x1, x2], axis=1) * 2.0).astype(np.float32)


class Datasets:
    def __init__(self, n_train=100, n_val=100, n_test=100, seed=42):
        rng = np.random.default_rng(seed)
        train = make_checkerboard(n_train, rng)
        self.mean, self.std = train.mean(axis=0), train.std(axis=0) + 1e-6
        self.train = (train - self.mean) / self.std
        val = make_checkerboard(n_val, np.random.default_rng(seed + 1))
        test = make_checkerboard(n_test, np.random.default_rng(123))
        self.val = jnp.asarray((val - self.mean) / self.std)
        self.test = jnp.asarray((test - self.mean) / self.std)

    def loader(self, batch_size, seed=0):
        rng = np.random.default_rng(seed)
        n, data = self.train.shape[0], self.train

        def dataloader(step):
            return jnp.asarray(data[rng.choice(n, size=batch_size, replace=False)])

        return dataloader


class SirenLayer(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    w0: float
    is_last: bool

    def __init__(self, in_size, out_size, *, w0, is_first=False, is_last=False, key):
        wkey, bkey = jr.split(key)
        bound = 1.0 / in_size if is_first else math.sqrt(6.0 / in_size) / w0
        self.weight = jr.uniform(wkey, (out_size, in_size), minval=-bound, maxval=bound)
        b_bound = 1.0 / math.sqrt(in_size)
        self.bias = jr.uniform(bkey, (out_size,), minval=-b_bound, maxval=b_bound)
        self.w0 = w0
        self.is_last = is_last

    def __call__(self, x):
        h = self.weight @ x + self.bias
        return h if self.is_last else jnp.sin(self.w0 * h)


class SirenField(eqx.Module):
    layers: list

    def __init__(self, cfg: Config, *, key):
        keys = jr.split(key, cfg.depth + 1)
        in_size, out_size, w = cfg.data_size + 1, cfg.data_size, cfg.width_size
        layers = [SirenLayer(in_size, w, w0=cfg.w0_first, is_first=True, key=keys[0])]
        for i in range(cfg.depth - 1):
            layers.append(SirenLayer(w, w, w0=cfg.w0_hidden, key=keys[i + 1]))
        layers.append(SirenLayer(w, out_size, w0=cfg.w0_hidden, is_last=True, key=keys[-1]))
        self.layers = layers

    def __call__(self, t, y, args):
        x = jnp.concatenate([y, jnp.asarray(t)[None]])
        for layer in self.layers:
            x = layer(x)
        return x


def normal_log_likelihood(y):
    return -0.5 * (y.size * math.log(2 * math.pi) + jnp.sum(y**2))


def exact_logp_dynamics(t, state, args):
    y, _ = state
    (func,) = args
    f, vjp_fn = jax.vjp(lambda z: func(t, z, None), y)
    (jac,) = jax.vmap(vjp_fn)(jnp.eye(y.shape[0]))
    return f, jnp.trace(jac)


def approx_logp_dynamics(t, state, args):
    y, _ = state
    eps, func = args
    f, vjp_fn = jax.vjp(lambda z: func(t, z, None), y)
    (eps_jac,) = vjp_fn(eps)
    return f, jnp.sum(eps_jac * eps)


def exact_quad_dynamics(t, state, args):
    y, _, _, _ = state
    (func,) = args
    f, vjp_fn = jax.vjp(lambda z: func(t, z, None), y)
    (jac,) = jax.vmap(vjp_fn)(jnp.eye(y.shape[0]))
    return f, jnp.trace(jac), 0.5 * jnp.sum(f**2), jnp.sum(jac**2)


def approx_quad_dynamics(t, state, args):
    y, _, _, _ = state
    eps, func = args
    f, vjp_fn = jax.vjp(lambda z: func(t, z, None), y)
    (eps_jac,) = vjp_fn(eps)
    return f, jnp.sum(eps_jac * eps), 0.5 * jnp.sum(f**2), jnp.sum(eps_jac**2)


class CNF(eqx.Module):
    func: Callable
    data_size: int
    exact_logp: bool
    mc_penalty: bool
    mc_samples: int
    mc_stratified: bool
    need_jac: bool
    t0: float
    t1: float
    dt0: float
    rtol: float
    atol: float
    max_steps: int

    def __init__(self, vector_field, cfg: Config):
        self.func = vector_field
        self.data_size = cfg.data_size
        self.exact_logp = cfg.exact_logp
        self.mc_penalty = cfg.mc_penalty
        self.mc_samples = cfg.mc_samples
        self.mc_stratified = cfg.mc_stratified
        self.need_jac = cfg.jacobian_alpha > 0.0
        self.t0, self.t1, self.dt0 = cfg.t0, cfg.t1, cfg.dt0
        self.rtol, self.atol, self.max_steps = cfg.rtol, cfg.atol, cfg.max_steps

    def _controller(self):
        return diffrax.PIDController(rtol=self.rtol, atol=self.atol)

    def _nfe(self, sol):
        num_steps = sol.stats["num_steps"]
        nfe = sol.stats.get("num_func_evaluations", None)
        if nfe is None:
            nfe = 6 * num_steps + 1
        return nfe, num_steps

    def _mc_times(self, key):
        m = self.mc_samples
        u = jr.uniform(key, (m,))
        if self.mc_stratified:
            u = (jnp.arange(m) + u) / m
        u = jnp.clip(u, 1e-5, 1.0 - 1e-5)
        return jnp.sort(self.t0 + (self.t1 - self.t0) * u)

    def _penalty_density(self, t, z, key):
        f, vjp_fn = jax.vjp(lambda w: self.func(t, w, None), z)
        ke = 0.5 * jnp.sum(f**2)
        if not self.need_jac:
            return ke, jnp.zeros_like(ke)
        if self.exact_logp:
            (jac,) = jax.vmap(vjp_fn)(jnp.eye(self.data_size))
            return ke, jnp.sum(jac**2)
        eps = jr.normal(key, (self.data_size,))
        (eps_jac,) = vjp_fn(eps)
        return ke, jnp.sum(eps_jac**2)

    def _logp_term(self, key):
        if self.exact_logp:
            return diffrax.ODETerm(exact_logp_dynamics), (self.func,)
        eps = jr.normal(key, (self.data_size,))
        return diffrax.ODETerm(approx_logp_dynamics), (eps, self.func)

    def _quad_term(self, key):
        if self.exact_logp:
            return diffrax.ODETerm(exact_quad_dynamics), (self.func,)
        eps = jr.normal(key, (self.data_size,))
        return diffrax.ODETerm(approx_quad_dynamics), (eps, self.func)

    def _solve_mc(self, y, *, key):
        tkey, hkey, pkey = jr.split(key, 3)
        ts = self._mc_times(tkey)
        ts_solve = jnp.concatenate([ts[::-1], jnp.asarray([self.t0])])
        term, args = self._logp_term(hkey)
        sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), self.t1, self.t0, -self.dt0, (y, 0.0), args=args,
            saveat=diffrax.SaveAt(ts=ts_solve),
            stepsize_controller=self._controller(),
            max_steps=self.max_steps, throw=False,
        )
        ys, logps = sol.ys
        y_final, delta_logp = ys[-1], logps[-1]
        zs, tzs = ys[:-1], ts_solve[:-1]
        kes, jfs = jax.vmap(self._penalty_density)(tzs, zs, jr.split(pkey, self.mc_samples))
        span = self.t1 - self.t0
        nfe, num_steps = self._nfe(sol)
        return y_final, delta_logp, span * jnp.mean(kes), span * jnp.mean(jfs), nfe, num_steps

    def _solve_quad(self, y, *, key, forward):
        term, args = self._quad_term(key)
        if forward:
            t0, t1, dt0 = self.t0, self.t1, self.dt0
        else:
            t0, t1, dt0 = self.t1, self.t0, -self.dt0
        sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), t0, t1, dt0, (y, 0.0, 0.0, 0.0), args=args,
            stepsize_controller=self._controller(),
            max_steps=self.max_steps, throw=False,
        )
        (y_final,), (delta_logp,), (ke,), (jf,) = sol.ys
        if not forward:
            ke, jf = -ke, -jf
        nfe, num_steps = self._nfe(sol)
        return y_final, delta_logp, ke, jf, nfe, num_steps

    def backward(self, y, *, key):
        if self.mc_penalty:
            return self._solve_mc(y, key=key)
        return self._solve_quad(y, key=key, forward=False)

    def sample(self, *, key):
        noise_key, solve_key, hkey = jr.split(key, 3)
        z0 = jr.normal(noise_key, (self.data_size,))
        term, args = self._logp_term(hkey)
        sol = diffrax.diffeqsolve(
            term, diffrax.Tsit5(), self.t0, self.t1, self.dt0, (z0, 0.0), args=args,
            stepsize_controller=self._controller(),
            max_steps=self.max_steps, throw=False,
        )
        (y_final,), _ = sol.ys
        nfe, _ = self._nfe(sol)
        return y_final, nfe

    def log_prob(self, y, *, key):
        latent, dlogp, _, _, nfe, _ = self.backward(y, key=key)
        return normal_log_likelihood(latent) + dlogp, nfe


class Metrics(eqx.Module):
    nll: jnp.ndarray
    kinetic: jnp.ndarray
    jac_frob: jnp.ndarray
    nfe: jnp.ndarray
    maxed: jnp.ndarray


def batch_metrics(model: CNF, data, key) -> Metrics:
    keys = jr.split(key, data.shape[0])
    latents, dlogps, kes, jfs, nfes, steps = jax.vmap(
        lambda x, k: model.backward(x, key=k)
    )(data, keys)
    ll = dlogps + jax.vmap(normal_log_likelihood)(latents)
    return Metrics(
        nll=-jnp.mean(ll),
        kinetic=jnp.mean(kes),
        jac_frob=jnp.mean(jfs),
        nfe=jnp.mean(nfes.astype(jnp.float32)),
        maxed=jnp.mean((steps >= model.max_steps).astype(jnp.float32)),
    )


def regularized_loss(model: CNF, data, key, cfg: Config):
    m = batch_metrics(model, data, key)
    return m.nll + cfg.kinetic_alpha * m.kinetic + cfg.jacobian_alpha * m.jac_frob, m


@dataclass
class TrialState:
    model: CNF
    opt_state: Any
    steps_done: int = 0
    diverged: bool = False
    train_time: float = 0.0
    nll_history: list = field(default_factory=list)
    ke_history: list = field(default_factory=list)
    jf_history: list = field(default_factory=list)
    nfe_history: list = field(default_factory=list)


def build_optimizer(cfg: Config):
    if cfg.lr_schedule == "cosine":
        lr = optax.cosine_decay_schedule(cfg.learning_rate, decay_steps=cfg.num_iters)
    else:
        lr = cfg.learning_rate
    if cfg.grad_clip and cfg.grad_clip > 0:
        return optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(lr))
    return optax.adam(lr)


def init_state(cfg: Config, key) -> TrialState:
    model = CNF(SirenField(cfg, key=key), cfg)
    optim = build_optimizer(cfg)
    return TrialState(model=model, opt_state=optim.init(eqx.filter(model, eqx.is_array)))


def train(cfg: Config, loader, key, *, until_step: int,
          state: Optional[TrialState] = None, print_every: int = 0, label: str = ""):
    if state is None:
        state = init_state(cfg, key)
    if state.steps_done >= until_step:
        return state

    optim = build_optimizer(cfg)

    @eqx.filter_jit
    def make_step(model, opt_state, data, step_key):
        (_, m), grads = eqx.filter_value_and_grad(regularized_loss, has_aux=True)(
            model, data, step_key, cfg)
        updates, opt_state = optim.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, m

    model, opt_state = state.model, state.opt_state
    start = time.time()
    for step in range(state.steps_done, until_step):
        model, opt_state, m = make_step(model, opt_state, loader(step),
                                        jr.fold_in(key, step))
        nll = float(m.nll)
        if not np.isfinite(nll):
            state.diverged, state.steps_done = True, step
            state.train_time += time.time() - start
            print(f"  [{label}] diverged at step {step} -- pruning", flush=True)
            return state

        state.nll_history.append(nll)
        state.ke_history.append(float(m.kinetic))
        state.jf_history.append(float(m.jac_frob))
        state.nfe_history.append(float(m.nfe))
        if print_every and step % print_every == 0:
            print(f"  [{label}] step {step:5d} | NLL {nll:7.4f} | KE {float(m.kinetic):7.4f} "
                  f"| JF {float(m.jac_frob):7.4f} | NFE {float(m.nfe):6.1f} "
                  f"| {time.time() - start:7.1f}s", flush=True)

    state.model, state.opt_state = model, opt_state
    state.steps_done = until_step
    state.train_time += time.time() - start
    return state


@eqx.filter_jit
def _chunk_metrics(model, data, key):
    return batch_metrics(model, data, key)


def evaluate(model: CNF, data, key, chunk: int = 50) -> dict:
    # Safely restrict evaluation batch size to current dataset size to prevent errors
    chunk = min(chunk, data.shape[0])
    out = {"nll": [], "kinetic": [], "jac_frob": [], "nfe": [], "maxed_frac": []}
    if chunk == 0:
        return {k: float("nan") for k in out}
        
    for i in range(0, (data.shape[0] // chunk) * chunk, chunk):
        m = _chunk_metrics(model, data[i:i + chunk], jr.fold_in(key, i))
        out["nll"].append(float(m.nll))
        out["kinetic"].append(float(m.kinetic))
        out["jac_frob"].append(float(m.jac_frob))
        out["nfe"].append(float(m.nfe))
        out["maxed_frac"].append(float(m.maxed))
    return {k: float(np.mean(v)) for k, v in out.items()}


@eqx.filter_jit
def generate_samples(model: CNF, key, n: int):
    keys = jr.split(key, n)
    samples, nfes = jax.vmap(lambda k: model.sample(key=k))(keys)
    return samples, jnp.mean(nfes.astype(jnp.float32))


def count_params(model) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


class Choice:
    def __init__(self, *values):
        self.values = list(values)

    def sample(self, rng):
        return self.values[rng.integers(len(self.values))]


class LogUniform:
    def __init__(self, lo, hi):
        self.lo, self.hi = float(lo), float(hi)

    def sample(self, rng):
        return float(math.exp(rng.uniform(math.log(self.lo), math.log(self.hi))))


SEARCH_SPACE = {
    "kinetic_alpha": LogUniform(1e-3, 1.0),
    "jacobian_alpha": LogUniform(1e-4, 1e-1),
    "w0_first": Choice(3.0, 5.0, 10.0, 20.0, 30.0),
    "w0_hidden": Choice(1.0, 2.0),
    "width_size": Choice(64, 128, 256),
    "depth": Choice(2, 3, 4),
    "learning_rate": LogUniform(3e-4, 3e-3),
    "lr_schedule": Choice("constant", "cosine"),
    "grad_clip": Choice(0.0, 1.0),
    "mc_samples": Choice(4, 8, 16),
}


def seed_configs(base: Config):
    return [
        base.replace(kinetic_alpha=0.0, jacobian_alpha=0.0),
        base.replace(kinetic_alpha=0.1, jacobian_alpha=0.0),
        base.replace(kinetic_alpha=0.1, jacobian_alpha=0.01, w0_first=10.0),
        base.replace(kinetic_alpha=1.0, jacobian_alpha=0.01, w0_first=5.0),
    ]


def score(nll, nfe, nfe_weight):
    if not (np.isfinite(nll) and np.isfinite(nfe)):
        return float("inf")
    return float(nll) + nfe_weight * float(nfe)


def is_feasible(nll, nfe):
    return bool(np.isfinite(nll) and np.isfinite(nfe)
                and nll <= NLL_TARGET and nfe <= NFE_TARGET)


def rank_key(rec, nfe_weight):
    return (not is_feasible(rec["val_nll"], rec["val_nfe"]),
            score(rec["val_nll"], rec["val_nfe"], nfe_weight))


def time_per_iter(cfg, loader, key, n=3):
    state = init_state(cfg, key)
    train(cfg, loader, key, until_step=1, state=state)
    t0 = time.time()
    train(cfg, loader, key, until_step=1 + n, state=state)
    return (time.time() - t0) / n


def run_search(args, data):
    stages = [int(s) for s in args.stage_iters.split(",")]
    base = Config(seed=args.seed, batch_size=args.search_batch, num_iters=stages[-1],
                  mc_penalty=not args.quadrature, mc_samples=args.mc_samples)
    loader = data.loader(args.search_batch, seed=7)

    rng = np.random.default_rng(args.seed)
    space = dict(SEARCH_SPACE)
    if args.quadrature:
        space.pop("mc_samples")
        
    # Strictly respect the total number of configurations requested.
    # Prioritizes up to 4 seeded configs, then samples remaining from SEARCH_SPACE.
    configs = seed_configs(base) + [
        base.replace(**{k: v.sample(rng) for k, v in space.items()})
        for _ in range(max(0, args.trials - len(seed_configs(base))))
    ]
    configs = configs[:args.trials]
    
    print(f"Candidates: {len(configs)} total configurations evaluated.", flush=True)

    print("Calibrating step time on this machine ...", flush=True)
    spi = time_per_iter(configs[0], loader, jr.PRNGKey(args.seed))
    total, alive, prev = 0, len(configs), 0
    plan = []
    for target in stages:
        total += alive * (target - prev)
        plan.append((alive, prev, target))
        prev, alive = target, max(1, int(np.ceil(alive * args.keep_frac)))
    print(f"  ~{spi:.2f}s/iter at batch {args.search_batch}", flush=True)
    for a, p, t in plan:
        print(f"    stage {p:>4d} -> {t:>4d} iters : {a:>2d} trials", flush=True)
    print(f"  total {total:,} iters ~= {total * spi / 3600:.1f} h", flush=True)
    if not args.no_final:
        print(f"  + final run ~= "
              f"{args.final_iters * spi * args.final_batch / args.search_batch / 3600:.1f} h",
              flush=True)
    print("=" * 78, flush=True)

    jsonl = os.path.join(args.out, "results.jsonl")
    records, t_start = [], time.time()
    trials = [{"id": i, "cfg": c, "state": None} for i, c in enumerate(configs)]

    for si, target in enumerate(stages):
        print(f"\n--- Stage {si + 1}/{len(stages)}: train to {target} iters "
              f"({len(trials)} alive) ---", flush=True)
        stage_recs = []
        for tr in trials:
            if (time.time() - t_start) / 3600 > args.time_budget:
                print(f"Time budget ({args.time_budget}h) hit -- stopping search.", flush=True)
                break
            cfg, label = tr["cfg"], f"t{tr['id']:02d}"
            print(f"[{label}] {cfg.summary()}", flush=True)
            tr["state"] = train(cfg, loader, jr.fold_in(jr.PRNGKey(args.seed), tr["id"]),
                                until_step=target, state=tr["state"], label=label)
            st = tr["state"]
            ev = ({"nll": float("inf"), "nfe": float("inf"), "kinetic": float("nan"),
                   "jac_frob": float("nan"), "maxed_frac": float("nan")}
                  if st.diverged else evaluate(st.model, data.val, jr.PRNGKey(1234)))
            rec = {"trial": tr["id"], "stage": si, "steps": st.steps_done,
                   "config": cfg.to_dict(), "val_nll": ev["nll"], "val_nfe": ev["nfe"],
                   "val_kinetic": ev["kinetic"], "val_jac_frob": ev["jac_frob"],
                   "maxed_frac": ev["maxed_frac"], "diverged": st.diverged,
                   "train_time_s": st.train_time}
            records.append(rec)
            stage_recs.append(rec)
            with open(jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            flag = " *BEATS BOTH BASELINES*" if is_feasible(ev["nll"], ev["nfe"]) else ""
            warn = " [hit max_steps]" if ev["maxed_frac"] > 0 else ""
            print(f"  [{label}] val NLL {ev['nll']:.4f} | val NFE {ev['nfe']:.1f} "
                  f"| score {score(ev['nll'], ev['nfe'], args.nfe_weight):.4f}{flag}{warn}",
                  flush=True)

        if not stage_recs:
            break
        stage_recs.sort(key=lambda r: rank_key(r, args.nfe_weight))
        n_keep = (len(stage_recs) if si == len(stages) - 1
                  else max(1, int(np.ceil(len(stage_recs) * args.keep_frac))))
        keep = {r["trial"] for r in stage_recs[:n_keep]}
        dropped = [t["id"] for t in trials if t["id"] not in keep]
        trials = [t for t in trials if t["id"] in keep]
        if dropped and si < len(stages) - 1:
            print(f"  pruned {len(dropped)}: {dropped}", flush=True)

    return records, (time.time() - t_start) / 3600


def plot_run(model, cfg, test_data, state, path, n_generate=500):
    nll = np.asarray(state.nll_history)
    ke = np.asarray(state.ke_history)
    jf = np.asarray(state.jf_history)
    nfe = np.asarray(state.nfe_history)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    ax.plot(nll, lw=0.8, label="NLL")
    ax.plot(nll + cfg.kinetic_alpha * ke + cfg.jacobian_alpha * jf, lw=0.8, alpha=0.6,
            label="NLL + penalties")
    ax.axhline(BASELINE_TANH_NLL, color="gray", ls="--", lw=1, label="tanh baseline")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (nats)")
    ax.set_title("Training loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(nfe, alpha=0.25, lw=0.6, label="NFE/step")
    if len(nfe) > 50:
        sm = np.convolve(nfe, np.ones(50) / 50, mode="valid")
        ax.plot(np.arange(49, len(nfe)), sm, color="C0", label="NFE (avg 50)")
    ax.axhline(BASELINE_TANH_NFE, color="gray", ls="--", lw=1, label="tanh ~51")
    ax.axhline(BASELINE_SIREN_NFE, color="red", ls=":", lw=1, label="plain SIREN ~127")
    ax2 = ax.twinx()
    ax2.plot(ke, color="green", alpha=0.5, lw=0.6)
    ax2.set_ylabel("Transport cost (MC estimate)", color="green")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("NFE")
    ax.set_title("Solver cost & transport cost")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.scatter(nfe, nll, s=2, alpha=0.3)
    ax.axhline(BASELINE_TANH_NLL, color="gray", ls="--", lw=1)
    ax.axvline(BASELINE_TANH_NFE, color="gray", ls="--", lw=1)
    ax.set_xlabel("NFE")
    ax.set_ylabel("NLL")
    ax.set_title("NLL vs NFE")
    ax.grid(True, alpha=0.3)

    test_np = np.asarray(test_data)
    axes[1, 0].scatter(test_np[:, 0], test_np[:, 1], s=5, alpha=0.5)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_title("True checkerboard (test)")

    samples, gen_nfe = generate_samples(model, jr.PRNGKey(999), n_generate)
    s_np = np.asarray(samples)
    print(f"Avg NFE of generation (forward solve): {float(gen_nfe):.1f}", flush=True)
    axes[1, 1].scatter(s_np[:, 0], s_np[:, 1], s=5, alpha=0.5, c="red")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title(f"Generated (n={n_generate}, NFE {float(gen_nfe):.0f})")

    axes[1, 2].scatter(test_np[:500, 0], test_np[:500, 1], s=10, alpha=0.4, c="blue",
                       label="true")
    axes[1, 2].scatter(s_np[:, 0], s_np[:, 1], s=10, alpha=0.4, c="red", label="generated")
    axes[1, 2].set_aspect("equal")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_title("Overlay")

    fig.suptitle(cfg.summary(), fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}", flush=True)


def pareto_front(points):
    keep = []
    for i, (xi, yi) in enumerate(points):
        if any((xj <= xi and yj <= yi) and (xj < xi or yj < yi)
               for j, (xj, yj) in enumerate(points) if j != i):
            continue
        keep.append(i)
    return sorted(keep, key=lambda i: points[i][0])


def plot_search(records, nfe_weight, path):
    latest = {}
    for r in records:
        if r["trial"] not in latest or r["steps"] > latest[r["trial"]]["steps"]:
            latest[r["trial"]] = r
    final = list(latest.values())
    ok = [r for r in final if np.isfinite(r["val_nll"]) and np.isfinite(r["val_nfe"])]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    if ok:
        x = np.array([r["val_nfe"] for r in ok])
        y = np.array([r["val_nll"] for r in ok])
        a = np.array([max(r["config"]["kinetic_alpha"], 1e-4) for r in ok])
        sc = ax.scatter(x, y, c=np.log10(a), cmap="viridis", s=60, edgecolor="k", lw=0.4)
        plt.colorbar(sc, ax=ax, label="log10 kinetic_alpha")
        idx = pareto_front(list(zip(x, y)))
        ax.plot(x[idx], y[idx], "k--", lw=1, alpha=0.6, label="Pareto front")
        best = min(ok, key=lambda r: score(r["val_nll"], r["val_nfe"], nfe_weight))
        ax.scatter([best["val_nfe"]], [best["val_nll"]], s=220, facecolors="none",
                   edgecolors="red", lw=2, label="best by score")
    ax.scatter([BASELINE_TANH_NFE], [BASELINE_TANH_NLL], marker="*", s=260, c="gray",
               edgecolor="k", label="tanh baseline", zorder=5)
    ax.scatter([BASELINE_SIREN_NFE], [BASELINE_SIREN_NLL], marker="*", s=260, c="red",
               edgecolor="k", label="plain SIREN", zorder=5)
    ax.axhline(NLL_TARGET, color="gray", ls=":", lw=1)
    ax.axvline(NFE_TARGET, color="gray", ls=":", lw=1)
    ax.set_xlabel("Validation NFE")
    ax.set_ylabel("Validation NLL (nats)")
    ax.set_title("Search: NLL / NFE trade-off")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if ok:
        a = np.array([max(r["config"]["kinetic_alpha"], 1e-4) for r in ok])
        ax.scatter(a, [score(r["val_nll"], r["val_nfe"], nfe_weight) for r in ok], s=45)
        ax.set_xscale("log")
        ax2 = ax.twinx()
        ax2.scatter(a, [r["val_nfe"] for r in ok], s=25, c="green", marker="^", alpha=0.6)
        ax2.set_ylabel("val NFE", color="green")
    ax.set_xlabel("kinetic_alpha")
    ax.set_ylabel(f"score = NLL + {nfe_weight}*NFE")
    ax.set_title("Does the transport-cost penalty pay off?")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if ok:
        groups = {}
        for r in ok:
            groups.setdefault(r["config"]["w0_first"], []).append(r["val_nfe"])
        ws = sorted(groups)
        ax.boxplot([groups[w] for w in ws])
        ax.set_xticks(range(1, len(ws) + 1))
        ax.set_xticklabels([f"{w:g}" for w in ws])
        ax.axhline(BASELINE_TANH_NFE, color="gray", ls="--", lw=1, label="tanh NFE")
        ax.legend(fontsize=8)
    ax.set_xlabel("w0_first")
    ax.set_ylabel("Validation NFE")
    ax.set_title("Solver cost vs SIREN frequency scale")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    ranked = sorted(final, key=lambda r: score(r["val_nll"], r["val_nfe"], nfe_weight))[:8]
    rows = [["#", "a_ke", "a_jac", "w0", "w/d", "mc", "NLL", "NFE", "score"]]
    for i, r in enumerate(ranked, 1):
        c, s = r["config"], score(r["val_nll"], r["val_nfe"], nfe_weight)
        rows.append([str(i), f"{c['kinetic_alpha']:.3g}", f"{c['jacobian_alpha']:.3g}",
                     f"{c['w0_first']:.0f}", f"{c['width_size']}/{c['depth']}",
                     str(c["mc_samples"]) if c["mc_penalty"] else "quad",
                     f"{r['val_nll']:.3f}" if np.isfinite(r["val_nll"]) else "nan",
                     f"{r['val_nfe']:.0f}" if np.isfinite(r["val_nfe"]) else "nan",
                     f"{s:.3f}" if np.isfinite(s) else "inf"])
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    for i, r in enumerate(ranked, 1):
        if is_feasible(r["val_nll"], r["val_nfe"]):
            for j in range(len(rows[0])):
                tbl[(i, j)].set_facecolor("#d8f0d8")
    ax.set_title("Leaderboard (validation; green = beats both baselines)")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tune", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default="results_ot")

    p.add_argument("--config", default=None)
    p.add_argument("--kinetic-alpha", type=float, default=None)
    p.add_argument("--jacobian-alpha", type=float, default=None)
    p.add_argument("--w0-first", type=float, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=50)  
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--mc-samples", type=int, default=8)
    p.add_argument("--no-stratified", action="store_true")
    p.add_argument("--quadrature", action="store_true")

    # This now sets a strict limit on total trials evaluated (seeded baseline + random space)
    p.add_argument("--trials", type=int, default=5, help="Total hyperparameter trials to evaluate (including baselines)")  
    p.add_argument("--stage-iters", default="120,350,900")
    p.add_argument("--keep-frac", type=float, default=0.34)
    p.add_argument("--search-batch", type=int, default=None, help="Batch size for HP tuning (falls back to --batch-size)")  
    p.add_argument("--nfe-weight", type=float, default=0.002)
    p.add_argument("--time-budget", type=float, default=float("inf"))
    p.add_argument("--final-iters", type=int, default=2000)
    p.add_argument("--final-batch", type=int, default=None, help="Batch size for final training (falls back to --batch-size)")  
    p.add_argument("--no-final", action="store_true")
    
    # New argument controlling dataset sample generation size
    p.add_argument("--dataset-size", type=int, default=1000, help="Number of samples to generate for train/val/test splits")

    args = p.parse_args()
    
    # Cascade default batch size to specialized search/final configurations if they were left unspecified
    if args.search_batch is None:
        args.search_batch = args.batch_size
    if args.final_batch is None:
        args.final_batch = args.batch_size

    args.trials = max(1, args.trials)

    return args


def single_run_config(args) -> Config:
    cfg = Config(seed=args.seed)
    if args.config:
        with open(args.config) as f:
            cfg = Config(**json.load(f))
    over = {"kinetic_alpha": args.kinetic_alpha, "jacobian_alpha": args.jacobian_alpha,
            "w0_first": args.w0_first, "width_size": args.width, "depth": args.depth,
            "learning_rate": args.lr, "batch_size": args.batch_size,
            "num_iters": args.iters}
    cfg = cfg.replace(**{k: v for k, v in over.items() if v is not None})
    return cfg.replace(mc_penalty=not args.quadrature, mc_samples=args.mc_samples,
                       mc_stratified=not args.no_stratified)


def report(tag, ev):
    print("=" * 78, flush=True)
    print(f"{tag} NLL {ev['nll']:.4f} | {tag} NFE {ev['nfe']:.1f} | "
          f"transport cost {ev['kinetic']:.4f} | ||J||_F^2 {ev['jac_frob']:.4f}", flush=True)
    print(f"tanh baseline : NLL {BASELINE_TANH_NLL} | NFE {BASELINE_TANH_NFE}", flush=True)
    print(f"plain SIREN   : NLL {BASELINE_SIREN_NLL} | NFE {BASELINE_SIREN_NFE}", flush=True)
    print("RESULT: " + ("beats BOTH baselines."
                        if is_feasible(ev["nll"], ev["nfe"])
                        else "does not beat both baselines."), flush=True)
    print("=" * 78, flush=True)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("=" * 78, flush=True)
    print("CNF + SIREN + OT transport-cost penalty (Monte Carlo) -- 2D checkerboard",
          flush=True)
    print(f"JAX {jax.__version__} | devices: {jax.devices()}", flush=True)
    print("=" * 78, flush=True)
    
    if args.smoke:
        print(">>> SMOKE TEST: tiny budgets, results are meaningless <<<", flush=True)
        args.stage_iters, args.trials, args.search_batch = "3,6", 1, 50  
        args.final_iters, args.final_batch = 5, 50
        args.dataset_size = min(args.dataset_size, 100)

    # Dataset instantiated using the configured command line size
    data = Datasets(n_train=args.dataset_size, n_val=args.dataset_size, n_test=args.dataset_size)

    if not args.tune:
        cfg = single_run_config(args)
        if args.smoke:
            cfg = cfg.replace(num_iters=5, batch_size=args.batch_size)  
        print(f"Config: {cfg.summary()}", flush=True)
        print(f"        iters={cfg.num_iters}, batch={cfg.batch_size}, "
              f"rtol=atol={cfg.rtol}, exact_logp={cfg.exact_logp}, "
              f"mc_penalty={cfg.mc_penalty}, mc_samples={cfg.mc_samples}, "
              f"stratified={cfg.mc_stratified}", flush=True)
        state = train(cfg, data.loader(cfg.batch_size, seed=42), jr.PRNGKey(cfg.seed),
                      until_step=cfg.num_iters, print_every=max(1, cfg.num_iters // 10),
                      label="run")
        print(f"Vector field parameters: {count_params(state.model.func):,}", flush=True)
        if state.diverged:
            print("Diverged -- no evaluation.", flush=True)
            return
        print(f"Trained in {state.train_time / 3600:.2f} h", flush=True)
        report("TEST", evaluate(state.model, data.test, jr.PRNGKey(4321)))
        eqx.tree_serialise_leaves(os.path.join(args.out, "model.eqx"), state.model)
        np.savez(os.path.join(args.out, "histories.npz"),
                 nll=np.array(state.nll_history), ke=np.array(state.ke_history),
                 jf=np.array(state.jf_history), nfe=np.array(state.nfe_history))
        plot_run(state.model, cfg, data.test, state, os.path.join(args.out, "run.png"))
        print("All done.", flush=True)
        return

    records, hours = run_search(args, data)
    if not records:
        print("No trials completed.", flush=True)
        return
    print(f"\nSearch finished in {hours:.2f} h ({len(records)} evaluations).", flush=True)

    latest = {}
    for r in records:
        if r["trial"] not in latest or r["steps"] > latest[r["trial"]]["steps"]:
            latest[r["trial"]] = r
    final = sorted(latest.values(), key=lambda r: rank_key(r, args.nfe_weight))

    print("\n" + "=" * 78, flush=True)
    print(f"LEADERBOARD (validation; score = NLL + {args.nfe_weight}*NFE; "
          f"targets NLL<={NLL_TARGET}, NFE<={NFE_TARGET})", flush=True)
    print("=" * 78, flush=True)
    print(f"{'#':>2} {'iters':>5} {'NLL':>8} {'NFE':>7} {'score':>8}  config", flush=True)
    for i, r in enumerate(final, 1):
        mark = " <== beats both baselines" if is_feasible(r["val_nll"], r["val_nfe"]) else ""
        print(f"{i:>2} {r['steps']:>5} {r['val_nll']:>8.4f} {r['val_nfe']:>7.1f} "
              f"{score(r['val_nll'], r['val_nfe'], args.nfe_weight):>8.4f}  "
              f"{Config(**r['config']).summary()}{mark}", flush=True)
    print("=" * 78, flush=True)

    with open(os.path.join(args.out, "results.csv"), "w") as f:
        keys = ["trial", "stage", "steps", "val_nll", "val_nfe", "val_kinetic",
                "val_jac_frob", "maxed_frac", "diverged", "train_time_s"]
        ckeys = sorted(records[0]["config"])
        f.write(",".join(keys + ckeys + ["score"]) + "\n")
        for r in sorted(records, key=lambda r: (r["trial"], r["stage"])):
            f.write(",".join([str(r[k]) for k in keys]
                             + [str(r["config"][k]) for k in ckeys]
                             + [str(score(r["val_nll"], r["val_nfe"], args.nfe_weight))])
                    + "\n")
    print(f"Saved: {os.path.join(args.out, 'results.csv')}", flush=True)
    plot_search(records, args.nfe_weight, os.path.join(args.out, "search.png"))

    best = Config(**final[0]["config"])
    with open(os.path.join(args.out, "best_config.json"), "w") as f:
        json.dump(best.to_dict(), f, indent=2)
    print(f"Saved: {os.path.join(args.out, 'best_config.json')}", flush=True)

    if args.no_final:
        print(f"\nRetrain later with:\n"
              f"    python cnf_siren_ot.py --config {args.out}/best_config.json", flush=True)
        return

    print("\n" + "=" * 78, flush=True)
    print(f"FINAL RUN: {args.final_iters} iters at batch {args.final_batch}", flush=True)
    print(f"  {best.summary()}", flush=True)
    print("=" * 78, flush=True)
    fcfg = best.replace(batch_size=args.final_batch, num_iters=args.final_iters)
    state = train(fcfg, data.loader(args.final_batch, seed=42), jr.PRNGKey(args.seed),
                  until_step=args.final_iters,
                  print_every=max(1, args.final_iters // 10), label="final")
    test = evaluate(state.model, data.test, jr.PRNGKey(4321))
    report("TEST", test)
    eqx.tree_serialise_leaves(os.path.join(args.out, "best_model.eqx"), state.model)
    np.savez(os.path.join(args.out, "final_histories.npz"),
             nll=np.array(state.nll_history), ke=np.array(state.ke_history),
             jf=np.array(state.jf_history), nfe=np.array(state.nfe_history))
    with open(os.path.join(args.out, "final_result.json"), "w") as f:
        json.dump({"config": fcfg.to_dict(), "test": test, "search_hours": hours},
                  f, indent=2)
    plot_run(state.model, fcfg, data.test, state, os.path.join(args.out, "final_run.png"))
    print("All done.", flush=True)


if __name__ == "__main__":
    main()