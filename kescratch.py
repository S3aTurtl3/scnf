

#kinetic energy penalty set as alpha=0.1
import os
import subprocess
import sys

def _force_install_dependencies():
    print("=" * 60, flush=True)
    print("Checking and installing dependencies...", flush=True)
    print("=" * 60, flush=True)

    need_install = False
    try:
        import numpy
        if numpy.__version__ != "1.26.4":
            need_install = True
    except ImportError:
        need_install = True

    if not need_install:
        try:
            import jax, equinox, optax, diffrax
            print("All dependencies already installed correctly.", flush=True)
            return
        except ImportError:
            need_install = True

    if need_install:
        print("Installing pinned dependencies (this may take a few minutes)...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "uninstall", "-y", "numpy", "jax",
"jaxlib"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        packages = [
            "numpy==1.26.4",
            "jax==0.4.35",
            "jaxlib==0.4.35",
            "equinox==0.11.8",
            "optax==0.2.3",
            "diffrax==0.6.0",
            "matplotlib",
        ]
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir"] +
packages,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(" Dependencies installed successfully!", flush=True)
        print(" IMPORTANT: Please restart the runtime/kernel and run again.", flush=True)
        print("   (In Jupyter/Colab: Runtime → Restart runtime)", flush=True)
        print("   (In Python script: This is a one-time setup, run again)",
flush=True)
        print("=" * 60, flush=True)
        sys.exit(0)

_force_install_dependencies()


import math
import time
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import diffrax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 0
DATA_SIZE = 2

# SIREN configuration (same as the plain SIREN run)
WIDTH_SIZE = 128
DEPTH = 3
SIREN_W0_FIRST = 30.0   # stiffness at init -- lower this too if NFE stays
high
SIREN_W0_HIDDEN = 1.0


KINETIC_ALPHA = 0.1     # weight of the kinetic-energy penalty


T0, T1, DT0 = 0.0, 0.5, 0.05
RTOL = ATOL = 1e-5
MAX_STEPS = 1000
EXACT_LOGP = True

# Data / training
N_TRAIN, N_TEST = 10_000, 2_000
BATCH_SIZE = 512
NUM_ITERS = 2_000
LEARNING_RATE = 1e-3
PRINT_EVERY = 200
N_GENERATE = 500

FIGURE_PATH = "cnf_siren_ke_results.png"
LOSS_PATH = "loss_history_siren_ke.npy"  
KE_PATH = "ke_history_siren_ke.npy"
NFE_PATH = "nfe_history_siren_ke.npy"



class SirenLayer(eqx.Module):
    """One SIREN layer: x -> sin(w0 * (W x + b)), SIREN-paper init."""

    weight: jnp.ndarray
    bias: jnp.ndarray
    w0: float
    is_first: bool
    is_last: bool

    def __init__(self, in_size, out_size, *, w0, is_first=False,
is_last=False, key):
        wkey, bkey = jr.split(key)
        if is_first:
            bound = 1.0 / in_size
        else:
            bound = math.sqrt(6.0 / in_size) / w0
        self.weight = jr.uniform(wkey, (out_size, in_size), minval=-bound,
maxval=bound)
        b_bound = 1.0 / math.sqrt(in_size)
        self.bias = jr.uniform(bkey, (out_size,), minval=-b_bound,
maxval=b_bound)
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last

    def __call__(self, x):
        h = self.weight @ x + self.bias
        if self.is_last:
            return h
        return jnp.sin(self.w0 * h)


class Siren(eqx.Module):
    """SIREN MLP: sinusoidal hidden layers, linear output layer."""

    layers: list

    def __init__(self, *, in_size, out_size, width_size, depth, w0_first,
w0_hidden, key):
        keys = jr.split(key, depth + 1)
        layers = [SirenLayer(in_size, width_size, w0=w0_first, is_first=True,
key=keys[0])]
        for i in range(depth - 1):
            layers.append(SirenLayer(width_size, width_size, w0=w0_hidden,
key=keys[i + 1]))
        layers.append(SirenLayer(width_size, out_size, w0=w0_hidden,
is_last=True, key=keys[-1]))
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class IsotropicSirenFunc(eqx.Module):


    net: Siren

    def __init__(self, *, data_size, width_size, depth,
w0_first=SIREN_W0_FIRST,
                w0_hidden=SIREN_W0_HIDDEN, key, **kwargs):
        super().__init__(**kwargs)
        self.net = Siren(
            in_size=data_size + 1,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            w0_first=w0_first,
            w0_hidden=w0_hidden,
            key=key,
        )

    def __call__(self, t, y, args):
        t = jnp.asarray(t)[None]
        return self.net(jnp.concatenate([y, t]))


def normal_log_likelihood(y):
    """log N(y; 0, I)."""
    return -0.5 * (y.size * math.log(2 * math.pi) + jnp.sum(y**2))


def exact_logp_ke_wrapper(t, y, args):
    """Augmented dynamics: exact Jacobian trace + kinetic energy."""
    y, _, _ = y
    (func,) = args
    fn = lambda y: func(t, y, None)
    f, vjp_fn = jax.vjp(fn, y)
    (dfdy,) = jax.vmap(vjp_fn)(jnp.eye(y.shape[0]))
    logp = jnp.trace(dfdy)
    ke = 0.5 * jnp.sum(f**2)
    return f, logp, ke


def approx_logp_ke_wrapper(t, y, args):
    y, _, _ = y
    eps, func = args
    fn = lambda y: func(t, y, None)
    f, vjp_fn = jax.vjp(fn, y)
    (eps_dfdy,) = vjp_fn(eps)
    logp = jnp.sum(eps_dfdy * eps)
    ke = 0.5 * jnp.sum(f**2)
    return f, logp, ke


def count_nfe(sol):

    nfe = sol.stats.get("num_func_evaluations", None)
    if nfe is None:
        nfe = 6 * sol.stats["num_steps"] + 1
    return nfe



class CNF(eqx.Module):

    func: Callable
    data_size: int
    exact_logp: bool
    t0: float
    t1: float
    dt0: float
    rtol: float
    atol: float
    max_steps: int

    def __init__(self, *, vector_field, data_size, exact_logp=True, t0=T0,
t1=T1,
                dt0=DT0, rtol=RTOL, atol=ATOL, max_steps=MAX_STEPS,
**kwargs):
        super().__init__(**kwargs)
        self.func = vector_field
        self.data_size = data_size
        self.exact_logp = exact_logp
        self.t0 = t0
        self.t1 = t1
        self.dt0 = dt0
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    def _solve(self, y, *, key, forward):

        if self.exact_logp:
            term = diffrax.ODETerm(exact_logp_ke_wrapper)
            args = (self.func,)
        else:
            eps = jr.normal(key, (self.data_size,))
            term = diffrax.ODETerm(approx_logp_ke_wrapper)
            args = (eps, self.func)
        y0 = (y, 0.0, 0.0)
        if forward:
            t0, t1, dt0 = self.t0, self.t1, self.dt0
        else:
            t0, t1, dt0 = self.t1, self.t0, -self.dt0
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0,
            t1,
            dt0,
            y0,
            args=args,
            stepsize_controller=diffrax.PIDController(rtol=self.rtol,
atol=self.atol),
            max_steps=self.max_steps,
        )
        (y_final,), (delta_logp,), (ke,) = sol.ys

        if not forward:
            ke = -ke
        return y_final, delta_logp, ke, count_nfe(sol)

    def sample_and_compute_density(self, y, *, key,
is_forward_direction=False):

        return self._solve(y, key=key, forward=is_forward_direction)

    def sample(self, *, key):
        noise_key, solve_key = jr.split(key)
        z0 = jr.normal(noise_key, (self.data_size,))
        y_final, _, _, nfe = self._solve(z0, key=solve_key, forward=True)
        return y_final, nfe

    def log_prob(self, y, *, key):
        latent, delta_logp, _, nfe = self.sample_and_compute_density(y,
key=key)
        return normal_log_likelihood(latent) + delta_logp, nfe



def nll_loss(model, data, key):

    keys = jr.split(key, data.shape[0])

    def compute_one(x, k):
        return model.sample_and_compute_density(x, key=k,
is_forward_direction=False)

    latents, delta_logps, kes, nfes = jax.vmap(compute_one)(data, keys)
    log_likelihood = delta_logps + jax.vmap(normal_log_likelihood)(latents)
    nll = -jnp.mean(log_likelihood)
    ke = jnp.mean(kes)
    avg_nfe = jnp.mean(nfes.astype(jnp.float32))
    return nll, ke, avg_nfe


def make_checkerboard(num_samples, rng):
    x1 = rng.uniform(-2.0, 2.0, size=num_samples)
    x2 = rng.uniform(0.0, 1.0, size=num_samples) - rng.integers(0, 2,
size=num_samples) * 2.0
    x2 = x2 + np.floor(x1) % 2
    return (np.stack([x1, x2], axis=1) * 2.0).astype(np.float32)


def create_checkerboard_dataloader(num_samples=N_TRAIN, batch_size=BATCH_SIZE,
seed=0):

    rng = np.random.default_rng(seed)
    data = make_checkerboard(num_samples, rng)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-6
    data = (data - mean) / std

    def dataloader(step):
        idx = rng.choice(num_samples, size=batch_size, replace=False)
        return jnp.asarray(data[idx])

    return dataloader, mean, std


def train_cnf(vector_field, train_loader, key, data_size=DATA_SIZE,
lr=LEARNING_RATE,
            steps=NUM_ITERS, exact_logp=EXACT_LOGP, print_every=PRINT_EVERY,
            kinetic_alpha=KINETIC_ALPHA):
    model = CNF(vector_field=vector_field, data_size=data_size,
exact_logp=exact_logp)
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    nll_history = []
    ke_history = []
    nfe_history = []

    @eqx.filter_jit
    def make_step(model, opt_state, data, key):
        def loss_fn(model, data, key):
            nll, ke, avg_nfe = nll_loss(model, data, key)
            total = nll + kinetic_alpha * ke  # <-- the regularized objective
            return total, (nll, ke, avg_nfe)

        (_, (nll, ke, avg_nfe)), grads = eqx.filter_value_and_grad(loss_fn,
has_aux=True)(
            model, data, key
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, nll, ke, avg_nfe

    print("=" * 60, flush=True)
    print("Training CNF with SIREN + kinetic-energy penalty ...", flush=True)
    print(f"w0_first={SIREN_W0_FIRST}, w0_hidden={SIREN_W0_HIDDEN}, "
        f"kinetic_alpha={kinetic_alpha}", flush=True)
    print("(first step includes JIT compilation and is slow)", flush=True)
    print("=" * 60, flush=True)
    start = time.time()
    for step in range(steps):
        data = train_loader(step)
        step_key = jr.fold_in(key, step)
        model, opt_state, nll, ke, avg_nfe = make_step(model, opt_state, data,
step_key)
        nll_history.append(float(nll))
        ke_history.append(float(ke))
        nfe_history.append(float(avg_nfe))
        if step % print_every == 0:
            elapsed = time.time() - start
            print(
                f"Step {step:5d} | NLL: {nll:.4f} | KE: {ke:.4f} | "
                f"NFE: {avg_nfe:.1f} | Elapsed: {elapsed:7.1f}s",
                flush=True,
            )

    nll_history = np.array(nll_history)
    ke_history = np.array(ke_history)
    nfe_history = np.array(nfe_history)
    print("=" * 60, flush=True)
    print(f"Training done in {time.time() - start:.1f}s", flush=True)
    print(f"Final NLL:                   {nll_history[-1]:.4f}", flush=True)
    print(f"Avg NLL  (last 100 steps):   {nll_history[-100:].mean():.4f}",
flush=True)
    print(f"Avg KE   (last 100 steps):   {ke_history[-100:].mean():.4f}",
flush=True)
    print(f"Avg NFE  (last 100 steps):   {nfe_history[-100:].mean():.1f}",
flush=True)
    print("=" * 60, flush=True)
    np.save(LOSS_PATH, nll_history)
    np.save(KE_PATH, ke_history)
    np.save(NFE_PATH, nfe_history)
    print(f"Saved: {LOSS_PATH}, {KE_PATH}, {NFE_PATH}", flush=True)
    return model, nll_history, ke_history, nfe_history


@eqx.filter_jit
def evaluate_nll(model, data, key):

    return nll_loss(model, data, key)


@eqx.filter_jit
def generate_samples(model, key, n):
    keys = jr.split(key, n)
    samples, nfes = jax.vmap(lambda k: model.sample(key=k))(keys)
    return samples, jnp.mean(nfes.astype(jnp.float32))



def plot_results(model, test_data, nll_history, ke_history, nfe_history):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(nll_history, label="NLL")
    axes[0, 0].plot(nll_history + KINETIC_ALPHA * ke_history, alpha=0.5,
                    label=f"NLL + {KINETIC_ALPHA}*KE")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss (nats)")
    axes[0, 0].set_title("Training loss (SIREN + KE penalty)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(nfe_history, alpha=0.3, label="NFE per step")
    window = 50
    if len(nfe_history) > window:
        smooth = np.convolve(nfe_history, np.ones(window) / window,
mode="valid")
        axes[0, 1].plot(np.arange(window - 1, len(nfe_history)), smooth,
label=f"NFE avg ({window})")
    ax_ke = axes[0, 1].twinx()
    ax_ke.plot(ke_history, color="green", alpha=0.4, label="KE")
    ax_ke.set_ylabel("Kinetic energy", color="green")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("NFE")
    axes[0, 1].set_title("Solver cost & kinetic energy")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].scatter(nfe_history, nll_history, alpha=0.3, s=2)
    axes[0, 2].set_xlabel("NFE")
    axes[0, 2].set_ylabel("NLL")
    axes[0, 2].set_title("NLL vs NFE (SIREN + KE)")
    axes[0, 2].grid(True, alpha=0.3)

    test_np = np.asarray(test_data)
    axes[1, 0].scatter(test_np[:, 0], test_np[:, 1], s=5, alpha=0.5)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_title("True Checkerboard data (test)")

    print(f"Generating {N_GENERATE} samples from the trained model ...",
flush=True)
    samples, gen_nfe = generate_samples(model, jr.PRNGKey(999), N_GENERATE)
    samples_np = np.asarray(samples)
    print(f"Avg NFE of generation (forward solve): {float(gen_nfe):.1f}",
flush=True)
    axes[1, 1].scatter(samples_np[:, 0], samples_np[:, 1], s=5, alpha=0.5,
c="red")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title(f"Generated samples (SIREN+KE, n={N_GENERATE})")

    axes[1, 2].scatter(test_np[:500, 0], test_np[:500, 1], s=10, alpha=0.4,
c="blue", label="true")
    axes[1, 2].scatter(samples_np[:, 0], samples_np[:, 1], s=10, alpha=0.4,
c="red", label="generated")
    axes[1, 2].set_aspect("equal")
    axes[1, 2].legend()
    axes[1, 2].set_title("Overlay (SIREN + KE)")

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {FIGURE_PATH}", flush=True)



def main():
    print("Starting CNF: SIREN + kinetic-energy penalty on 2D Checkerboard",
flush=True)
    print(f"JAX {jax.__version__} | devices: {jax.devices()}", flush=True)
    print(
        f"Config: SIREN field, width={WIDTH_SIZE}, depth={DEPTH}, "
        f"w0_first={SIREN_W0_FIRST}, w0_hidden={SIREN_W0_HIDDEN}, "
        f"kinetic_alpha={KINETIC_ALPHA}, iters={NUM_ITERS},batch={BATCH_SIZE}, "
        f"lr={LEARNING_RATE}, Tsit5 rtol=atol={RTOL}, max_steps={MAX_STEPS}, "
        f"exact_logp={EXACT_LOGP}",
        flush=True,
    )

    train_loader, train_mean, train_std = create_checkerboard_dataloader(
        num_samples=N_TRAIN, batch_size=BATCH_SIZE, seed=42
    )
    test_rng = np.random.default_rng(123)
    test_data = make_checkerboard(N_TEST, test_rng)
    test_data = (test_data - train_mean) / train_std
    test_data = jnp.asarray(test_data)
    print(f"Data: {N_TRAIN} train / {N_TEST} test samples, standardized",
flush=True)

    key = jr.PRNGKey(SEED)
    model_key, train_key, eval_key = jr.split(key, 3)

    vector_field = IsotropicSirenFunc(
        data_size=DATA_SIZE,
        width_size=WIDTH_SIZE,
        depth=DEPTH,
        w0_first=SIREN_W0_FIRST,
        w0_hidden=SIREN_W0_HIDDEN,
        key=model_key,
    )

    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(vector_field,
eqx.is_array))
    )
    print(f"Vector field parameters: {n_params:,}", flush=True)

    model, nll_history, ke_history, nfe_history = train_cnf(
        vector_field=vector_field,
        train_loader=train_loader,
        key=train_key,
        data_size=DATA_SIZE,
        lr=LEARNING_RATE,
        steps=NUM_ITERS,
        exact_logp=EXACT_LOGP,
        print_every=PRINT_EVERY,
        kinetic_alpha=KINETIC_ALPHA,
    )

    test_nll, test_ke, test_nfe = evaluate_nll(model, test_data, eval_key)
    print(
        f"Test NLL: {float(test_nll):.4f} | Test KE: {float(test_ke):.4f} | "
        f"Test avg NFE: {float(test_nfe):.1f}",
        flush=True,
    )

    plot_results(model, test_data, nll_history, ke_history, nfe_history)
    print("All done (SIREN + kinetic-energy run).", flush=True)


if __name__ == "__main__":
    main()