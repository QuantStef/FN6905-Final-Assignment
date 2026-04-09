"""
Physics-Driven Galerkin Method (PDGM) for barrier option pricing
under Black-Scholes dynamics.

Adapted from:  https://github.com/zhaoyu-zhang/PDGM-Geometric_Asian
Purpose:       Extend PDGM to barrier options and compare with the
               closed-form expressions of Table 8.1 (Reiner-Rubinstein).

Architecture (same as the original geometric-Asian notebook)
------------------------------------------------------------
  LSTM (full-sequence)  : encodes path history S_0, …, S_t → hidden state h_t
  Feedforward NN (3×64) : maps (S_t, t, h_t) → option price f_t
  PDE loss              : Black-Scholes PDE residual at every path point
  Terminal loss         : barrier-adjusted payoff condition at maturity

Key modifications vs. the original
------------------------------------
  • terminal_condition()  replaced by make_barrier_payoff()
    – payoff = max(S_T−K,0) * 1{barrier not crossed}  (knock-out)
    – payoff = max(S_T−K,0) * 1{barrier crossed}       (knock-in)
  • Supports call/put × down-out/down-in/up-out/up-in
  • Re-implemented in PyTorch (original uses TF 1.x)
  • Vectorised LSTM + FFN evaluation for speed

Usage
-----
  python pdgm_barrier.py --barrier_type down-out --option_type call --B 0.9
  python pdgm_barrier.py --barrier_type up-out   --option_type put  --B 1.1
  python pdgm_barrier.py --barrier_type down-in  --option_type call --B 0.9
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from scipy.stats import norm


# ══════════════════════════════════════════════════════════════════════════════
# Closed-form helpers  (Table 8.1 — Reiner-Rubinstein)
# ══════════════════════════════════════════════════════════════════════════════

def _d_plus(z, r, sigma, tau):
    return (np.log(z) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

def _d_minus(z, r, sigma, tau):
    return (np.log(z) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

def bs_call(S, K, r, sigma, T):
    d1 = _d_plus(S / K, r, sigma, T)
    d2 = _d_minus(S / K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, r, sigma, T):
    d1 = _d_plus(S / K, r, sigma, T)
    d2 = _d_minus(S / K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def cf_barrier(S0, K, B, r, sigma, T, option_type='call', barrier_type='down-out'):
    """Closed-form barrier option price via Reiner-Rubinstein (Table 8.1)."""
    mu  = (r - 0.5 * sigma**2) / sigma**2
    sqT = sigma * np.sqrt(T)

    x1 = np.log(S0 / K)        / sqT + (1 + mu) * sqT
    x2 = np.log(S0 / B)        / sqT + (1 + mu) * sqT
    y1 = np.log(B**2 / (S0*K)) / sqT + (1 + mu) * sqT
    y2 = np.log(B / S0)        / sqT + (1 + mu) * sqT

    eta = 1.0 if option_type == 'call' else -1.0
    phi = 1.0 if 'down' in barrier_type else -1.0

    A  = (eta * S0 * norm.cdf(eta * x1)
          - eta * K * np.exp(-r * T) * norm.cdf(eta * (x1 - sqT)))
    Bv = (eta * S0 * norm.cdf(eta * x2)
          - eta * K * np.exp(-r * T) * norm.cdf(eta * (x2 - sqT)))
    C  = (eta * S0 * (B / S0)**(2*(mu+1)) * norm.cdf(phi * y1)
          - eta * K * np.exp(-r * T) * (B / S0)**(2*mu) * norm.cdf(phi * (y1 - sqT)))
    D  = (eta * S0 * (B / S0)**(2*(mu+1)) * norm.cdf(phi * y2)
          - eta * K * np.exp(-r * T) * (B / S0)**(2*mu) * norm.cdf(phi * (y2 - sqT)))

    vanilla = (bs_call(S0, K, r, sigma, T) if option_type == 'call'
               else bs_put(S0, K, r, sigma, T))

    if barrier_type == 'down-out':
        return A - C if B <= K else Bv - D
    elif barrier_type == 'up-out':
        if option_type == 'call' and B <= K:
            return 0.0
        elif option_type == 'call':
            return A - Bv + C - D
        elif option_type == 'put' and B >= K:
            return Bv - D
        else:
            return A - C
    elif barrier_type == 'down-in':
        return vanilla - cf_barrier(S0, K, B, r, sigma, T, option_type, 'down-out')
    elif barrier_type == 'up-in':
        return vanilla - cf_barrier(S0, K, B, r, sigma, T, option_type, 'up-out')


# ══════════════════════════════════════════════════════════════════════════════
# PDGM Model
# ══════════════════════════════════════════════════════════════════════════════

class PDGM(nn.Module):
    """
    LSTM + Feedforward NN for path-dependent option pricing (PDGM approach).

    The LSTM plays the same role as in the original geometric-Asian notebook:
    it encodes the path history S_0, …, S_t into a hidden state h_t.
    The FFN then approximates f(S_t, t; h_t) ≈ option price at time t.

    Training loss
    -------------
    L = Σ_t |∂f/∂t + r·S·∂f/∂S + ½σ²S²·∂²f/∂S² − r·f|²   (PDE residual)
      + n_steps · E[|f(S_T, T; h_T) − payoff(path)|²]          (terminal BC)

    The spatial derivatives ∂f/∂S and ∂²f/∂S² are computed by finite
    differences in S while keeping the LSTM state h fixed (same technique as
    the original notebook).  The time derivative ∂f/∂t is approximated by
    advancing the full system one step forward along the path.
    """

    def __init__(self, n_a: int = 64, ffn_hidden: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_a = n_a

        # Full-sequence LSTM (more efficient than LSTMCell on CPU/GPU)
        self.lstm = nn.LSTM(input_size=1, hidden_size=n_a, batch_first=True)

        # FFN:  [S_t | t | h_t]  →  f(S_t, t)
        dims = [1 + 1 + n_a] + [ffn_hidden] * n_layers + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.ffn = nn.Sequential(*layers)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _ffn(self, S: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Evaluate the FFN.  S, t: (N,)  |  h: (N, n_a)  →  (N,)."""
        inp = torch.cat([S.unsqueeze(-1), t.unsqueeze(-1), h], dim=-1)
        return self.ffn(inp).squeeze(-1)

    # ── loss function ──────────────────────────────────────────────────────────

    def loss_fn(self, paths, ts, r, sigma, payoff_fn):
        """
        Parameters
        ----------
        paths      : (B, n_steps+1)  GBM sample paths
        ts         : (n_steps+1,)    time grid [0, T]
        r, sigma   : Black-Scholes parameters
        payoff_fn  : Callable[[Tensor(B, n_steps+1)], Tensor(B,)]

        Returns
        -------
        loss  : scalar
        f_T   : (B,) NN price at maturity (detached, for monitoring)
        """
        B, T1 = paths.shape
        n_steps = T1 - 1
        dt = (ts[1] - ts[0]).item()

        # 1. Run LSTM over the full path at once
        #    lstm_out[:, i, :] = h_i  (hidden state after seeing S_0 … S_i)
        lstm_out, _ = self.lstm(paths.unsqueeze(-1))          # (B, n_steps+1, n_a)

        # 2. PDE residual loss  (all interior time steps, vectorised)
        S_c = paths[:, :-1].reshape(-1)                       # (B*n_steps,)
        S_n = paths[:, 1:].reshape(-1)
        h_c = lstm_out[:, :-1].reshape(-1, self.n_a)
        h_n = lstm_out[:, 1:].reshape(-1, self.n_a)

        t_c = ts[:-1].unsqueeze(0).expand(B, -1).reshape(-1)  # (B*n_steps,)
        t_n = ts[1:].unsqueeze(0).expand(B, -1).reshape(-1)

        bump = (0.01 * S_c).clamp(min=1e-4)

        # Evaluate FFN at current, bumped-up, bumped-down, and next time step
        f      = self._ffn(S_c,        t_c, h_c)
        f_up   = self._ffn(S_c + bump, t_c, h_c)   # h fixed → spatial deriv
        f_down = self._ffn(S_c - bump, t_c, h_c)   # h fixed → spatial deriv
        f_next = self._ffn(S_n,        t_n, h_n)   # advance path → time deriv

        df_dS   = (f_up - f_down) / (2.0 * bump)
        d2f_dS2 = (f_up - 2.0 * f + f_down) / bump.pow(2)
        df_dt   = (f_next - f) / dt

        # Black-Scholes PDE: ∂f/∂t + r·S·∂f/∂S + ½σ²S²·∂²f/∂S² − r·f = 0
        res = (df_dt
               + r * S_c * df_dS
               + 0.5 * sigma**2 * S_c.pow(2) * d2f_dS2
               - r * f)
        pde_loss = res.pow(2).mean()

        # 3. Terminal condition loss (weighted by n_steps, same as original)
        S_T  = paths[:, -1]
        h_T  = lstm_out[:, -1]
        t_T  = ts[-1].expand(B)
        f_T  = self._ffn(S_T, t_T, h_T)

        payoff = payoff_fn(paths)
        terminal_loss = (f_T - payoff).pow(2).mean() * n_steps

        return pde_loss + terminal_loss, f_T.detach()

    # ── inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def price_at_t0(self, S0: float, ts: torch.Tensor) -> float:
        """Return f(S0, 0) using the LSTM state after observing only S0."""
        device = next(self.parameters()).device
        S0_t = torch.tensor([[[S0]]], dtype=torch.float32, device=device)
        h0, _ = self.lstm(S0_t)                               # (1, 1, n_a)
        S_in = torch.tensor([S0], device=device)
        t_in = torch.tensor([ts[0].item()], device=device)
        return self._ffn(S_in, t_in, h0[:, 0, :]).item()


# ══════════════════════════════════════════════════════════════════════════════
# GBM path generation  (Euler-Maruyama under risk-neutral measure)
# ══════════════════════════════════════════════════════════════════════════════

def generate_paths(batch_size, n_steps, T, r, sigma, S0=1.0, device='cpu'):
    dt = T / n_steps
    paths = torch.zeros(batch_size, n_steps + 1, device=device)
    paths[:, 0] = S0
    sqrt_dt = math.sqrt(dt)
    for i in range(n_steps):
        Z = torch.randn(batch_size, device=device)
        paths[:, i + 1] = paths[:, i] * torch.exp(
            (r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * Z
        )
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Barrier payoff factory
# ══════════════════════════════════════════════════════════════════════════════

def make_barrier_payoff(K, B, option_type, barrier_type):
    """
    Returns a function  paths → (B,) terminal payoff.
    Barrier monitoring is discrete (at every recorded time step).
    This replaces terminal_condition() from the original notebook.
    """
    def payoff_fn(paths):
        S_T = paths[:, -1]
        vanilla = (torch.clamp(S_T - K, min=0.0) if option_type == 'call'
                   else torch.clamp(K - S_T, min=0.0))
        crossed = (paths.min(dim=1)[0] <= B if 'down' in barrier_type
                   else paths.max(dim=1)[0] >= B)
        zero = torch.zeros_like(vanilla)
        return (torch.where(crossed, zero, vanilla) if 'out' in barrier_type
                else torch.where(crossed, vanilla, zero))
    return payoff_fn


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = 'cpu'
    torch.manual_seed(args.seed)

    out_dir = os.path.join(args.base_dir, 'PDGM_barrier',
                           f'{args.option_type}_{args.barrier_type}')
    os.makedirs(out_dir, exist_ok=True)
    logfile = os.path.join(out_dir, 'log.txt')

    ts        = torch.linspace(0, args.T, args.n_steps + 1, device=device)
    payoff_fn = make_barrier_payoff(args.K, args.B, args.option_type, args.barrier_type)

    model     = PDGM(n_a=args.n_a, ffn_hidden=args.ffn_hidden, n_layers=args.n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)

    losses = []
    pbar   = tqdm.tqdm(total=args.epochs, desc='Training')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        paths = generate_paths(args.batch_size, args.n_steps, args.T,
                               args.r, args.sigma, device=device)
        loss, _ = model.loss_fn(paths, ts, args.r, args.sigma, payoff_fn)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                ep = generate_paths(5000, args.n_steps, args.T,
                                    args.r, args.sigma, device=device)
                mc = math.exp(-args.r * args.T) * payoff_fn(ep).mean().item()
            nn_p = model.price_at_t0(1.0, ts)
            msg = (f"[{epoch+1:5d}] loss={loss.item():.5f}  "
                   f"MC={mc:.4f}  PDGM(t=0)={nn_p:.4f}")
            pbar.write(msg)
            with open(logfile, 'a') as fh:
                fh.write(msg + '\n')

        pbar.update(1)
    pbar.close()

    # ── Final comparison ───────────────────────────────────────────────────────
    cf_price = cf_barrier(1.0, args.K, args.B, args.r, args.sigma, args.T,
                          args.option_type, args.barrier_type)

    with torch.no_grad():
        mc_paths = generate_paths(100_000, args.n_steps, args.T,
                                  args.r, args.sigma, device=device)
        mc_final = math.exp(-args.r * args.T) * payoff_fn(mc_paths).mean().item()

    model.eval()
    nn_final = model.price_at_t0(1.0, ts)

    print("\n" + "=" * 60)
    print(f"  Option : {args.option_type}  {args.barrier_type}")
    print(f"  S0=1.0, K={args.K}, B={args.B}, r={args.r}, sigma={args.sigma}, T={args.T}")
    print("=" * 60)
    print(f"  Closed-form (Table 8.1) : {cf_price:.4f}")
    print(f"  PDGM price   (t=0)      : {nn_final:.4f}")
    print(f"  Monte Carlo  price      : {mc_final:.4f}")
    print("=" * 60)

    with open(logfile, 'a') as fh:
        fh.write("\n=== FINAL COMPARISON ===\n")
        fh.write(f"Closed-form : {cf_price:.4f}\n")
        fh.write(f"PDGM        : {nn_final:.4f}\n")
        fh.write(f"Monte Carlo : {mc_final:.4f}\n")

    # Save artefacts
    np.save(os.path.join(out_dir, 'losses.npy'), np.array(losses))
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))

    fig, ax = plt.subplots()
    ax.semilogy(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title(f'PDGM - {args.option_type} {args.barrier_type}, B={args.B}')
    fig.savefig(os.path.join(out_dir, 'loss.pdf'))
    plt.close(fig)

    print("THE END")
    return cf_price, nn_final, mc_final


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='PDGM pricing of barrier options under Black-Scholes')
    p.add_argument('--base_dir',     default='./numerical_results/', type=str)
    p.add_argument('--seed',         default=42,   type=int)
    p.add_argument('--batch_size',   default=256,  type=int)
    p.add_argument('--n_steps',      default=50,   type=int,
                   help='time-discretisation steps (dt = T/n_steps)')
    p.add_argument('--n_a',          default=64,   type=int,
                   help='LSTM hidden-state size  (n_a in original notebook)')
    p.add_argument('--ffn_hidden',   default=64,   type=int,
                   help='FFN hidden-layer width')
    p.add_argument('--n_layers',     default=3,    type=int,
                   help='FFN depth (number of hidden layers)')
    p.add_argument('--epochs',       default=5000, type=int)
    p.add_argument('--lr',           default=1e-3, type=float)
    p.add_argument('--T',            default=1.0,  type=float)
    p.add_argument('--r',            default=0.05, type=float)
    p.add_argument('--sigma',        default=0.2,  type=float)
    p.add_argument('--K',            default=1.0,  type=float)
    p.add_argument('--B',            default=0.9,  type=float,
                   help='barrier level (B < K for down-out, B > K for up-out)')
    p.add_argument('--option_type',  default='call',
                   choices=['call', 'put'])
    p.add_argument('--barrier_type', default='down-out',
                   choices=['down-out', 'down-in', 'up-out', 'up-in'])
    args = p.parse_args()
    train(args)
