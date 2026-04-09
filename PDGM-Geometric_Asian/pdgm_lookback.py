"""
Physics-Driven Galerkin Method (PDGM) for lookback option pricing
under Black-Scholes dynamics.

Adapted from:  https://github.com/zhaoyu-zhang/PDGM-Geometric_Asian
Purpose:       Extend PDGM to lookback options and compare with the
               closed-form expressions of Propositions 9.1 and 9.5.

Architecture (same as the original geometric-Asian notebook)
------------------------------------------------------------
  LSTM (full-sequence)  : encodes path history S_0, …, S_t → hidden state h_t
                          The LSTM naturally learns to track the running
                          maximum (for PUT) or running minimum (for CALL).
  Feedforward NN (3×64) : maps (S_t, t, h_t) → option price f_t
  PDE loss              : Black-Scholes PDE residual at every path point
  Terminal loss         : lookback payoff condition at maturity

Key modifications vs. the original
------------------------------------
  • terminal_condition()  replaced by lookback payoff functions:
      PUT  (Prop 9.1):  payoff = max_{0≤s≤T}(S_s) − S_T   (floating)
      CALL (Prop 9.5):  payoff = S_T − min_{0≤s≤T}(S_s)   (floating)
  • Closed-form comparison: Propositions 9.1 and 9.5
  • Re-implemented in PyTorch (original uses TF 1.x)
  • Vectorised LSTM + FFN evaluation for speed

Usage
-----
  python pdgm_lookback.py --option_type put
  python pdgm_lookback.py --option_type call
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
# Closed-form helpers
# δ±^τ(z) = [log(z) + (r ± σ²/2)τ] / (σ√τ)   (eq. 8.2.2)
# ══════════════════════════════════════════════════════════════════════════════

def _d_plus(z, r, sigma, tau):
    return (np.log(z) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

def _d_minus(z, r, sigma, tau):
    return (np.log(z) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def cf_lookback_put(S0, r, sigma, T):
    """
    Proposition 9.1 — Floating PUT, payoff = max(S) − S_T.
    Evaluated at t=0, so M_0^0 = S0 (the running max at time 0 is just S0).
    """
    M0 = S0
    dp  = _d_plus (S0 / M0, r, sigma, T)    # = d+(1, …) = (r + σ²/2)T / (σ√T)
    dm  = _d_minus(S0 / M0, r, sigma, T)
    dm_inv = _d_minus(M0 / S0, r, sigma, T)  # = −dm
    price = (M0 * np.exp(-r * T) * norm.cdf(-dm)
             + S0 * (1 + sigma**2 / (2 * r)) * norm.cdf(dp)
             - S0 * np.exp(-r * T) * (sigma**2 / (2 * r))
               * (M0 / S0)**(2 * r / sigma**2) * norm.cdf(-dm_inv)
             - S0)
    return price


def cf_lookback_call(S0, r, sigma, T):
    """
    Proposition 9.5 — Floating CALL, payoff = S_T − min(S).
    Evaluated at t=0, so m_0^0 = S0.
    """
    m0 = S0
    dp      = _d_plus (S0 / m0, r, sigma, T)
    dm      = _d_minus(S0 / m0, r, sigma, T)
    dm_inv  = _d_minus(m0 / S0, r, sigma, T)
    dp_inv  = _d_plus (m0 / S0, r, sigma, T)
    price = (S0 * norm.cdf(dp)
             - m0 * np.exp(-r * T) * norm.cdf(dm)
             + np.exp(-r * T) * S0 * (sigma**2 / (2 * r))
               * (m0 / S0)**(2 * r / sigma**2) * norm.cdf(dm_inv)
             - S0 * (sigma**2 / (2 * r)) * norm.cdf(-dp))
    return price


# ══════════════════════════════════════════════════════════════════════════════
# PDGM Model
# ══════════════════════════════════════════════════════════════════════════════

class PDGM(nn.Module):
    """
    LSTM + Feedforward NN for path-dependent option pricing (PDGM approach).

    For lookback options the LSTM hidden state h_t encodes the running
    maximum (PUT) or running minimum (CALL) of the stock price path.
    The FFN then maps (S_t, t, h_t) to the current option value.

    Training loss
    -------------
    L = Σ_t |∂f/∂t + r·S·∂f/∂S + ½σ²S²·∂²f/∂S² − r·f|²   (PDE residual)
      + n_steps · E[|f(S_T, T; h_T) − payoff(path)|²]          (terminal BC)

    Spatial derivatives are computed by finite differences in S with the
    LSTM state h held fixed (same technique as the original notebook).
    The time derivative is approximated by advancing the full system one
    step forward along the simulated path.
    """

    def __init__(self, n_a: int = 64, ffn_hidden: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_a = n_a

        # Full-sequence LSTM
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
        """S, t: (N,)  |  h: (N, n_a)  →  (N,)."""
        inp = torch.cat([S.unsqueeze(-1), t.unsqueeze(-1), h], dim=-1)
        return self.ffn(inp).squeeze(-1)

    # ── loss function ──────────────────────────────────────────────────────────

    def loss_fn(self, paths, ts, r, sigma, payoff_fn):
        """
        Parameters
        ----------
        paths      : (B, n_steps+1)
        ts         : (n_steps+1,)
        r, sigma   : Black-Scholes parameters
        payoff_fn  : Callable[[Tensor(B, n_steps+1)], Tensor(B,)]

        Returns
        -------
        loss  : scalar
        f_T   : (B,) NN price at maturity (detached)
        """
        B, T1 = paths.shape
        n_steps = T1 - 1
        dt = (ts[1] - ts[0]).item()

        # 1. Run LSTM over the full path
        lstm_out, _ = self.lstm(paths.unsqueeze(-1))          # (B, n_steps+1, n_a)

        # 2. PDE residual (all interior steps, vectorised)
        S_c = paths[:, :-1].reshape(-1)
        S_n = paths[:, 1:].reshape(-1)
        h_c = lstm_out[:, :-1].reshape(-1, self.n_a)
        h_n = lstm_out[:, 1:].reshape(-1, self.n_a)

        t_c = ts[:-1].unsqueeze(0).expand(B, -1).reshape(-1)
        t_n = ts[1:].unsqueeze(0).expand(B, -1).reshape(-1)

        bump = (0.01 * S_c).clamp(min=1e-4)

        f      = self._ffn(S_c,        t_c, h_c)
        f_up   = self._ffn(S_c + bump, t_c, h_c)   # h fixed → spatial deriv
        f_down = self._ffn(S_c - bump, t_c, h_c)   # h fixed → spatial deriv
        f_next = self._ffn(S_n,        t_n, h_n)   # advance → time deriv

        df_dS   = (f_up - f_down) / (2.0 * bump)
        d2f_dS2 = (f_up - 2.0 * f + f_down) / bump.pow(2)
        df_dt   = (f_next - f) / dt

        res = (df_dt
               + r * S_c * df_dS
               + 0.5 * sigma**2 * S_c.pow(2) * d2f_dS2
               - r * f)
        pde_loss = res.pow(2).mean()

        # 3. Terminal condition
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
        h0, _ = self.lstm(S0_t)
        S_in = torch.tensor([S0], device=device)
        t_in = torch.tensor([ts[0].item()], device=device)
        return self._ffn(S_in, t_in, h0[:, 0, :]).item()


# ══════════════════════════════════════════════════════════════════════════════
# GBM path generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_paths(batch_size, n_steps, T, r, sigma, S0=1.0, device='cpu'):
    """GBM paths under risk-neutral measure (Euler-Maruyama)."""
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
# Lookback payoff functions
# ══════════════════════════════════════════════════════════════════════════════

def lookback_put_payoff(paths):
    """Prop 9.1 — Floating PUT: payoff = max(S_0,…,S_T) − S_T."""
    return paths.max(dim=1)[0] - paths[:, -1]

def lookback_call_payoff(paths):
    """Prop 9.5 — Floating CALL: payoff = S_T − min(S_0,…,S_T)."""
    return paths[:, -1] - paths.min(dim=1)[0]


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = 'cpu'
    torch.manual_seed(args.seed)

    out_dir = os.path.join(args.base_dir, 'PDGM_lookback', args.option_type)
    os.makedirs(out_dir, exist_ok=True)
    logfile = os.path.join(out_dir, 'log.txt')

    ts = torch.linspace(0, args.T, args.n_steps + 1, device=device)

    if args.option_type == 'put':
        payoff_fn  = lookback_put_payoff
        prop_label = 'Prop 9.1 - Floating PUT  (max(S) - S_T)'
    else:
        payoff_fn  = lookback_call_payoff
        prop_label = 'Prop 9.5 - Floating CALL (S_T - min(S))'

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
    S0_cf = 1.0
    cf_price = (cf_lookback_put(S0_cf, args.r, args.sigma, args.T)
                if args.option_type == 'put'
                else cf_lookback_call(S0_cf, args.r, args.sigma, args.T))

    with torch.no_grad():
        mc_paths = generate_paths(100_000, args.n_steps, args.T,
                                  args.r, args.sigma, device=device)
        mc_final = math.exp(-args.r * args.T) * payoff_fn(mc_paths).mean().item()

    model.eval()
    nn_final = model.price_at_t0(1.0, ts)

    prop_id = 'Prop 9.1' if args.option_type == 'put' else 'Prop 9.5'

    print("\n" + "=" * 60)
    print(f"  {prop_label}")
    print(f"  S0=1.0, r={args.r}, sigma={args.sigma}, T={args.T}")
    print("=" * 60)
    print(f"  Closed-form ({prop_id}) : {cf_price:.4f}")
    print(f"  PDGM price   (t=0)      : {nn_final:.4f}")
    print(f"  Monte Carlo  price      : {mc_final:.4f}")
    print("=" * 60)

    with open(logfile, 'a') as fh:
        fh.write("\n=== FINAL COMPARISON ===\n")
        fh.write(f"Closed-form ({prop_id}) : {cf_price:.4f}\n")
        fh.write(f"PDGM                   : {nn_final:.4f}\n")
        fh.write(f"Monte Carlo            : {mc_final:.4f}\n")

    # Save artefacts
    np.save(os.path.join(out_dir, 'losses.npy'), np.array(losses))
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))

    fig, ax = plt.subplots()
    ax.semilogy(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title(f'PDGM - {prop_label}')
    fig.savefig(os.path.join(out_dir, 'loss.pdf'))
    plt.close(fig)

    print("THE END")
    return cf_price, nn_final, mc_final


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='PDGM pricing of lookback options under Black-Scholes')
    p.add_argument('--base_dir',    default='./numerical_results/', type=str)
    p.add_argument('--seed',        default=42,    type=int)
    p.add_argument('--batch_size',  default=256,   type=int)
    p.add_argument('--n_steps',     default=50,    type=int,
                   help='time-discretisation steps')
    p.add_argument('--n_a',         default=64,    type=int,
                   help='LSTM hidden-state size  (n_a in original notebook)')
    p.add_argument('--ffn_hidden',  default=64,    type=int)
    p.add_argument('--n_layers',    default=3,     type=int)
    p.add_argument('--epochs',      default=5000,  type=int)
    p.add_argument('--lr',          default=1e-3,  type=float)
    p.add_argument('--T',           default=1.0,   type=float)
    p.add_argument('--r',           default=0.05,  type=float)
    p.add_argument('--sigma',       default=0.2,   type=float)
    p.add_argument('--option_type', default='put',
                   choices=['put', 'call'],
                   help='put = Prop 9.1 (max(S)−S_T) | call = Prop 9.5 (S_T−min(S))')
    args = p.parse_args()
    train(args)
