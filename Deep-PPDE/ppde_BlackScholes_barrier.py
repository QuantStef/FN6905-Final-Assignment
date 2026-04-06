"""
Deep-PPDE pricing of barrier options under Black-Scholes.
Compares neural-network prices against the closed-form formulas of Table 8.1.

Usage examples
--------------
# down-and-out call  (B <= K)
python ppde_BlackScholes_barrier.py --barrier_type down-out --option_type call --B 0.9

# up-and-out put  (B >= K)
python ppde_BlackScholes_barrier.py --barrier_type up-out --option_type put --B 1.1
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

from lib.bsde import PPDE_BlackScholes as PPDE
from lib.options import BarrierOption


# =============================================================================
# Closed-form helpers  (same notation as question_a.py / notes eq 8.2.2)
# =============================================================================
def delta_plus(s, r, sigma, tau):
    return (np.log(s) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

def delta_minus(s, r, sigma, tau):
    return (np.log(s) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

def bs_call(S, K, r, sigma, T):
    d1 = delta_plus(S / K, r, sigma, T)
    d2 = delta_minus(S / K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, r, sigma, T):
    d1 = delta_plus(S / K, r, sigma, T)
    d2 = delta_minus(S / K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def cf_barrier(S0, K, B, r, sigma, T, option_type='call', barrier_type='down-out'):
    """Closed-form barrier option price (Reiner-Rubinstein / notes Table 8.1)."""
    mu  = (r - 0.5 * sigma**2) / sigma**2

    x1 = np.log(S0 / K)         / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    x2 = np.log(S0 / B)         / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y1 = np.log(B**2 / (S0*K))  / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y2 = np.log(B / S0)         / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)

    eta = 1.0 if option_type == 'call' else -1.0
    phi = 1.0 if 'down' in barrier_type else -1.0

    sqT = sigma * np.sqrt(T)
    A = (eta*S0*norm.cdf(eta*x1)
         - eta*K*np.exp(-r*T)*norm.cdf(eta*x1 - eta*sqT))
    Bv = (eta*S0*norm.cdf(eta*x2)
          - eta*K*np.exp(-r*T)*norm.cdf(eta*x2 - eta*sqT))
    C = (eta*S0*(B/S0)**(2*(mu+1))*norm.cdf(phi*y1)
         - eta*K*np.exp(-r*T)*(B/S0)**(2*mu)*norm.cdf(phi*y1 - phi*sqT))
    D = (eta*S0*(B/S0)**(2*(mu+1))*norm.cdf(phi*y2)
         - eta*K*np.exp(-r*T)*(B/S0)**(2*mu)*norm.cdf(phi*y2 - phi*sqT))

    vanilla = bs_call(S0, K, r, sigma, T) if option_type == 'call' else bs_put(S0, K, r, sigma, T)

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


# =============================================================================
# Helpers
# =============================================================================
def sample_x0(batch_size, dim, device):
    """Log-normally distributed initial prices around 1."""
    sigma = 0.3
    mu    = 0.08
    tau   = 0.1
    z  = torch.randn(batch_size, dim, device=device)
    x0 = torch.exp((mu - 0.5*sigma**2)*tau + 0.3*math.sqrt(tau)*z)
    return x0


def write(msg, logfile, pbar):
    pbar.write(msg)
    with open(logfile, "a") as f:
        f.write(msg + "\n")


# =============================================================================
# Training
# =============================================================================
def train(T, n_steps, d, mu, sigma, K, B,
          option_type, barrier_type,
          depth, rnn_hidden, ffn_hidden,
          max_updates, batch_size, lag,
          base_dir, device, method):

    logfile = os.path.join(base_dir, "log.txt")
    ts      = torch.linspace(0, T, n_steps+1, device=device)
    option  = BarrierOption(K=K, B=B, option_type=option_type, barrier_type=barrier_type)
    ppde    = PPDE(d, mu, sigma, depth, rnn_hidden, ffn_hidden)
    ppde.to(device)

    optimizer = torch.optim.RMSprop(ppde.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.2)
    pbar   = tqdm.tqdm(total=max_updates)
    losses = []

    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size, d, device)
        if method == "bsde":
            loss, _, _ = ppde.fbsdeint(ts=ts, x0=x0, option=option, lag=lag)
        else:
            loss, _, _ = ppde.conditional_expectation(ts=ts, x0=x0, option=option, lag=lag)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().cpu().item())

        if (idx+1) % 10 == 0:
            with torch.no_grad():
                x0_eval = torch.ones(5000, d, device=device)
                loss_e, Y, payoff = ppde.fbsdeint(ts=ts, x0=x0_eval, option=option, lag=lag)
                mc_price = torch.exp(-mu * ts[-1]) * payoff.mean()
            pbar.update(10)
            write("loss={:.4f}, MC price={:.4f}, predicted={:.4f}".format(
                  loss_e.item(), mc_price.item(), Y[0,0,0].item()), logfile, pbar)

    torch.save({"state": ppde.state_dict(), "loss": losses},
               os.path.join(base_dir, "result.pth.tar"))

    # ── Evaluation plot ──────────────────────────────────────────────────────
    x0_single = torch.ones(1, d, device=device)
    with torch.no_grad():
        x, _ = ppde.sdeint(ts=ts, x0=x0_single)

    fig, ax = plt.subplots()
    ax.plot(ts.cpu().numpy(), x[0, :, 0].cpu().numpy())
    ax.set_ylabel(r"$S(t)$")
    ax.set_title(f"{option_type} {barrier_type}, B={B}, K={K}")
    fig.savefig(os.path.join(base_dir, "path_eval.pdf"))

    pred, mc_pred = [], []
    for idx, t in enumerate(ts[::lag]):
        pred.append(ppde.eval(ts=ts, x=x[:, :(idx*lag)+1, :], lag=lag).detach())
        mc_pred.append(ppde.eval_mc(ts=ts, x=x[:, :(idx*lag)+1, :],
                                    lag=lag, option=option, mc_samples=10000))
    pred    = torch.cat(pred, 0).view(-1).cpu().numpy()
    mc_pred = torch.cat(mc_pred, 0).view(-1).cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(ts[::lag].cpu().numpy(), pred,    '--', label="Deep PPDE")
    ax.plot(ts[::lag].cpu().numpy(), mc_pred, '-',  label="MC")
    ax.set_ylabel(r"$v(t, S_t)$")
    ax.set_title(f"{option_type} {barrier_type}, B={B}, K={K}")
    ax.legend()
    fig.savefig(os.path.join(base_dir, "BS_barrier_PPDE_sol.pdf"))

    # ── Closed-form comparison ────────────────────────────────────────────────
    S0_cf = 1.0   # our x0 = 1 at evaluation
    cf_price = cf_barrier(S0_cf, K, B, r=mu, sigma=sigma, T=T,
                          option_type=option_type, barrier_type=barrier_type)

    # MC price from large sample
    with torch.no_grad():
        x0_mc = torch.ones(50000, d, device=device)
        _, _, payoff_mc = ppde.fbsdeint(ts=ts, x0=x0_mc, option=option, lag=lag)
        mc_price_cf = (math.exp(-mu * T) * payoff_mc.mean()).item()

    nn_price = pred[0]   # PPDE price at t=0

    print("\n" + "="*60)
    print(f"  Option : {option_type}  {barrier_type}")
    print(f"  S0={S0_cf}, K={K}, B={B}, r={mu}, sigma={sigma}, T={T}")
    print("="*60)
    print(f"  Closed-form (Table 8.1) : {cf_price:.4f}")
    print(f"  Deep PPDE price (t=0)   : {nn_price:.4f}")
    print(f"  Monte Carlo price       : {mc_price_cf:.4f}")
    print("="*60)

    with open(logfile, "a") as f:
        f.write("\n=== FINAL COMPARISON ===\n")
        f.write(f"Closed-form : {cf_price:.4f}\n")
        f.write(f"Deep PPDE   : {nn_price:.4f}\n")
        f.write(f"Monte Carlo : {mc_price_cf:.4f}\n")

    print("THE END")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir',     default='./numerical_results/', type=str)
    parser.add_argument('--device',       default=0, type=int)
    parser.add_argument('--use_cuda',     action='store_true', default=True)
    parser.add_argument('--seed',         default=1, type=int)

    parser.add_argument('--batch_size',   default=500, type=int)
    parser.add_argument('--d',            default=1, type=int,
                        help="asset dimension (use 1 for closed-form comparison)")
    parser.add_argument('--max_updates',  default=5000, type=int)
    parser.add_argument('--ffn_hidden',   default=[20, 20], nargs="+", type=int)
    parser.add_argument('--rnn_hidden',   default=20, type=int)
    parser.add_argument('--depth',        default=3, type=int)
    parser.add_argument('--T',            default=1.0, type=float)
    parser.add_argument('--n_steps',      default=100, type=int)
    parser.add_argument('--lag',          default=10, type=int)
    parser.add_argument('--mu',           default=0.05, type=float, help="risk-free rate")
    parser.add_argument('--sigma',        default=0.2,  type=float, help="volatility")
    parser.add_argument('--K',            default=1.0,  type=float, help="strike price")
    parser.add_argument('--B',            default=0.9,  type=float, help="barrier level")
    parser.add_argument('--option_type',  default='call', type=str,
                        choices=['call', 'put'])
    parser.add_argument('--barrier_type', default='down-out', type=str,
                        choices=['down-out', 'down-in', 'up-out', 'up-in'])
    parser.add_argument('--method',       default='bsde', type=str,
                        choices=['bsde', 'orthogonal'])

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = ("cuda:{}".format(args.device)
              if torch.cuda.is_available() and args.use_cuda else "cpu")

    results_path = os.path.join(args.base_dir, "BS_barrier",
                                f"{args.option_type}_{args.barrier_type}", args.method)
    os.makedirs(results_path, exist_ok=True)

    train(T=args.T,
          n_steps=args.n_steps,
          d=args.d,
          mu=args.mu,
          sigma=args.sigma,
          K=args.K,
          B=args.B,
          option_type=args.option_type,
          barrier_type=args.barrier_type,
          depth=args.depth,
          rnn_hidden=args.rnn_hidden,
          ffn_hidden=args.ffn_hidden,
          max_updates=args.max_updates,
          batch_size=args.batch_size,
          lag=args.lag,
          base_dir=results_path,
          device=device,
          method=args.method)
