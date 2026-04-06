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
from lib.options import Lookback, LookbackCall


# =============================================================================
# Closed-form helpers  (Propositions 9.1 and 9.5)
# delta_± from equation (8.2.2): δ±^τ(s) = [log(s) + (r ± σ²/2)τ] / (σ√τ)
# =============================================================================
def delta_plus(s, r, sigma, tau):
    return (np.log(s) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))

def delta_minus(s, r, sigma, tau):
    return (np.log(s) + (r - 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))

def cf_lookback_put(S0, r, sigma, T):
    """Proposition 9.1: floating PUT, payoff = max(S) - S_T.
    Evaluated at t=0 so M_0^0 = S0."""
    M0 = S0
    d_minus = delta_minus(S0/M0, r, sigma, T)   # = delta_minus(1, ...)
    d_plus  = delta_plus (S0/M0, r, sigma, T)
    d_minus_inv = delta_minus(M0/S0, r, sigma, T)
    price = (M0 * np.exp(-r*T) * norm.cdf(-d_minus)
             + S0 * (1 + sigma**2/(2*r)) * norm.cdf(d_plus)
             - S0 * np.exp(-r*T) * (sigma**2/(2*r))
               * (M0/S0)**(2*r/sigma**2) * norm.cdf(-d_minus_inv)
             - S0)
    return price

def cf_lookback_call(S0, r, sigma, T):
    """Proposition 9.5: floating CALL, payoff = S_T - min(S).
    Evaluated at t=0 so m_0^0 = S0."""
    m0 = S0
    d_plus  = delta_plus (S0/m0, r, sigma, T)
    d_minus = delta_minus(S0/m0, r, sigma, T)
    d_minus_inv = delta_minus(m0/S0, r, sigma, T)
    d_plus_inv  = delta_plus (m0/S0, r, sigma, T)
    price = (S0 * norm.cdf(d_plus)
             - m0 * np.exp(-r*T) * norm.cdf(d_minus)
             + np.exp(-r*T) * S0 * (sigma**2/(2*r))
               * (m0/S0)**(2*r/sigma**2) * norm.cdf(d_minus_inv)
             - S0 * (sigma**2/(2*r)) * norm.cdf(-d_plus))
    return price


# =============================================================================
# Helpers
# =============================================================================
def sample_x0(batch_size, dim, device):
    sigma = 0.3
    mu    = 0.08
    tau   = 0.1
    z  = torch.randn(batch_size, dim, device=device)
    x0 = torch.exp((mu - 0.5*sigma**2)*tau + 0.3*math.sqrt(tau)*z)
    return x0


def write(msg, logfile, pbar):
    pbar.write(msg)
    with open(logfile, "a") as f:
        f.write(msg)
        f.write("\n")


# =============================================================================
# Training
# =============================================================================
def train(T, n_steps, d, mu, sigma, depth, rnn_hidden, ffn_hidden,
          max_updates, batch_size, lag, base_dir, device, method, option_type):

    logfile = os.path.join(base_dir, "log.txt")
    ts = torch.linspace(0, T, n_steps+1, device=device)

    # select option: PUT (Prop 9.1) or CALL (Prop 9.5)
    if option_type == "put":
        option = Lookback()
        prop_label = "Prop 9.1 — Floating PUT  (max(S) - S_T)"
    else:
        option = LookbackCall()
        prop_label = "Prop 9.5 — Floating CALL (S_T - min(S))"

    ppde = PPDE(d, mu, sigma, depth, rnn_hidden, ffn_hidden)
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
                mc_price = torch.exp(-mu*ts[-1]) * payoff.mean()
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
    ax.plot(ts.cpu().numpy(), x[0,:,0].cpu().numpy())
    ax.set_ylabel(r"$X(t)$")
    fig.savefig(os.path.join(base_dir, "path_eval.pdf"))

    pred, mc_pred = [], []
    for idx, t in enumerate(ts[::lag]):
        pred.append(ppde.eval(ts=ts, x=x[:,:(idx*lag)+1,:], lag=lag).detach())
        mc_pred.append(ppde.eval_mc(ts=ts, x=x[:,:(idx*lag)+1,:],
                                    lag=lag, option=option, mc_samples=10000))
    pred    = torch.cat(pred, 0).view(-1).cpu().numpy()
    mc_pred = torch.cat(mc_pred, 0).view(-1).cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(ts[::lag].cpu().numpy(), pred,    '--', label="Deep PPDE (LSTM+BSDE)")
    ax.plot(ts[::lag].cpu().numpy(), mc_pred, '-',  label="Monte Carlo")
    ax.set_ylabel(r"$v(t, X_t)$")
    ax.set_title(prop_label)
    ax.legend()
    fig.savefig(os.path.join(base_dir, "BS_lookback_LSTM_sol.pdf"))

    # ── Closed-form comparison ────────────────────────────────────────────────
    S0_cf = 1.0   # x0 = 1 at evaluation
    if option_type == "put":
        cf_price = cf_lookback_put(S0_cf, mu, sigma, T)
    else:
        cf_price = cf_lookback_call(S0_cf, mu, sigma, T)

    with torch.no_grad():
        x0_mc = torch.ones(50000, d, device=device)
        _, _, payoff_mc = ppde.fbsdeint(ts=ts, x0=x0_mc, option=option, lag=lag)
        mc_price_final = (torch.exp(-mu*T) * payoff_mc.mean()).item()

    nn_price = float(pred[0])

    print("\n" + "="*60)
    print(f"  {prop_label}")
    print(f"  S0={S0_cf}, r={mu}, sigma={sigma}, T={T}, d={d}")
    print("="*60)
    print(f"  Closed-form ({('Prop 9.1' if option_type=='put' else 'Prop 9.5')}) : {cf_price:.4f}")
    print(f"  Deep PPDE price (t=0)              : {nn_price:.4f}")
    print(f"  Monte Carlo price                  : {mc_price_final:.4f}")
    print("="*60)

    with open(logfile, "a") as f:
        f.write("\n=== FINAL COMPARISON ===\n")
        f.write(f"Closed-form : {cf_price:.4f}\n")
        f.write(f"Deep PPDE   : {nn_price:.4f}\n")
        f.write(f"Monte Carlo : {mc_price_final:.4f}\n")

    print("THE END")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir',    default='./numerical_results/', type=str)
    parser.add_argument('--device',      default=0, type=int)
    parser.add_argument('--use_cuda',    action='store_true', default=True)
    parser.add_argument('--seed',        default=1, type=int)

    parser.add_argument('--batch_size',  default=500, type=int)
    parser.add_argument('--d',           default=4, type=int)
    parser.add_argument('--max_updates', default=5000, type=int)
    parser.add_argument('--ffn_hidden',  default=[20,20], nargs="+", type=int)
    parser.add_argument('--rnn_hidden',  default=20, type=int)
    parser.add_argument('--depth',       default=3, type=int)
    parser.add_argument('--T',           default=1., type=float)
    parser.add_argument('--n_steps',     default=100, type=int)
    parser.add_argument('--lag',         default=10, type=int)
    parser.add_argument('--mu',          default=0.05, type=float, help="risk-free rate")
    parser.add_argument('--sigma',       default=0.3,  type=float, help="volatility")
    parser.add_argument('--method',      default="bsde", type=str,
                        choices=["bsde", "orthogonal"])
    parser.add_argument('--option_type', default="put", type=str,
                        choices=["put", "call"],
                        help="put = Prop 9.1 (max-S_T), call = Prop 9.5 (S_T-min)")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = ("cuda:{}".format(args.device)
              if torch.cuda.is_available() and args.use_cuda else "cpu")

    results_path = os.path.join(args.base_dir, "BS", args.option_type, args.method)
    os.makedirs(results_path, exist_ok=True)

    train(T=args.T,
          n_steps=args.n_steps,
          d=args.d,
          mu=args.mu,
          sigma=args.sigma,
          depth=args.depth,
          rnn_hidden=args.rnn_hidden,
          ffn_hidden=args.ffn_hidden,
          max_updates=args.max_updates,
          batch_size=args.batch_size,
          lag=args.lag,
          base_dir=results_path,
          device=device,
          method=args.method,
          option_type=args.option_type)
