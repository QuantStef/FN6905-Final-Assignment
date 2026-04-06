import numpy as np
from scipy.stats import norm
import pandas as pd

# =============================================================================
# PARAMETERS
# =============================================================================
S0    = 100.0   # initial stock price
K     = 100.0   # strike price
r     = 0.05    # risk-free rate
sigma = 0.20    # volatility
T     = 1.0     # maturity (years)
N     = 252     # time steps (daily)
M     = 100_000 # number of MC paths

B_down = 90.0   # barrier below S0 (for down-type barriers), B <= K
B_up   = 110.0  # barrier above S0 (for up-type barriers),   B >= K

# =============================================================================
# HELPER: simulate GBM paths
# Returns array of shape (M, N+1), including S0 at column 0
# =============================================================================
def simulate_paths(S0, r, sigma, T, N, M, seed=42):
    np.random.seed(seed)
    dt = T / N
    Z  = np.random.standard_normal((M, N))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.log(S0) + np.cumsum(increments, axis=1)
    paths = np.exp(log_S)
    return np.hstack([np.full((M, 1), S0), paths])   # shape (M, N+1)


# =============================================================================
# MONTE CARLO: BARRIER OPTIONS  (Table 8.1)
# barrier_type: 'down-out', 'down-in', 'up-out', 'up-in'
# option_type : 'call', 'put'
# =============================================================================
def mc_barrier(S0, K, B, r, sigma, T, N, M,
               option_type='call', barrier_type='down-out', seed=42):
    paths = simulate_paths(S0, r, sigma, T, N, M, seed)
    S_T   = paths[:, -1]

    # determine which paths crossed the barrier
    if 'down' in barrier_type:
        crossed = np.min(paths, axis=1) <= B
    else:                                    # up
        crossed = np.max(paths, axis=1) >= B

    # vanilla payoff
    if option_type == 'call':
        vanilla = np.maximum(S_T - K, 0.0)
    else:
        vanilla = np.maximum(K - S_T, 0.0)

    # apply barrier condition
    if 'out' in barrier_type:
        payoff = np.where(crossed, 0.0, vanilla)   # killed if crossed
    else:                                            # in
        payoff = np.where(crossed, vanilla, 0.0)   # alive only if crossed

    price = np.exp(-r * T) * np.mean(payoff)
    se    = np.exp(-r * T) * np.std(payoff) / np.sqrt(M)
    return price, se


# =============================================================================
# MONTE CARLO: LOOKBACK OPTIONS  (Propositions 9.1 and 9.5)
#
# Prop 9.1  -> floating put:  payoff = max(S) - S_T   (M_0^T - S_T)
# Prop 9.5  -> floating call: payoff = S_T - min(S)   (S_T - m_0^T)
# =============================================================================
def mc_lookback(S0, r, sigma, T, N, M, option_type='floating_call', seed=42):
    paths = simulate_paths(S0, r, sigma, T, N, M, seed)
    S_T   = paths[:, -1]

    if option_type == 'floating_call':        # Prop 9.5
        payoff = S_T - np.min(paths, axis=1)
    elif option_type == 'floating_put':       # Prop 9.1
        payoff = np.max(paths, axis=1) - S_T
    else:
        raise ValueError(f"Unknown option_type: {option_type}")

    price = np.exp(-r * T) * np.mean(payoff)
    se    = np.exp(-r * T) * np.std(payoff) / np.sqrt(M)
    return price, se


# =============================================================================
# CLOSED-FORM: helper functions
# delta_pm from equation (8.2.2):
#   delta_±^tau(s) = [ log(s) + (r ± sigma^2/2)*tau ] / (sigma * sqrt(tau))
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


# =============================================================================
# CLOSED-FORM: BARRIER OPTIONS
# Standard Black-Scholes barrier formulas (Merton / Reiner-Rubinstein)
# mu = (r - sigma^2/2) / sigma^2
# =============================================================================
def cf_barrier(S0, K, B, r, sigma, T, option_type='call', barrier_type='down-out'):
    mu  = (r - 0.5 * sigma**2) / (sigma**2)
    lam = np.sqrt(mu**2 + 2 * r / sigma**2)

    x1 = np.log(S0 / K)  / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    x2 = np.log(S0 / B)  / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y1 = np.log(B**2 / (S0 * K)) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y2 = np.log(B / S0) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)

    eta = 1 if option_type == 'call' else -1
    phi = 1 if 'down' in barrier_type else -1

    A = eta * S0 * norm.cdf(eta * x1) \
        - eta * K * np.exp(-r * T) * norm.cdf(eta * x1 - eta * sigma * np.sqrt(T))

    B_ = eta * S0 * norm.cdf(eta * x2) \
         - eta * K * np.exp(-r * T) * norm.cdf(eta * x2 - eta * sigma * np.sqrt(T))

    C = eta * S0 * (B / S0)**(2 * (mu + 1)) * norm.cdf(phi * y1) \
        - eta * K * np.exp(-r * T) * (B / S0)**(2 * mu) * norm.cdf(phi * y1 - phi * sigma * np.sqrt(T))

    D = eta * S0 * (B / S0)**(2 * (mu + 1)) * norm.cdf(phi * y2) \
        - eta * K * np.exp(-r * T) * (B / S0)**(2 * mu) * norm.cdf(phi * y2 - phi * sigma * np.sqrt(T))

    if barrier_type == 'down-out':
        # phi=1, eta=1 for call
        if option_type == 'call' and B <= K:
            return A - C
        elif option_type == 'call' and B > K:
            return B_ - D
        elif option_type == 'put' and B >= K:
            return 0.0          # always knocked out before put pays
        else:                   # put, B < K
            return A - B_ + C - D   # put down-out B<=K: eq (8.2.8)

    elif barrier_type == 'up-out':
        if option_type == 'call' and B <= K:
            return 0.0          # always knocked out before call pays
        elif option_type == 'call' and B > K:
            return A - B_ + C - D
        elif option_type == 'put' and B >= K:
            return B_ - D       # Haug (2007): up-and-out put, H >= K
        else:                   # put, B < K (degenerate: stock starts above barrier)
            return A - C

    elif barrier_type == 'down-in':
        vanilla = bs_call(S0, K, r, sigma, T) if option_type == 'call' \
                  else bs_put(S0, K, r, sigma, T)
        return vanilla - cf_barrier(S0, K, B, r, sigma, T, option_type, 'down-out')

    elif barrier_type == 'up-in':
        vanilla = bs_call(S0, K, r, sigma, T) if option_type == 'call' \
                  else bs_put(S0, K, r, sigma, T)
        return vanilla - cf_barrier(S0, K, B, r, sigma, T, option_type, 'up-out')


# =============================================================================
# CLOSED-FORM: LOOKBACK OPTIONS
# Evaluated at t=0, so M_0^0 = m_0^0 = S0
# Uses delta_± notation from the notes
# =============================================================================
def cf_lookback_put(S0, r, sigma, T):
    """Proposition 9.1: floating put, payoff = max(S) - S_T"""
    # At t=0: M_0^t = S0, tau = T-t = T
    M0 = S0
    ratio = S0 / M0   # = 1

    d_minus = delta_minus(ratio, r, sigma, T)   # delta_-^T(S_t / M_0^t)
    d_plus  = delta_plus(ratio, r, sigma, T)    # delta_+^T(S_t / M_0^t)

    # Prop 9.1 formula:
    # M0*e^{-rT}*Phi(-d_-^T(S/M0)) + S*(1 + sigma^2/(2r))*Phi(d_+^T(S/M0))
    # - S*e^{-rT}*(sigma^2/(2r))*(M0/S)^{2r/sigma^2}*Phi(-d_-^T(M0/S)) - S

    inv_ratio  = M0 / S0
    d_minus_inv = delta_minus(inv_ratio, r, sigma, T)

    price = (M0 * np.exp(-r * T) * norm.cdf(-d_minus)
             + S0 * (1 + sigma**2 / (2 * r)) * norm.cdf(d_plus)
             - S0 * np.exp(-r * T) * (sigma**2 / (2 * r))
               * inv_ratio**(2 * r / sigma**2) * norm.cdf(-d_minus_inv)
             - S0)
    return price


def cf_lookback_call(S0, r, sigma, T):
    """Proposition 9.5: floating call, payoff = S_T - min(S)"""
    # At t=0: m_0^t = S0, tau = T
    m0 = S0
    ratio = S0 / m0   # = 1

    d_plus  = delta_plus(ratio, r, sigma, T)
    d_minus = delta_minus(ratio, r, sigma, T)

    inv_ratio   = m0 / S0
    d_minus_inv = delta_minus(inv_ratio, r, sigma, T)
    d_plus_inv  = delta_plus(inv_ratio, r, sigma, T)

    # Prop 9.5 formula:
    # S*Phi(d_+^T(S/m0)) - m0*e^{-rT}*Phi(d_-^T(S/m0))
    # + e^{-rT}*S*(sigma^2/(2r))*(m0/S)^{2r/sigma^2}*Phi(d_-^T(m0/S))
    # - S*(sigma^2/(2r))*Phi(-d_+^T(S/m0))

    price = (S0 * norm.cdf(d_plus)
             - m0 * np.exp(-r * T) * norm.cdf(d_minus)
             + np.exp(-r * T) * S0 * (sigma**2 / (2 * r))
               * inv_ratio**(2 * r / sigma**2) * norm.cdf(d_minus_inv)
             - S0 * (sigma**2 / (2 * r)) * norm.cdf(-d_plus))
    return price


# =============================================================================
# RESULTS TABLE
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print(f"Parameters: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
    print(f"            N={N} steps, M={M:,} paths")
    print("=" * 70)

    # ── Barrier options ──────────────────────────────────────────────────────
    print("\n--- BARRIER OPTIONS ---\n")
    barrier_cases = [
        ('call', 'down-out', B_down, 'B<=K'),
        ('call', 'down-in',  B_down, 'B<=K'),
        ('call', 'up-out',   B_up,   'B>=K'),
        ('call', 'up-in',    B_up,   'B>=K'),
        ('put',  'down-out', B_down, 'B<=K'),
        ('put',  'down-in',  B_down, 'B<=K'),
        ('put',  'up-out',   B_up,   'B>=K'),
        ('put',  'up-in',    B_up,   'B>=K'),
    ]

    rows = []
    for opt, bar, B, note in barrier_cases:
        mc_price, mc_se = mc_barrier(S0, K, B, r, sigma, T, N, M, opt, bar)
        cf_price = cf_barrier(S0, K, B, r, sigma, T, opt, bar)
        rows.append({
            'Type': f'{opt} {bar}',
            'B': B,
            'Condition': note,
            'MC Price': round(mc_price, 4),
            '±2 SE': round(2 * mc_se, 4),
            'Closed-Form': round(cf_price, 4),
            'Diff': round(mc_price - cf_price, 4),
        })

    df_barrier = pd.DataFrame(rows)
    print(df_barrier.to_string(index=False))

    # ── Lookback options ─────────────────────────────────────────────────────
    print("\n--- LOOKBACK OPTIONS ---\n")

    mc_put,  mc_put_se  = mc_lookback(S0, r, sigma, T, N, M, 'floating_put')
    mc_call, mc_call_se = mc_lookback(S0, r, sigma, T, N, M, 'floating_call')
    cf_put  = cf_lookback_put(S0, r, sigma, T)
    cf_call = cf_lookback_call(S0, r, sigma, T)

    lb_rows = [
        {'Option': 'Floating Put  (Prop 9.1)',
         'MC Price': round(mc_put, 4),  '±2 SE': round(2*mc_put_se, 4),
         'Closed-Form': round(cf_put, 4),  'Diff': round(mc_put - cf_put, 4)},
        {'Option': 'Floating Call (Prop 9.5)',
         'MC Price': round(mc_call, 4), '±2 SE': round(2*mc_call_se, 4),
         'Closed-Form': round(cf_call, 4), 'Diff': round(mc_call - cf_call, 4)},
    ]
    df_lb = pd.DataFrame(lb_rows)
    print(df_lb.to_string(index=False))

    # ── Parity checks ────────────────────────────────────────────────────────
    print("\n--- PARITY CHECKS (knock-out + knock-in = vanilla) ---\n")
    vanilla_call = bs_call(S0, K, r, sigma, T)
    vanilla_put  = bs_put(S0, K, r, sigma, T)

    do_c, _ = mc_barrier(S0, K, B_down, r, sigma, T, N, M, 'call', 'down-out')
    di_c, _ = mc_barrier(S0, K, B_down, r, sigma, T, N, M, 'call', 'down-in')
    uo_p, _ = mc_barrier(S0, K, B_up,   r, sigma, T, N, M, 'put',  'up-out')
    ui_p, _ = mc_barrier(S0, K, B_up,   r, sigma, T, N, M, 'put',  'up-in')

    print(f"  MC:  down-out call + down-in call = {do_c+di_c:.4f}  |  BS Call = {vanilla_call:.4f}")
    print(f"  MC:  up-out put   + up-in put     = {uo_p+ui_p:.4f}  |  BS Put  = {vanilla_put:.4f}")

    # ── Note on discretization bias ───────────────────────────────────────────
    print("\n--- NOTE ON DISCRETIZATION BIAS ---")
    print(f"  MC uses N={N} discrete time steps (daily monitoring).")
    print(f"  Closed-form assumes *continuous* monitoring.")
    print(f"  Discrete barriers are effectively shifted by +/-0.5826*sigma*sqrt(T/N).")
    corr = 0.5826 * sigma * np.sqrt(T / N)
    print(f"  Continuity correction: {corr:.5f} (Broadie-Glasserman-Kou 1997)")
    print(f"  => Differences between MC and CF are expected; they vanish as N -> inf.")
