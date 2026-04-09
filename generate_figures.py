"""
Generate all figures (PNG) for the LaTeX report.
Run this script first, then compile main.tex with pdflatex.

Output: figures/fig1_barrier_mc.png  ... figures/fig7_crossmethod.png
"""
import os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

NR = os.path.join(BASE, "numerical_results")
NR_PDGM = os.path.join(BASE, "PDGM-Geometric_Asian", "numerical_results")

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150,
})

BLUE   = "#1976D2"
RED    = "#E53935"
GREEN  = "#388E3C"
ORANGE = "#F57C00"
PURPLE = "#7B1FA2"
BLACK  = "#212121"

# ─── helpers ──────────────────────────────────────────────────────────────────

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {name}")

def parse_log(path):
    losses = []
    with open(path) as f:
        for line in f:
            m = re.search(r"loss=([\d.]+)", line)
            if m:
                losses.append(float(m.group(1)))
    return np.array(losses)

# ─── Fig 1: Part (a) barrier bar chart ────────────────────────────────────────
labels = ["DO\ncall","DI\ncall","UO\ncall","UI\ncall",
          "DO\nput", "DI\nput", "UO\nput", "UI\nput"]
mc_v = [8.8635,1.5359,0.1564,10.2429,0.1932,5.3586,4.3636,1.1882]
cf_v = [8.6655,1.7851,0.1186,10.3320,0.1512,5.4223,4.0796,1.4939]

fig, ax = plt.subplots(figsize=(9,4))
x = np.arange(8); w = 0.35
ax.bar(x-w/2, mc_v, w, label="Monte Carlo", color=BLUE,  alpha=0.88)
ax.bar(x+w/2, cf_v, w, label="Closed-Form",  color=RED,   alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Option Price  ($S_0 = 100$)")
ax.set_title("Part (a) — Barrier Options: Monte Carlo vs.\ Closed-Form")
ax.legend(); ax.grid(axis="y", lw=0.5, alpha=0.4)
fig.tight_layout(); save(fig, "fig1_barrier_mc.png")

# ─── Fig 2: Part (a) lookback bar chart ───────────────────────────────────────
lb_labels = ["Floating PUT\n$\\max(S)-S_T$", "Floating CALL\n$S_T-\\min(S)$"]
lb_mc = [13.4291, 16.5870]
lb_cf = [14.2906, 17.2168]

fig, ax = plt.subplots(figsize=(5.5, 4))
x2 = np.arange(2)
b1 = ax.bar(x2-0.2, lb_mc, 0.38, label="Monte Carlo", color=BLUE,  alpha=0.88)
b2 = ax.bar(x2+0.2, lb_cf, 0.38, label="Closed-Form",  color=RED,   alpha=0.88)
ax.set_xticks(x2); ax.set_xticklabels(lb_labels)
ax.set_ylabel("Option Price  ($S_0 = 100$)")
ax.set_title("Part (a) — Lookback Options (Props 9.1 \& 9.5)")
ax.legend(); ax.grid(axis="y", lw=0.5, alpha=0.4)
for bar, v in zip(b1, lb_mc):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f"{v:.2f}", ha="center", fontsize=7)
for bar, v in zip(b2, lb_cf):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f"{v:.2f}", ha="center", fontsize=7)
fig.tight_layout(); save(fig, "fig2_lookback_mc.png")

# ─── Fig 3: Part (b) Deep PPDE lookback loss ──────────────────────────────────
lp  = parse_log(os.path.join(NR, "BS/put/bsde/log.txt"))
lc  = parse_log(os.path.join(NR, "BS/call/bsde/log.txt"))
itr = lambda n: np.arange(10, 10*n+1, 10)

fig, ax = plt.subplots(figsize=(7,3.5))
ax.semilogy(itr(len(lp)), lp, color=BLUE,   lw=1.5, label="Lookback PUT  (Prop 9.1)")
ax.semilogy(itr(len(lc)), lc, color=RED,    lw=1.5, label="Lookback CALL (Prop 9.5)")
ax.set_xlabel("Training iteration"); ax.set_ylabel("Loss (log scale)")
ax.set_title("Part (b) — Deep PPDE Training Loss: Lookback Options")
ax.legend(); ax.grid(lw=0.5, alpha=0.35)
fig.tight_layout(); save(fig, "fig3_ppde_lookback_loss.png")

# ─── Fig 4: Part (b) Deep PPDE barrier loss ───────────────────────────────────
ld = parse_log(os.path.join(NR, "BS_barrier/call_down-out/bsde/log.txt"))
lu = parse_log(os.path.join(NR, "BS_barrier/put_up-out/bsde/log.txt"))

fig, ax = plt.subplots(figsize=(7,3.5))
ax.semilogy(itr(len(ld)), ld, color=GREEN,  lw=1.5, label="Down-out CALL ($B=0.9$)")
ax.semilogy(itr(len(lu)), lu, color=ORANGE, lw=1.5, label="Up-out PUT   ($B=1.1$)")
ax.set_xlabel("Training iteration"); ax.set_ylabel("Loss (log scale)")
ax.set_title("Part (b) — Deep PPDE Training Loss: Barrier Options")
ax.legend(); ax.grid(lw=0.5, alpha=0.35)
fig.tight_layout(); save(fig, "fig4_ppde_barrier_loss.png")

# ─── Fig 5: Part (c) PDGM lookback loss ───────────────────────────────────────
lp2 = np.load(os.path.join(NR_PDGM, "PDGM_lookback/put/losses.npy"))
lc2 = np.load(os.path.join(NR_PDGM, "PDGM_lookback/call/losses.npy"))

fig, ax = plt.subplots(figsize=(7,3.5))
ax.semilogy(lp2, color=BLUE,  lw=1.5, label="Lookback PUT  (Prop 9.1)")
ax.semilogy(lc2, color=RED,   lw=1.5, label="Lookback CALL (Prop 9.5)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
ax.set_title("Part (c) — PDGM Training Loss: Lookback Options (2000 epochs)")
ax.legend(); ax.grid(lw=0.5, alpha=0.35)
fig.tight_layout(); save(fig, "fig5_pdgm_lookback_loss.png")

# ─── Fig 6: Part (c) PDGM barrier loss ────────────────────────────────────────
ld2 = np.load(os.path.join(NR_PDGM, "PDGM_barrier/call_down-out/losses.npy"))
lu2 = np.load(os.path.join(NR_PDGM, "PDGM_barrier/put_up-out/losses.npy"))

fig, ax = plt.subplots(figsize=(7,3.5))
ax.semilogy(ld2, color=GREEN,  lw=1.5, label="Down-out CALL ($B=0.9$)")
ax.semilogy(lu2, color=ORANGE, lw=1.5, label="Up-out PUT   ($B=1.1$)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
ax.set_title("Part (c) — PDGM Training Loss: Barrier Options (3000 epochs)")
ax.legend(); ax.grid(lw=0.5, alpha=0.35)
fig.tight_layout(); save(fig, "fig6_pdgm_barrier_loss.png")

# ─── Fig 7: Cross-method barrier comparison ───────────────────────────────────
opts   = ["Down-out call ($B{=}0.9$)", "Up-out put ($B{=}1.1$)"]
cf_p   = [0.0867, 0.0408]
mc_p   = [0.0886, 0.0436]
ppde_p = [0.0802, 0.0529]
pdgm_p = [0.0914, 0.0398]

fig, ax = plt.subplots(figsize=(7,4))
x = np.arange(2); w = 0.18
ax.bar(x-1.5*w, cf_p,   w, label="Closed-Form",      color=BLACK,  alpha=0.88)
ax.bar(x-0.5*w, mc_p,   w, label="MC (Part a)",       color=BLUE,   alpha=0.88)
ax.bar(x+0.5*w, ppde_p, w, label="Deep PPDE (Part b)",color=PURPLE, alpha=0.88)
ax.bar(x+1.5*w, pdgm_p, w, label="PDGM (Part c)",     color=RED,    alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(opts)
ax.set_ylabel("Option Price  ($S_0{=}1.0$, $\\sigma{=}0.2$)")
ax.set_title("Cross-Method Comparison: Barrier Options (Parts a, b, c)")
ax.legend(fontsize=8); ax.grid(axis="y", lw=0.5, alpha=0.4)
fig.tight_layout(); save(fig, "fig7_crossmethod.png")

print("\nAll figures saved to:", FIG_DIR)
