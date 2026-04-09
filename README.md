# FN6905 Exotic Options & Structured Products — Final Assignment

**Student:** Stefanus Gunawan (G2500310J)
**Course:** FN6905, Nanyang Business School, NTU
**Due:** April 10, 2026

## Overview

Pricing of barrier and lookback options using three methods:

| Part | Method | Code |
|------|--------|------|
| (a) | Monte Carlo | `question_a.py` |
| (b) | Deep PPDE (Sabate-Vidales et al., 2020) | `Deep-PPDE/` |
| (c) | PDGM (Saporito & Zhang, 2021) | `PDGM-Geometric_Asian/pdgm_barrier.py`, `pdgm_lookback.py` |

All results are compared against closed-form expressions (Table 8.1, Propositions 9.1 & 9.5).

## Repository Structure

```
.
├── question_a.py              # Part (a): Monte Carlo pricer
├── generate_report_a.py       # Part (a): PDF report generator
├── generate_report_b.py       # Part (b): PDF report generator
├── generate_report_c.py       # Part (c): PDF report generator
├── generate_figures.py        # Generates all figures for LaTeX
├── main.tex                   # Combined LaTeX report (all 3 parts)
├── pyproject.toml             # Python dependencies (managed by uv)
├── uv.lock                    # Locked dependency versions
├── numerical_results/         # Deep PPDE training logs & results
│   ├── BS/                    # Lookback option results
│   └── BS_barrier/            # Barrier option results
└── PDGM-Geometric_Asian/
    ├── pdgm_barrier.py        # Modified PDGM for barrier options
    ├── pdgm_lookback.py       # Modified PDGM for lookback options
    ├── PDGM_geometric_asian.ipynb  # Original notebook (reference)
    └── numerical_results/     # PDGM training logs & loss curves
```

## Reproducibility

### Setup

```bash
# Install uv (if not already installed)
pip install uv

# Install all dependencies
uv sync
```

### Part (a) — Monte Carlo

```bash
uv run python question_a.py
```

### Part (b) — Deep PPDE

```bash
# Lookback options (already run; results in numerical_results/)
cd Deep-PPDE
python ppde_BlackScholes_lookback.py --option_type put
python ppde_BlackScholes_lookback.py --option_type call

# Barrier options
python ppde_BlackScholes_barrier.py --option_type call --barrier_type down-out
python ppde_BlackScholes_barrier.py --option_type put  --barrier_type up-out
```

### Part (c) — PDGM

```bash
# Lookback options
uv run python PDGM-Geometric_Asian/pdgm_lookback.py --option_type put  --epochs 2000
uv run python PDGM-Geometric_Asian/pdgm_lookback.py --option_type call --epochs 2000

# Barrier options
uv run python PDGM-Geometric_Asian/pdgm_barrier.py --option_type call --barrier_type down-out --epochs 3000
uv run python PDGM-Geometric_Asian/pdgm_barrier.py --option_type put  --barrier_type up-out   --epochs 3000
```

## Key Results

| Option | Method | Price | Closed-Form | Error |
|--------|--------|-------|-------------|-------|
| Lookback PUT  | MC (Part a)       | 13.43  | 14.29  | -6.0%  |
| Lookback CALL | MC (Part a)       | 16.59  | 17.22  | -3.7%  |
| Lookback PUT  | Deep PPDE (Part b)| 0.2160 | 0.2330 | -7.3%  |
| Lookback CALL | Deep PPDE (Part b)| 0.2289 | 0.2379 | -3.8%  |
| Lookback PUT  | PDGM (Part c)     | 0.1266 | 0.1429 | -11.4% |
| Lookback CALL | PDGM (Part c)     | 0.1642 | 0.1722 | -4.6%  |
| Down-out CALL | PDGM (Part c)     | 0.0914 | 0.0867 | +5.4%  |
| Up-out PUT    | PDGM (Part c)     | 0.0398 | 0.0408 | -2.5%  |

## Dependencies

- Python 3.8
- PyTorch 2.x
- NumPy, SciPy, Matplotlib
- ReportLab 4.0.9
- [signatory](https://github.com/patrick-kidger/signatory) (for Deep PPDE log-signature)

See `pyproject.toml` for full pinned versions.

## References

1. M. Sabate-Vidales, D. Siska, L. Szpruch. *Solving path dependent PDEs with LSTM networks and path signatures.* arXiv:2011.10630, 2020.
2. Y.F. Saporito and Z. Zhang. *Path-dependent deep Galerkin method.* SIAM J. Financial Math., 12(3):912-940, 2021.
3. M. Broadie, P. Glasserman, S. Kou. *A continuity correction for discrete barrier options.* Mathematical Finance, 7(4):325-349, 1997.
