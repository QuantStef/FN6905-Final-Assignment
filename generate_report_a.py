"""
Generate PDF report for Part (a) of FN6905 Final Assignment.
Monte Carlo pricing of barrier and lookback options with closed-form comparison.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import os

OUTPUT_PATH = os.path.join(
    r"C:\Laptop Data\Msc Financial Engineering NTU\Main & Elective Courses"
    r"\FN6905 Exotic Options & Structured Products\Final Assignment",
    "report_part_a.pdf"
)

# ── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name='MainTitle', parent=styles['Title'],
    fontSize=16, leading=20, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name='Subtitle', parent=styles['Normal'],
    fontSize=10, leading=14, alignment=TA_CENTER, spaceAfter=14,
    textColor=HexColor('#555555'),
))
styles.add(ParagraphStyle(
    name='SectionHead', parent=styles['Heading1'],
    fontSize=13, leading=16, spaceBefore=14, spaceAfter=5,
    textColor=HexColor('#1a1a2e'),
))
styles.add(ParagraphStyle(
    name='SubHead', parent=styles['Heading2'],
    fontSize=11, leading=14, spaceBefore=8, spaceAfter=4,
    textColor=HexColor('#16213e'),
))
styles.add(ParagraphStyle(
    name='Body', parent=styles['Normal'],
    fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name='CodeBlock', parent=styles['Normal'],
    fontName='Courier', fontSize=8, leading=11, spaceAfter=4,
    leftIndent=12, backColor=HexColor('#f5f5f5'),
))
styles.add(ParagraphStyle(
    name='Caption', parent=styles['Normal'],
    fontSize=9, leading=12, alignment=TA_CENTER, spaceAfter=10,
    textColor=HexColor('#444444'), fontName='Helvetica-Oblique',
))

# ── Helpers ──────────────────────────────────────────────────────────────────
HEADER_BG = HexColor('#1a1a2e')
ALT_ROW   = HexColor('#f0f0f8')
POS_COLOR = HexColor('#c8e6c9')   # light green  — MC above CF
NEG_COLOR = HexColor('#ffcdd2')   # light red    — MC below CF

def make_table(data, col_widths=None, highlight_col=None):
    """Create a styled table with alternating rows."""
    n_rows = len(data)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ('BACKGROUND',   (0, 0), (-1, 0), HEADER_BG),
        ('TEXTCOLOR',    (0, 0), (-1, 0), white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, 0), 8.5),
        ('FONTNAME',     (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',     (0, 1), (-1, -1), 8.5),
        ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID',         (0, 0), (-1, -1), 0.4, HexColor('#cccccc')),
        ('TOPPADDING',   (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 3),
    ]
    for i in range(1, n_rows):
        if i % 2 == 0:
            cmds.append(('BACKGROUND', (0, i), (-1, i), ALT_ROW))
    t.setStyle(TableStyle(cmds))
    return t

# ── Document ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT_PATH, pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)
story = []

# ── Title ─────────────────────────────────────────────────────────────────────
story.append(Paragraph(
    "Part (a): Monte Carlo Pricing of Barrier and Lookback Options",
    styles['MainTitle']
))
story.append(Paragraph(
    "FN6905 Exotic Options &amp; Structured Products &mdash; Final Assignment",
    styles['Subtitle']
))

# ═════════════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("1. Introduction", styles['SectionHead']))
story.append(Paragraph(
    "This section implements Monte Carlo (MC) pricers for barrier options (Table 8.1) "
    "and floating lookback options (Propositions 9.1 and 9.5) under the Black-Scholes "
    "model. All MC prices are compared against the corresponding closed-form analytical "
    "expressions. The code is available in <font face='Courier' size=9>question_a.py</font>.",
    styles['Body']
))

# ═════════════════════════════════════════════════════════════════════════════
# 2. METHOD
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("2. Monte Carlo Method", styles['SectionHead']))

story.append(Paragraph("2.1 Path Simulation", styles['SubHead']))
story.append(Paragraph(
    "Stock price paths are simulated under the risk-neutral measure using the "
    "Euler-Maruyama discretisation of Geometric Brownian Motion (GBM):",
    styles['Body']
))
story.append(Paragraph(
    "S(t + dt) = S(t) exp[(r - sigma^2/2) dt  +  sigma sqrt(dt) Z]",
    styles['CodeBlock']
))
story.append(Paragraph(
    "where Z ~ N(0,1) is an independent standard normal draw at each step. "
    "Using the log-Euler scheme (exact for GBM) avoids any discretisation error "
    "in the path dynamics themselves; only the barrier/extremum monitoring is discrete.",
    styles['Body']
))

story.append(Paragraph("2.2 Barrier Option Payoffs", styles['SubHead']))
story.append(Paragraph(
    "For each simulated path, the running maximum and minimum are tracked. "
    "The barrier condition is checked at every time step:",
    styles['Body']
))

payoff_data = [
    ['Type', 'Condition', 'Payoff'],
    ['Down-and-out call', 'min(S) > B', 'max(S_T - K, 0)'],
    ['Down-and-in call',  'min(S) <= B', 'max(S_T - K, 0)'],
    ['Up-and-out call',   'max(S) < B', 'max(S_T - K, 0)'],
    ['Up-and-in call',    'max(S) >= B', 'max(S_T - K, 0)'],
    ['Down-and-out put',  'min(S) > B', 'max(K - S_T, 0)'],
    ['Down-and-in put',   'min(S) <= B', 'max(K - S_T, 0)'],
    ['Up-and-out put',    'max(S) < B', 'max(K - S_T, 0)'],
    ['Up-and-in put',     'max(S) >= B', 'max(K - S_T, 0)'],
]
story.append(make_table(payoff_data, col_widths=[4*cm, 3.5*cm, 5*cm]))
story.append(Paragraph(
    "Table 1: Barrier option payoff conditions. Knock-out options pay only if the "
    "barrier is never crossed; knock-in options pay only if it is crossed.",
    styles['Caption']
))

story.append(Paragraph("2.3 Lookback Option Payoffs", styles['SubHead']))
story.append(Paragraph(
    "Floating lookback options depend on the running extremum of the path:",
    styles['Body']
))
lb_payoff_data = [
    ['Option', 'Payoff', 'Closed-form'],
    ['Floating PUT  (Prop 9.1)', 'max(S) - S_T', 'Proposition 9.1'],
    ['Floating CALL (Prop 9.5)', 'S_T - min(S)', 'Proposition 9.5'],
]
story.append(make_table(lb_payoff_data, col_widths=[4.5*cm, 3.5*cm, 4*cm]))
story.append(Paragraph(
    "Table 2: Lookback option payoff definitions.",
    styles['Caption']
))

story.append(Paragraph("2.4 Closed-Form Formulas", styles['SubHead']))
story.append(Paragraph(
    "Barrier option closed-form prices use the Reiner-Rubinstein formulas "
    "(equivalent to Table 8.1), expressed in terms of the delta function from "
    "equation (8.2.2):",
    styles['Body']
))
story.append(Paragraph(
    "delta_+/-^tau(s) = [log(s) + (r +/- sigma^2/2) tau] / (sigma sqrt(tau))",
    styles['CodeBlock']
))
story.append(Paragraph(
    "In-out parity (Equations 8.1.1-8.1.4) is used to derive knock-in prices "
    "from knock-out prices: C_in + C_out = C_BS (vanilla). "
    "Lookback prices use Propositions 9.1 and 9.5, evaluated at t = 0 "
    "where M<sub>0</sub><super>0</super> = m<sub>0</sub><super>0</super> = S<sub>0</sub>.",
    styles['Body']
))

story.append(Paragraph("2.5 Parameters", styles['SubHead']))
params_data = [
    ['Parameter', 'Value', 'Description'],
    ['S_0', '100', 'Initial stock price'],
    ['K',   '100', 'Strike price (at-the-money)'],
    ['B',   '90 / 110', 'Barrier level (down / up)'],
    ['r',   '0.05', 'Risk-free interest rate'],
    ['sigma','0.20', 'Volatility'],
    ['T',   '1.0', 'Time to maturity (years)'],
    ['N',   '252', 'Time steps (daily monitoring)'],
    ['M',   '100,000', 'Number of Monte Carlo paths'],
]
story.append(make_table(params_data, col_widths=[2.5*cm, 3*cm, 7*cm]))
story.append(Paragraph(
    "Table 3: Simulation parameters. The 95% confidence interval for MC prices "
    "is price +/- 2 x SE, where SE = e^{-rT} x std(payoff) / sqrt(M).",
    styles['Caption']
))

# ═════════════════════════════════════════════════════════════════════════════
# 3. RESULTS — BARRIER OPTIONS
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("3. Results", styles['SectionHead']))

story.append(Paragraph("3.1 Barrier Options (Table 8.1)", styles['SubHead']))
story.append(Paragraph(
    "Table 4 compares MC and closed-form prices for all eight non-degenerate "
    "barrier option types. Four degenerate cases (up-and-out call with B &lt;= K, "
    "and down-and-out put with B &gt;= K) have price identically zero and are "
    "excluded. The ±2 SE column gives the 95% confidence interval half-width.",
    styles['Body']
))

barrier_data = [
    ['Type', 'B', 'Cond.', 'MC Price', '±2 SE', 'Closed-Form', 'Diff', 'Diff %'],
    ['call down-out', '90',  'B≤K', '8.8635', '0.0923', '8.6655', '+0.198', '+2.3%'],
    ['call down-in',  '90',  'B≤K', '1.5359', '0.0338', '1.7851', '-0.249', '-14.0%'],
    ['call up-out',   '110', 'B≥K', '0.1564', '0.0056', '0.1186', '+0.038', '+31.9%'],
    ['call up-in',    '110', 'B≥K', '10.2429','0.0932', '10.3320','-0.089', '-0.9%'],
    ['put down-out',  '90',  'B≤K', '0.1932', '0.0062', '0.1512', '+0.042', '+27.8%'],
    ['put down-in',   '90',  'B≤K', '5.3586', '0.0552', '5.4223', '-0.064', '-1.2%'],
    ['put up-out',    '110', 'B≥K', '4.3636', '0.0529', '4.0796', '+0.284', '+7.0%'],
    ['put up-in',     '110', 'B≥K', '1.1882', '0.0248', '1.4939', '-0.306', '-20.5%'],
]
story.append(make_table(barrier_data,
    col_widths=[2.3*cm, 1*cm, 1*cm, 1.7*cm, 1.3*cm, 2.1*cm, 1.4*cm, 1.4*cm]))
story.append(Paragraph(
    "Table 4: Barrier option MC vs closed-form prices. S_0=100, K=100, r=0.05, "
    "sigma=0.20, T=1, N=252 steps, M=100,000 paths.",
    styles['Caption']
))

# ═════════════════════════════════════════════════════════════════════════════
# 3.2 LOOKBACK OPTIONS
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("3.2 Lookback Options (Propositions 9.1 and 9.5)", styles['SubHead']))
story.append(Paragraph(
    "Table 5 compares MC and closed-form prices for the two floating lookback options.",
    styles['Body']
))

lookback_data = [
    ['Option', 'Proposition', 'MC Price', '±2 SE', 'Closed-Form', 'Diff', 'Diff %'],
    ['Floating PUT  (max(S) - S_T)', '9.1', '13.4291', '0.0623', '14.2906', '-0.862', '-6.0%'],
    ['Floating CALL (S_T - min(S))', '9.5', '16.5870', '0.0916', '17.2168', '-0.630', '-3.7%'],
]
story.append(make_table(lookback_data,
    col_widths=[4.2*cm, 1.8*cm, 1.7*cm, 1.2*cm, 2.1*cm, 1.2*cm, 1.2*cm]))
story.append(Paragraph(
    "Table 5: Lookback option MC vs closed-form prices. S_0=100, r=0.05, "
    "sigma=0.20, T=1, N=252 steps, M=100,000 paths.",
    styles['Caption']
))

# ═════════════════════════════════════════════════════════════════════════════
# 3.3 PARITY
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("3.3 In-Out Parity Verification", styles['SubHead']))
story.append(Paragraph(
    "As a correctness check, the in-out parity relations (Equations 8.1.1-8.1.4) "
    "state that C_out + C_in = C_BS. Table 6 verifies this numerically.",
    styles['Body']
))

parity_data = [
    ['Parity Check', 'Knock-Out (MC)', 'Knock-In (MC)', 'Sum', 'BS Vanilla', 'Error'],
    ['Down call (B=90)', '8.8635', '1.5359', '10.3994', '10.4506', '0.49%'],
    ['Up put   (B=110)', '4.3636', '1.1882', '5.5518',  '5.5735',  '0.39%'],
]
story.append(make_table(parity_data,
    col_widths=[2.8*cm, 2.2*cm, 2.2*cm, 1.5*cm, 2.1*cm, 1.4*cm]))
story.append(Paragraph(
    "Table 6: In-out parity check. Errors below 0.5% confirm the payoff "
    "implementations are correct.",
    styles['Caption']
))

# ═════════════════════════════════════════════════════════════════════════════
# 4. DISCUSSION
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("4. Discussion", styles['SectionHead']))

story.append(Paragraph("4.1 Discrete Monitoring Bias", styles['SubHead']))
story.append(Paragraph(
    "The closed-form formulas assume the barrier and extremum are monitored "
    "<b>continuously</b>, while the MC simulation checks only at N = 252 discrete "
    "daily steps. This introduces a systematic bias known as the <b>discrete "
    "monitoring bias</b> (Broadie, Glasserman, and Kou, 1997). The bias can be "
    "approximated by a continuity correction that shifts the barrier by:",
    styles['Body']
))
story.append(Paragraph(
    "continuity correction = 0.5826 x sigma x sqrt(T/N) = 0.5826 x 0.20 x sqrt(1/252) = 0.00734",
    styles['CodeBlock']
))
story.append(Paragraph(
    "For knock-out options, discrete monitoring means the barrier may be breached "
    "between observation points without being recorded, causing the MC price to be "
    "<b>higher</b> than the continuous-monitoring closed-form (fewer paths appear to "
    "knock out). For knock-in options, the opposite holds: the MC price is "
    "<b>lower</b> than the closed-form.",
    styles['Body']
))
story.append(Paragraph(
    "This pattern is clearly visible in Table 4: all knock-out prices (down-out, up-out) "
    "have positive Diff, while all knock-in prices have negative Diff. "
    "The same direction applies to the lookback options in Table 5, where the MC "
    "running extremum is an underestimate of the continuous maximum (or overestimate "
    "of the continuous minimum), producing a lower lookback price.",
    styles['Body']
))

story.append(Paragraph("4.2 Magnitude of Errors", styles['SubHead']))
story.append(Paragraph(
    "The percentage errors in Table 4 range from 0.9% to 31.9%. Larger relative errors "
    "appear for options with small prices (e.g. up-out call at 0.1186 has a 31.9% error "
    "in absolute terms of only 0.038). This is because the continuity correction "
    "is a fixed absolute shift, which becomes large relative to a small option price. "
    "For options with larger prices (e.g. down-out call at 8.67), the relative error "
    "is only 2.3%. The parity check in Table 6 confirms errors below 0.5%, validating "
    "the implementations are internally consistent.",
    styles['Body']
))

story.append(Paragraph("4.3 Convergence and Variance Reduction", styles['SubHead']))
story.append(Paragraph(
    "With M = 100,000 paths, the 95% confidence intervals (±2 SE) are narrow relative "
    "to the option prices — typically less than 2% of the price. This is sufficient "
    "for comparison purposes. The bias from discrete monitoring is much larger than "
    "the MC sampling error, confirming that increasing M further would not close the "
    "gap to the closed-form; only increasing N (finer time grid) achieves this.",
    styles['Body']
))

# ═════════════════════════════════════════════════════════════════════════════
# 5. REPRODUCIBILITY
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("5. Reproducibility", styles['SectionHead']))
story.append(Paragraph(
    "All results in this section are produced by "
    "<font face='Courier' size=9>question_a.py</font>. "
    "To reproduce from a clean checkout:",
    styles['Body']
))
for cmd in ["uv python install 3.8", "uv sync", "uv run python question_a.py"]:
    story.append(Paragraph(cmd, styles['CodeBlock']))
story.append(Paragraph(
    "The script prints the barrier option table, lookback table, parity checks, "
    "and the discretisation bias note. A fixed random seed (seed=42) ensures "
    "identical output on every run.",
    styles['Body']
))

# ═════════════════════════════════════════════════════════════════════════════
# 6. REFERENCES
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("References", styles['SectionHead']))
story.append(Paragraph(
    "[1] N. Privault, \"FN6905 Exotic Options &amp; Structured Products,\" "
    "Lecture Notes, NTU, February 2026.",
    styles['Body']
))
story.append(Paragraph(
    "[2] M. Broadie, P. Glasserman, and S. Kou, "
    "\"A continuity correction for discrete barrier options,\" "
    "<i>Mathematical Finance</i>, 7(4):325-349, 1997.",
    styles['Body']
))
story.append(Paragraph(
    "[3] R. C. Reiner and M. Rubinstein, "
    "\"Breaking down the barriers,\" <i>Risk</i>, 4(8):28-35, 1991.",
    styles['Body']
))

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"Report saved to: {OUTPUT_PATH}")
