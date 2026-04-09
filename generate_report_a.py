"""
Generate PDF report for Part (a) of FN6905 Final Assignment.
Monte Carlo pricing of barrier and lookback options with closed-form comparison.
"""
import io
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, KeepTogether,
)

OUTPUT_PATH = os.path.join(
    r"C:\Laptop Data\Msc Financial Engineering NTU\Main & Elective Courses"
    r"\FN6905 Exotic Options & Structured Products\Final Assignment",
    "report_part_a.pdf"
)

# ── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='MainTitle', parent=styles['Title'],
    fontSize=16, leading=20, spaceAfter=4))
styles.add(ParagraphStyle(name='Subtitle', parent=styles['Normal'],
    fontSize=10, leading=14, alignment=TA_CENTER, spaceAfter=14,
    textColor=HexColor('#555555')))
styles.add(ParagraphStyle(name='SectionHead', parent=styles['Heading1'],
    fontSize=13, leading=16, spaceBefore=14, spaceAfter=5,
    textColor=HexColor('#1a1a2e')))
styles.add(ParagraphStyle(name='SubHead', parent=styles['Heading2'],
    fontSize=11, leading=14, spaceBefore=8, spaceAfter=4,
    textColor=HexColor('#16213e')))
styles.add(ParagraphStyle(name='Body', parent=styles['Normal'],
    fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle(name='Formula', parent=styles['Normal'],
    fontName='Courier', fontSize=9, leading=13, spaceAfter=2,
    leftIndent=20))
styles.add(ParagraphStyle(name='CodeBlock', parent=styles['Normal'],
    fontName='Courier', fontSize=8, leading=11, spaceAfter=4,
    leftIndent=12, backColor=HexColor('#f5f5f5')))
styles.add(ParagraphStyle(name='Caption', parent=styles['Normal'],
    fontSize=9, leading=12, alignment=TA_CENTER, spaceAfter=10,
    textColor=HexColor('#444444'), fontName='Helvetica-Oblique'))
styles.add(ParagraphStyle(name='TableNote', parent=styles['Normal'],
    fontSize=8, leading=11, spaceAfter=4,
    textColor=HexColor('#666666'), fontName='Helvetica-Oblique'))

# ── Helpers ──────────────────────────────────────────────────────────────────
HEADER_BG = HexColor('#1a1a2e')
ALT_ROW   = HexColor('#f0f0f8')

def make_table(data, col_widths=None):
    n = len(data)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ('BACKGROUND',   (0,0),(-1,0), HEADER_BG),
        ('TEXTCOLOR',    (0,0),(-1,0), white),
        ('FONTNAME',     (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0,0),(-1,0), 8.5),
        ('FONTNAME',     (0,1),(-1,-1),'Helvetica'),
        ('FONTSIZE',     (0,1),(-1,-1), 8.5),
        ('ALIGN',        (0,0),(-1,-1),'CENTER'),
        ('VALIGN',       (0,0),(-1,-1),'MIDDLE'),
        ('GRID',         (0,0),(-1,-1), 0.4, HexColor('#cccccc')),
        ('TOPPADDING',   (0,0),(-1,-1), 3),
        ('BOTTOMPADDING',(0,0),(-1,-1), 3),
    ]
    for i in range(1, n):
        if i % 2 == 0:
            cmds.append(('BACKGROUND',(0,i),(-1,i),ALT_ROW))
    t.setStyle(TableStyle(cmds))
    return t

def fig_to_rl(fig, width=14.5*cm, height=7.5*cm):
    """Convert matplotlib Figure to a ReportLab Image (in-memory PNG)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return RLImage(buf, width=width, height=height)

# ── Chart 1: Barrier options bar chart ───────────────────────────────────────
barrier_labels = ['DO call','DI call','UO call','UI call',
                  'DO put', 'DI put', 'UO put', 'UI put']
mc_vals  = [8.8635, 1.5359, 0.1564, 10.2429, 0.1932, 5.3586, 4.3636, 1.1882]
cf_vals  = [8.6655, 1.7851, 0.1186, 10.3320, 0.1512, 5.4223, 4.0796, 1.4939]

fig1, ax1 = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(barrier_labels))
w = 0.35
ax1.bar(x - w/2, mc_vals,  w, label='Monte Carlo', color='#1976D2', alpha=0.88)
ax1.bar(x + w/2, cf_vals,  w, label='Closed-Form', color='#E53935', alpha=0.88)
ax1.set_xticks(x)
ax1.set_xticklabels(barrier_labels, rotation=25, ha='right', fontsize=9)
ax1.set_ylabel('Option Price (S\u2080=100)')
ax1.set_title('Figure 1. Barrier Options: Monte Carlo vs Closed-Form\n'
              '(S\u2080=100, K=100, r=0.05, \u03c3=0.20, T=1, N=252, M=100,000)',
              fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3, linewidth=0.6)
fig1.tight_layout()
chart_barrier = fig_to_rl(fig1, width=14.5*cm, height=7*cm)

# ── Chart 2: Lookback options bar chart ──────────────────────────────────────
lb_labels = ['Floating PUT\n(max(S)\u2212S\u1d40)', 'Floating CALL\n(S\u1d40\u2212min(S))']
lb_mc = [13.4291, 16.5870]
lb_cf = [14.2906, 17.2168]

fig2, ax2 = plt.subplots(figsize=(5.5, 4))
x2 = np.arange(2)
ax2.bar(x2 - 0.2, lb_mc, 0.38, label='Monte Carlo', color='#1976D2', alpha=0.88)
ax2.bar(x2 + 0.2, lb_cf, 0.38, label='Closed-Form', color='#E53935', alpha=0.88)
ax2.set_xticks(x2)
ax2.set_xticklabels(lb_labels, fontsize=9)
ax2.set_ylabel('Option Price (S\u2080=100)')
ax2.set_title('Figure 2. Lookback Options\n(Props 9.1 & 9.5)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3, linewidth=0.6)
for bars, vals in [(ax2.containers[0], lb_mc), (ax2.containers[1], lb_cf)]:
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=8)
fig2.tight_layout()
chart_lookback = fig_to_rl(fig2, width=8*cm, height=6*cm)

# ── Document ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(OUTPUT_PATH, pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2*cm, bottomMargin=2*cm)
story = []

# ── Title ─────────────────────────────────────────────────────────────────────
story.append(Paragraph(
    "Part (a): Monte Carlo Pricing of Barrier and Lookback Options",
    styles['MainTitle']))
story.append(Paragraph(
    "FN6905 Exotic Options &amp; Structured Products &mdash; Final Assignment",
    styles['Subtitle']))

# ═════════════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("1. Introduction", styles['SectionHead']))
story.append(Paragraph(
    "This section implements Monte Carlo (MC) pricers for barrier options (Table 8.1) "
    "and floating lookback options (Propositions 9.1 and 9.5) under the Black-Scholes "
    "model. All MC prices are compared against the corresponding closed-form analytical "
    "expressions. The code is available in <font face='Courier' size=9>question_a.py</font>.",
    styles['Body']))

# ═════════════════════════════════════════════════════════════════════════════
# 2. METHOD
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("2. Monte Carlo Method", styles['SectionHead']))

story.append(Paragraph("2.1 Path Simulation", styles['SubHead']))
story.append(Paragraph(
    "Stock price paths are simulated under the risk-neutral measure using the "
    "exact log-Euler discretisation of Geometric Brownian Motion (GBM):",
    styles['Body']))
story.append(Paragraph(
    "S(t+dt) = S(t) \u00b7 exp[ (r \u2212 \u03c3\u00b2/2)\u00b7dt  +  \u03c3\u00b7\u221adt\u00b7Z ]",
    styles['Formula']))
story.append(Paragraph(
    "where Z\u223cN(0,1) independently at each step. This is the exact solution to the "
    "GBM SDE; only the discrete-monitoring of barriers and extrema introduces bias.",
    styles['Body']))

story.append(Paragraph("2.2 Barrier Option Payoffs", styles['SubHead']))
payoff_data = [
    ['Type', 'Condition', 'Payoff'],
    ['Down-and-out call', 'min(S) > B',  'max(S\u1d40\u2212K, 0)'],
    ['Down-and-in call',  'min(S) \u2264 B', 'max(S\u1d40\u2212K, 0)'],
    ['Up-and-out call',   'max(S) < B',  'max(S\u1d40\u2212K, 0)'],
    ['Up-and-in call',    'max(S) \u2265 B', 'max(S\u1d40\u2212K, 0)'],
    ['Down-and-out put',  'min(S) > B',  'max(K\u2212S\u1d40, 0)'],
    ['Down-and-in put',   'min(S) \u2264 B', 'max(K\u2212S\u1d40, 0)'],
    ['Up-and-out put',    'max(S) < B',  'max(K\u2212S\u1d40, 0)'],
    ['Up-and-in put',     'max(S) \u2265 B', 'max(K\u2212S\u1d40, 0)'],
]
story.append(make_table(payoff_data, col_widths=[3.8*cm, 3.2*cm, 4.5*cm]))
story.append(Paragraph("Table 1. Barrier option payoff conditions.", styles['Caption']))

story.append(Paragraph("2.3 Lookback Option Payoffs", styles['SubHead']))
lb_payoff = [
    ['Option', 'Payoff at T', 'Closed-form'],
    ['Floating PUT  (Prop 9.1)', 'max(S\u2080,...,S\u1d40) \u2212 S\u1d40', 'Proposition 9.1'],
    ['Floating CALL (Prop 9.5)', 'S\u1d40 \u2212 min(S\u2080,...,S\u1d40)', 'Proposition 9.5'],
]
story.append(make_table(lb_payoff, col_widths=[4.5*cm, 4*cm, 3.5*cm]))
story.append(Paragraph("Table 2. Lookback option payoff definitions.", styles['Caption']))

# ── 2.4 Closed-Form Formulas ─────────────────────────────────────────────────
story.append(Paragraph("2.4 Closed-Form Expressions", styles['SubHead']))

story.append(Paragraph(
    "<b>Delta function (eq. 8.2.2).</b> "
    "All barrier and lookback formulas are expressed in terms of:",
    styles['Body']))
story.append(Paragraph(
    "\u03b4\u207a\u1d40(z) = [ ln(z) + (r + \u03c3\u00b2/2)\u00b7\u03c4 ] / (\u03c3\u221a\u03c4)",
    styles['Formula']))
story.append(Paragraph(
    "\u03b4\u207b\u1d40(z) = [ ln(z) + (r \u2212 \u03c3\u00b2/2)\u00b7\u03c4 ] / (\u03c3\u221a\u03c4)  =  \u03b4\u207a\u1d40(z) \u2212 \u03c3\u221a\u03c4",
    styles['Formula']))
story.append(Spacer(1, 4))

story.append(Paragraph(
    "<b>Barrier options \u2014 Reiner-Rubinstein (Table 8.1).</b> "
    "Define \u03bc = (r \u2212 \u03c3\u00b2/2)/\u03c3\u00b2. The four building blocks are:",
    styles['Body']))
story.append(Paragraph(
    "A = \u03b7[S\u00b7\u03a6(\u03b7\u03b4\u207a\u1d40(S/K)) \u2212 Ke\u207b\u02b3\u1d40\u00b7\u03a6(\u03b7\u03b4\u207b\u1d40(S/K))]",
    styles['Formula']))
story.append(Paragraph(
    "B = \u03b7[S\u00b7\u03a6(\u03b7\u03b4\u207a\u1d40(S/B)) \u2212 Ke\u207b\u02b3\u1d40\u00b7\u03a6(\u03b7\u03b4\u207b\u1d40(S/B))]",
    styles['Formula']))
story.append(Paragraph(
    "C = \u03b7[(B/S)\u207b\u00b2\u207f\u00b7S\u00b7(B/S)\u00b2\u207f\u00b7\u03a6(\u03c6\u03b4\u207a\u1d40(B\u00b2/SK)) \u2212 Ke\u207b\u02b3\u1d40\u00b7(B/S)\u00b2\u207f\u00b7\u03a6(\u03c6\u03b4\u207b\u1d40(B\u00b2/SK))]",
    styles['Formula']))
story.append(Paragraph(
    "D = \u03b7[(B/S)\u207b\u00b2\u207f\u00b7S\u00b7(B/S)\u00b2\u207f\u00b7\u03a6(\u03c6\u03b4\u207a\u1d40(B/S))  \u2212 Ke\u207b\u02b3\u1d40\u00b7(B/S)\u00b2\u207f\u00b7\u03a6(\u03c6\u03b4\u207b\u1d40(B/S))]",
    styles['Formula']))
story.append(Paragraph(
    "where \u03b7=+1 for calls, \u03b7=\u22121 for puts; \u03c6=+1 for down-barriers, \u03c6=\u22121 for up-barriers. "
    "Selected prices: down-out call (B\u2264K): A\u2212C; up-out put (B\u2265K): B\u2212D; "
    "knock-in prices via parity: C\u1d62\u2099 = C\u1d65\u1d43\u1d3a\u1d35\u1d38\u1d38\u1d39\u1d43 \u2212 C\u1d52\u1d58\u1d57.",
    styles['Body']))
story.append(Spacer(1, 4))

story.append(Paragraph(
    "<b>Proposition 9.1 \u2014 Floating Lookback PUT</b> "
    "(payoff = M\u1d40 \u2212 S\u1d40, where M\u1d40 = max\u2080\u2264\u209c\u2264\u1d40 S\u209c). "
    "At t=0 with M\u2080 = S\u2080:",
    styles['Body']))
story.append(Paragraph(
    "P = M\u2080\u00b7e\u207b\u02b3\u1d40\u00b7\u03a6(\u2212\u03b4\u207b\u1d40(S\u2080/M\u2080))",
    styles['Formula']))
story.append(Paragraph(
    "  + S\u2080\u00b7(1 + \u03c3\u00b2/2r)\u00b7\u03a6(\u03b4\u207a\u1d40(S\u2080/M\u2080))",
    styles['Formula']))
story.append(Paragraph(
    "  \u2212 S\u2080\u00b7e\u207b\u02b3\u1d40\u00b7(\u03c3\u00b2/2r)\u00b7(M\u2080/S\u2080)\u207b\u00b2\u02b3/\u03c3\u00b2\u00b7\u03a6(\u2212\u03b4\u207b\u1d40(M\u2080/S\u2080))",
    styles['Formula']))
story.append(Paragraph("  \u2212 S\u2080", styles['Formula']))
story.append(Spacer(1, 4))

story.append(Paragraph(
    "<b>Proposition 9.5 \u2014 Floating Lookback CALL</b> "
    "(payoff = S\u1d40 \u2212 m\u1d40, where m\u1d40 = min\u2080\u2264\u209c\u2264\u1d40 S\u209c). "
    "At t=0 with m\u2080 = S\u2080:",
    styles['Body']))
story.append(Paragraph(
    "C = S\u2080\u00b7\u03a6(\u03b4\u207a\u1d40(S\u2080/m\u2080))",
    styles['Formula']))
story.append(Paragraph(
    "  \u2212 m\u2080\u00b7e\u207b\u02b3\u1d40\u00b7\u03a6(\u03b4\u207b\u1d40(S\u2080/m\u2080))",
    styles['Formula']))
story.append(Paragraph(
    "  + e\u207b\u02b3\u1d40\u00b7S\u2080\u00b7(\u03c3\u00b2/2r)\u00b7(m\u2080/S\u2080)\u207b\u00b2\u02b3/\u03c3\u00b2\u00b7\u03a6(\u03b4\u207b\u1d40(m\u2080/S\u2080))",
    styles['Formula']))
story.append(Paragraph(
    "  \u2212 S\u2080\u00b7(\u03c3\u00b2/2r)\u00b7\u03a6(\u2212\u03b4\u207a\u1d40(S\u2080/m\u2080))",
    styles['Formula']))
story.append(Spacer(1, 4))

story.append(Paragraph("2.5 Parameters", styles['SubHead']))
params_data = [
    ['Parameter', 'Value', 'Description'],
    ['S\u2080', '100', 'Initial stock price'],
    ['K',    '100', 'Strike price (at-the-money)'],
    ['B',    '90 / 110', 'Barrier level (down / up)'],
    ['r',    '0.05', 'Risk-free interest rate'],
    ['\u03c3', '0.20', 'Volatility'],
    ['T',    '1.0', 'Time to maturity (years)'],
    ['N',    '252', 'Time steps (daily monitoring)'],
    ['M',    '100,000', 'Number of Monte Carlo paths'],
]
story.append(make_table(params_data, col_widths=[2.5*cm, 3*cm, 7*cm]))
story.append(Paragraph(
    "Table 3. Simulation parameters. The 95% confidence interval is price \u00b1 2\u00d7SE, "
    "where SE = e\u207b\u02b3\u1d40 \u00d7 std(payoff) / \u221aM.",
    styles['Caption']))

# ═════════════════════════════════════════════════════════════════════════════
# 3. RESULTS — BARRIER OPTIONS
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("3. Results", styles['SectionHead']))
story.append(Paragraph("3.1 Barrier Options (Table 8.1)", styles['SubHead']))
story.append(Paragraph(
    "Table 4 compares MC and closed-form prices for all eight non-degenerate "
    "barrier option types. Figure 1 provides a visual comparison.",
    styles['Body']))

barrier_data = [
    ['Type', 'B', 'Cond.', 'MC Price', '\u00b12 SE', 'Closed-Form', 'Diff', 'Diff %'],
    ['call down-out', '90',  'B\u2264K', '8.8635', '0.0923', '8.6655', '+0.198', '+2.3%'],
    ['call down-in',  '90',  'B\u2264K', '1.5359', '0.0338', '1.7851', '-0.249', '-14.0%'],
    ['call up-out',   '110', 'B\u2265K', '0.1564', '0.0056', '0.1186', '+0.038', '+31.9%'],
    ['call up-in',    '110', 'B\u2265K', '10.2429','0.0932', '10.3320','-0.089', '-0.9%'],
    ['put down-out',  '90',  'B\u2264K', '0.1932', '0.0062', '0.1512', '+0.042', '+27.8%'],
    ['put down-in',   '90',  'B\u2264K', '5.3586', '0.0552', '5.4223', '-0.064', '-1.2%'],
    ['put up-out',    '110', 'B\u2265K', '4.3636', '0.0529', '4.0796', '+0.284', '+7.0%'],
    ['put up-in',     '110', 'B\u2265K', '1.1882', '0.0248', '1.4939', '-0.306', '-20.5%'],
]
story.append(make_table(barrier_data,
    col_widths=[2.3*cm,1*cm,1*cm,1.7*cm,1.3*cm,2.1*cm,1.4*cm,1.4*cm]))
story.append(Paragraph(
    "Table 4. Barrier option MC vs closed-form prices. "
    "S\u2080=100, K=100, r=0.05, \u03c3=0.20, T=1, N=252 steps, M=100,000 paths.",
    styles['Caption']))

story.append(KeepTogether([
    chart_barrier,
    Spacer(1, 2),
]))

story.append(Paragraph("3.2 Lookback Options (Propositions 9.1 and 9.5)", styles['SubHead']))
lookback_data = [
    ['Option', 'Prop.', 'MC Price', '\u00b12 SE', 'Closed-Form', 'Diff', 'Diff %'],
    ['Floating PUT  (max(S)\u2212S\u1d40)', '9.1', '13.4291', '0.0623', '14.2906', '-0.862', '-6.0%'],
    ['Floating CALL (S\u1d40\u2212min(S))', '9.5', '16.5870', '0.0916', '17.2168', '-0.630', '-3.7%'],
]
story.append(make_table(lookback_data,
    col_widths=[4.2*cm,1.3*cm,1.7*cm,1.2*cm,2.1*cm,1.2*cm,1.2*cm]))
story.append(Paragraph(
    "Table 5. Lookback option MC vs closed-form prices. "
    "S\u2080=100, r=0.05, \u03c3=0.20, T=1, N=252 steps, M=100,000 paths.",
    styles['Caption']))

story.append(KeepTogether([chart_lookback, Spacer(1, 2)]))

story.append(Paragraph("3.3 In-Out Parity Verification", styles['SubHead']))
story.append(Paragraph(
    "The in-out parity C\u1d52\u1d58\u1d57 + C\u1d62\u2099 = C\u1d35\u1d38\u1d38\u1d39\u1d43 confirms the implementations are internally consistent.",
    styles['Body']))
parity_data = [
    ['Parity Check', 'Knock-Out (MC)', 'Knock-In (MC)', 'Sum', 'BS Vanilla', 'Error'],
    ['Down call (B=90)', '8.8635', '1.5359', '10.3994', '10.4506', '0.49%'],
    ['Up put   (B=110)', '4.3636', '1.1882', '5.5518',  '5.5735',  '0.39%'],
]
story.append(make_table(parity_data,
    col_widths=[2.8*cm,2.2*cm,2.2*cm,1.5*cm,2.1*cm,1.4*cm]))
story.append(Paragraph(
    "Table 6. In-out parity check. Errors below 0.5% confirm payoff correctness.",
    styles['Caption']))

# ═════════════════════════════════════════════════════════════════════════════
# 4. DISCUSSION
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("4. Discussion", styles['SectionHead']))
story.append(Paragraph("4.1 Discrete Monitoring Bias", styles['SubHead']))
story.append(Paragraph(
    "The closed-form formulas assume <b>continuous</b> barrier/extremum monitoring, "
    "while MC checks only at N=252 discrete daily steps. "
    "Broadie, Glasserman &amp; Kou (1997) show the effective barrier shift is:",
    styles['Body']))
story.append(Paragraph(
    "\u0394B \u2248 \u00b1 0.5826 \u00d7 \u03c3 \u00d7 \u221a(T/N)  =  \u00b1 0.5826 \u00d7 0.20 \u00d7 \u221a(1/252)  \u2248  \u00b1 0.00734",
    styles['Formula']))
story.append(Paragraph(
    "For knock-<b>out</b> options: discrete monitoring misses some crossings \u2192 "
    "fewer paths knock out \u2192 MC price <b>above</b> continuous CF. "
    "For knock-<b>in</b>: fewer paths trigger \u2192 MC price <b>below</b> CF. "
    "This pattern is visible in every row of Table 4: all knock-out Diff values are "
    "positive; all knock-in Diff values are negative. Figure 1 makes this asymmetry "
    "visually clear.",
    styles['Body']))

story.append(Paragraph("4.2 Magnitude of Errors", styles['SubHead']))
story.append(Paragraph(
    "Percentage errors range from 0.9% to 31.9%. Larger relative errors appear for "
    "options with small absolute prices (e.g. up-out call at 0.1186 has 31.9% error "
    "on an absolute difference of only 0.038). Since the continuity correction is a "
    "fixed absolute shift, it dominates when the option price is small. "
    "For high-priced options (down-out call at 8.67), relative error is just 2.3%. "
    "Parity errors below 0.5% (Table 6) confirm the implementations are correct.",
    styles['Body']))

story.append(Paragraph("4.3 Convergence", styles['SubHead']))
story.append(Paragraph(
    "With M=100,000 paths, confidence intervals (\u00b12 SE) are typically under 2% "
    "of the option price. The dominant error is discrete-monitoring bias, not "
    "sampling variance: increasing M further will not close the gap to CF. "
    "Only increasing N (finer time grid) achieves this.",
    styles['Body']))

# ═════════════════════════════════════════════════════════════════════════════
# 5. REPRODUCIBILITY
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("5. Reproducibility", styles['SectionHead']))
story.append(Paragraph(
    "All results are produced by "
    "<font face='Courier' size=9>question_a.py</font>. "
    "To reproduce from a clean checkout:",
    styles['Body']))
for cmd in ["uv python install 3.8", "uv sync", "uv run python question_a.py"]:
    story.append(Paragraph(cmd, styles['CodeBlock']))
story.append(Paragraph(
    "Fixed random seed (seed=42) ensures identical output on every run. "
    "Following the workflow-vs-script principles (Tidyverse, 2017), the script uses "
    "relative paths, no hardcoded directories, and runs cleanly after "
    "<font face='Courier' size=9>uv sync</font>.",
    styles['Body']))

# ═════════════════════════════════════════════════════════════════════════════
# 6. REFERENCES
# ═════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("References", styles['SectionHead']))
story.append(Paragraph(
    "[1] N. Privault, \"FN6905 Exotic Options &amp; Structured Products,\" "
    "Lecture Notes, NTU, February 2026.", styles['Body']))
story.append(Paragraph(
    "[2] M. Broadie, P. Glasserman, and S. Kou, "
    "\"A continuity correction for discrete barrier options,\" "
    "<i>Mathematical Finance</i>, 7(4):325-349, 1997.", styles['Body']))
story.append(Paragraph(
    "[3] E. Reiner and M. Rubinstein, "
    "\"Breaking down the barriers,\" <i>Risk</i>, 4(8):28-35, 1991.", styles['Body']))

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"Report saved to: {OUTPUT_PATH}")
