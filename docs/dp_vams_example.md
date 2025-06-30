Below is a **step-by-step recipe** for turning a plain *Volatility-Adjusted Momentum Score* (VAMS) into a **“downside-penalised VAMS” (dp-VAMS)** that keeps the speed and cross-sectional robustness of momentum signals while explicitly discouraging assets whose upside has come with large left-tail moves.

---

## 1 Define the building blocks

| Symbol | Definition | Notes |
|--------|------------|-------|
| \(P_t\) | close price at time *t* | daily data assumed |
| \(r_t = \ln(P_t/P_{t-1})\) | log-return | additive, robust to gaps |
| \(N\) | look-back window (e.g. 63, 126, 252) | responsiveness vs stability |
| \(R_N = P_t/P_{t-N}-1\) | **terminal cumulative return** over *N* days | classic momentum numerator |
| \(\sigma_N\) | **total volatility** (std. dev. of \(r_t\)) | used by vanilla VAMS |
| \(r_t^{-} = \min(r_t,0)\) | negative (downside) component | truncates positives |
| \(\sigma_N^{-}\) | **downside volatility** (std. dev. of \(r_t^{-}\)) | semi-deviation |

---

## 2 Choose a penalisation scheme

### 2.1 Pure replacement (“Sortino-like VAMS”)
\[
\text{dp-VAMS}_{\text{pure}} = \frac{R_N}{\sigma_N^{-}}
\]

*Pros:* simplest, intuitive  
*Cons:* ignores upside noise; can under-penalise fat-upside assets with small drops.

---

### 2.2 Hybrid denominator (balance total vs downside)
\[
\text{Denom} = (1-\alpha)\,\sigma_N + \alpha\,\sigma_N^{-}, \qquad \alpha\in[0,1]
\]
\[
\text{dp-VAMS}_{\text{hyb}} = \frac{R_N}{\text{Denom}}
\]

*Pros:* dial harshness via \(\alpha\)  
*Good default:* \(\alpha = 0.5\)

---

### 2.3 Multiplicative penalty factor
1. **Vanilla VAMS**  
   \( \text{VAMS} = R_N / \sigma_N\)
2. **Penalty**  
   \( \text{Penalty} = \bigl(1 + \kappa \tfrac{\sigma_N^{-}}{\sigma_N}\bigr)^{-1}\)
3. **Combine**  
   \( \text{dp-VAMS}_{\text{mult}} = \text{VAMS} \times \text{Penalty}\)

*Pros:* keeps VAMS scale; penalty only kicks in when downside share is large  
*Typical \(\kappa\):* 1–3

---

## 3 Make the score robust

| Technique | Why it helps | How to do it |
|-----------|--------------|--------------|
| **Winsorise returns** | tame outliers | clip \(r_t\) to e.g. ±5σ before σ-calcs |
| **Exponential weights** | recency bias | EWMA for σ and downside σ |
| **Ranking / z-scoring** | comparability | convert dp-VAMS to percentile rank |
| **Signal smoothing** | reduce turnover | 3–5-day MA of scores or threshold carry-over |
| **Look-back ensemble** | balance horizons | average dp-VAMS for N = 63, 126, 252 |

---

## 4 Full algorithm (monthly rebalance)

```text
For each asset on rebalance date t:

1. Pull last N daily closes
2. Compute log returns r
3. R_N  = P_t / P_{t-N} - 1
4. σ_N       = std(r)
5. σ_N_minus = std(min(r,0))
6. dpVAMS = R_N / ( (1-α)*σ_N + α*σ_N_minus )   # α configurable
7. Optional:
   - winsorise r before std calcs
   - rank dpVAMS across universe
   - apply top-K or top-quintile filter

6 Parameter suggestions
Parameter	Typical values	Impact
lookback	63 / 126 / 252	shorter = faster, noisier
alpha	0.3 – 0.7	↑α → harsher downside penalty
winsor_clip	4 – 6σ	lower → more robust, less responsive
Portfolio size	5 – 20 longs	diversification vs concentration

Back-test grid-searching alpha and lookback; often a middle alpha (~0.5) gives best Information Ratio: it retains momentum yet cuts the worst left-tail names.

Key takeaway
dp-VAMS = momentum strength ÷ downside risk, tuned to keep fast ranking ability but explicitly punish left-tail noise.
Balance speed and robustness with lookback, alpha, and simple cross-sectional ranking tricks.

Happy coding & trading—let me know if you need a full back-test template or further tweaks!