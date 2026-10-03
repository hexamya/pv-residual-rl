# Technical decision log — Dynamic PV deployment planning (RL)

Each entry: decision, rationale, source/justification. This file feeds the thesis
"research method" and "implementation" chapters.

## D01 — Network model: IEEE 33-bus radial feeder (Baran & Wu 1989)
Proposal-mandated. Base voltage 12.66 kV, 32 load buses, total 3.715 MW / 2.3 MVAr.
Each load bus is treated as one "neighborhood"; its nominal load is the diversified
coincident peak of that neighborhood. Justification: no public Tehran feeder data
(proposal risk table R1); standard test feeder ensures reproducibility and
comparability with the DRL voltage-control literature (Chen et al. 2024 use the same).

## D02 — Power flow: LinDistFlow sensitivity matrices for training, OpenDSS for validation
Full OpenDSS in the training loop is computationally infeasible (~1e8 solves for 1M
training steps x 24h x rep-days). The network topology is fixed and radial, so
LinDistFlow reduces to two constant matrices (common-path R and X); a full network
evaluation is a matrix multiply. OpenDSS (via OpenDSSDirect.py) is used to (a) quantify
the approximation error of the linear model and (b) re-evaluate final policies.
Reference: Baran & Wu (1989); LinDistFlow accuracy on 33-bus is typically <1% voltage error.

## D03 — Distribution transformers per neighborhood
Each load bus i is served by n_i = ceil(S_peak_i / (0.75 * 250 kVA)) transformers of
250 kVA (proposal's Iranian urban standard), i.e. designed for ~75% loading at nominal
coincident peak. Neighborhood transformer capacity = n_i * 250 kVA. Transformer stress
uses |net apparent power| so midday PV backfeed also counts (reverse power flow).

## D04 — Load archetypes, heterogeneous mixes (correction to proposal)
The proposal used one mix (65/25/10, residential peak 20-22h) for all neighborhoods.
That makes every feeder evening-peaking -> PV-without-storage cannot reduce any peak and
CF is homogeneous (allocation problem degenerates). Correction: per-bus mixes drawn from
four neighborhood classes (residential / mixed / commercial / administrative), and summer
archetypes include the cooling (A/C) afternoon component so that the *system* coincident
peak lands in the afternoon (Iran's national summer peak occurs ~13:00-17:00), while
residential feeders keep a dominant evening peak. Class assignment is seeded and fixed.

## D05 — PV production: PVGIS hourly series (SARAH3, 2005-2023), per-kWp
PVGIS seriescalc with pvcalculation=1 returns hourly AC power of a 1-kWp crystalline-Si
building-mounted system at optimal tilt for Tehran (35.6892 N, 51.3890 E), 14% system
losses, temperature effect included via T2m — this implements the proposal's
P = eta*A*G*(1 - beta*(T-25)) with JRC's validated model. Stochastic irradiance is
modeled by bootstrap-sampling real summer days from the multi-year record (primary),
with Beta distributions fitted to the daily clearness/production index reported for the
thesis text (proposal compatibility).

## D06 — Dynamic coincidence factor (correction to proposal)
CF_i is NOT a static input. At every period it is recomputed as the marginal local peak
reduction per additional MW at the current net-load profile:
CF_i(t) = [max_h netload_i(t,h) - max_h (netload_i(t,h) - eps*pv_shape(h))] / eps.
This captures CF erosion (duck-curve effect): as penetration grows the local net peak
migrates to evening hours and marginal CF -> 0. Static CF would contradict the thesis's
own path-dependence premise.

## D07 — Action spaces
Continuous (PPO/SAC): raw vector length N+1 -> softmax -> shares of available budget;
component 0 is a reserve share carried over to the next year. This restores meaning to
budget efficiency (the proposal's softmax-over-N forced full spending every year).
Discrete (DQN): the annual budget is split into M=8 tranches; each sub-step DQN picks
one neighborhood (or reserve) to receive the next tranche; physics and reward are
evaluated at year end. This replaces the proposal's per-neighborhood discretization,
which would produce K^32 joint actions (infeasible for DQN).

## D08 — Reward (normalized, corrected)
R_t = w_peak * dPeak_t/Peak_gross_t  -  w_tr * overload_bushours_norm
      - w_v * voltage_violation_bushours_norm   [- optional delta shaping, ablation only]
dPeak_t = max_h grossload_sys(t,h) - max_h netload_sys(t,h) (peak reduction attributable
to PV in year t). All terms dimensionless in [0,1]-ish ranges so weights are meaningful.
The proposal's delta*sum(CF_i*b_i) input-shaping term is excluded from the main reward
(double-counts the mechanism, gameable) and studied as an ablation. The gamma*sum(b)
budget-cost term is dropped: with a fixed annual budget plus explicit reserve action,
budget efficiency is an evaluation metric, not a reward term (under forced full
spending it was a constant anyway).

## D09 — Uncertainty models
Adoption: accept_i,t ~ Beta(a_i, b_i), mean in [0.35, 0.75] heterogeneous by
neighborhood class (income/awareness proxy), i.i.d. across years (Bass diffusion left
as ablation/future work, per proposal limitations). Demand growth: g_t ~ N(2.5%, 1%)
truncated at [0, 6%], common shock + small per-bus noise. Irradiance: bootstrap of real
PVGIS summer days (see D05). Load shape noise: multiplicative N(1, 0.05) per hour-draw.

## D10 — Rooftop potential per neighborhood
potential_i = rho_i * P_peak_i with rho_i in [0.3, 1.0] by class (residential higher
roof/load ratio, commercial lower). City-average rho ~ 0.5, consistent with Ranjgar &
Niccolai (2023): Tehran rooftop potential 2151 MW ~ 1/3 of Tehran's summer peak.

## D11 — Budget calibration
Annual budget expressed in MW of subsidizable capacity: B = 0.25 MW/yr (~13% of total
potential per year offered; with mean acceptance ~0.55 the 10-year program can reach
roughly 70-80% of potential if fully targeted). Sensitivity: {50%, 100%, 150%} of B.

## D12 — Baselines
(1) Random Dirichlet allocation; (2) Peak-first greedy; (3) CF-first greedy (static-CF
variant per proposal + adaptive-CF variant); (4) Rolling-horizon deterministic LP (MPC):
each year solves a linear epigraph peak-minimization over the remaining horizon with
expected acceptance/growth, applies year-1 decision; (5) Perfect-information oracle: the
same LP solved on realized scenario draws -> upper bound, enables optimality-gap
reporting. LP solved with scipy.optimize.linprog (HiGHS).

## D13 — Evaluation protocol
5 seeds per algorithm; 100 held-out test episodes (fixed scenario seeds shared across
all policies, paired); Wilcoxon signed-rank + Cliff's delta effect size + bootstrap CIs.
Sensitivity: one-at-a-time (OAT) retraining on {BESS share, demand growth, horizon,
budget, gamma}; robustness: nominal-trained policy evaluated on perturbed envs without
retraining. Full-factorial retraining (proposal wording) is computationally infeasible
(~576 combos x seeds) and statistically unnecessary.

## D15 — Feeder impedance calibration (x0.5)
The classic IEEE 33-bus feeder sags to ~0.904 pu at nominal load — a weak
overhead feeder, not an urban 20 kV cable network, and it floods the reward with
action-independent voltage penalties. Branch impedances are scaled by 0.5 so the
feeder meets the +/-5% planning criterion at design load (urban underground XLPE
cables: larger cross-sections, shorter spans). Validated: nominal-load V_min
~0.95 pu after scaling.

## D16 — Differential (PV-attributable) grid-stress penalties
Raw violation counts are dominated by load growth the agent cannot influence
(evening peaks, feeder-end sag), which adds variance without signal. The reward
penalizes the *difference* between grid stress with PV and the no-PV
counterfactual evaluated on identical stochastic draws:
  R = w_peak * dPeak_frac - w_tr * (overload_net - overload_gross)
                          - w_v  * (violations_net - violations_gross).
Negative deltas (PV relieving midday transformer loading / voltage sag) are thus
rewarded; PV-caused backfeed overload and overvoltage are penalized. Raw
violation fractions remain reported as evaluation metrics.

## D14 — BESS sensitivity parameter
A share phi of installed PV is battery-paired; batteries shift 30% of that share's
daily PV energy from hours 10-16 to hours 18-23 (greedy toward highest net-load evening
hours). phi in {0, 10%, 20%, 30%} (OAT).

## D18 — The peak term ignores location
Irradiance is common to all buses, so the feeder-peak term depends only on total
installed PV. Verified numerically:
- 200 random placements of the same total MW give the same dPeak (to 1e-16).
- Across 21,000 ordered pairs of stored runs, more MW never gave a smaller
  final-year reduction.

On the stiff feeder the transformer term is zero and the voltage term is the
same for every policy, so differences between policies are total-capacity
effects only. Two acceptance-aware baselines were added:
- accept-greedy: offers in decreasing order of mean acceptance.
- max-exp-cap: exact allocation that maximises expected installed capacity
  (allocation.py).

max-exp-cap beats PPO by +0.016 (better in 99/100 scenarios) and the frozen
nominal PPO in all 14 OAT settings.
Script: scripts/16_review_checks.py. Results: results/review_*.

## D19 — Weak feeder as the network-binding case
With the original impedances (x1.0), undervoltage relief depends on placement.
For 1 MW installed, the annual voltage term is 0.006 when PV is near the
substation and 0.030 when it is at the far end; the peak term is the same.

New baselines:
- rolling-lp-V: adds a LinDistFlow undervoltage-magnitude term with weight lambda.
- max-exp-cap-V: weights w_i = 1 + kappa * s_i, where s_i is the undervoltage
  relief per MW.

Both were tuned on validation seeds 30000-30049, giving kappa = 0.5 and
lambda = 0.01. The small lambda acts as a tie-breaker among peak-optimal plans.
Larger lambda hurts, because undervoltage magnitude is not the same as the
violation count used in the reward.

PPO was trained on the weak feeder (5 seeds x 500k steps).
Script: scripts/17_weak_feeder.py. Results: results/weak_*. The max-exp-cap rows
in weak_evaluation.csv and weak_summary.csv come from an earlier greedy solver
and are superseded by results/hybrid_*.

## D20 — Hybrid and residual PPO over an analytic allocation layer
PPO outputs per-bus log-priorities u in [-1, 1]^32, with weights
w_i = prior_i * exp(2 u_i). Offers solve
max sum_i w_i E[min(o_i xi_i, H_i)] s.t. sum_i o_i = B exactly (KKT with
bisection), so the whole budget is offered and no offer goes to waste on
saturated buses. The observation also includes s_i.

Variants:
- hybrid: prior = 1.
- residual: prior = 1 + 0.5 s_i, so u = 0 reproduces max-exp-cap-V.

PPO hyperparameters, seeds, validation stream and checkpoint rule are the same
as D13. Weak feeder, 100 test scenarios:

| Policy        | Return |
|---------------|--------|
| residual      | 1.1146 |
| hybrid        | 1.1113 |
| max-exp-cap-V | 1.1071 |
| PPO           | 1.0949 |
| rolling-lp-V  | 1.0842 |

Residual minus the rule: +0.0076 [0.0054, 0.0097], better in 74/100 scenarios.
The gain comes from the voltage term (0.141 vs 0.133); the peak term is
unchanged.
Script: scripts/18_hybrid.py. Results: results/hybrid_*.

## D21 — Robustness of the hybrid and residual PPO (weak feeder, no retraining)
14 one-at-a-time perturbations: BESS share, demand growth, horizon, budget,
rooftop potential, and Bass-type acceptance. Policies trained on the nominal
weak feeder are evaluated frozen; max-exp-cap-V (kappa = 0.5) and rolling-lp-V
(lambda = 0.01) are re-run in each setting. 100 paired scenarios per setting.

Residual, counted by the 95% CI of the paired difference:

| Reference     | Better | Tie | Worse |
|---------------|--------|-----|-------|
| max-exp-cap-V | 7      | 7   | 0     |
| rolling-lp-V  | 13     | 1   | 0     |
| plain PPO     | 13     | 1   | 0     |

Against rolling-lp-V, the tie is potential x2, where LP-V is slightly ahead
(-0.006, p = 0.03).

The hybrid variant is less robust: against the rule it is better in 4 settings,
ties in 7 and is worse in 3. It is, however, best of all policies under Bass
acceptance.

In the settings where residual ties with the rule (BESS share, 4% growth,
budget changes), it still gains on the voltage term but installs less capacity
(about 1.42 vs 1.48 MW). With BESS, capacity is worth more, so the trade-off
learned on the nominal case no longer pays; retraining should recover it.
Script: scripts/19_hybrid_robustness.py. Results: results/robust_*.

## D22 — Reporting in engineering units
How the conversions work:
- The voltage term is -0.5 x the summed change in the fraction of bus-hours
  outside 0.95-1.05 pu over the D sampled days. One 92-day summer has
  32 x 24 x 92 = 70,656 bus-hours, so relieved bus-hours per summer =
  -volt_delta_sum / T x 70,656.
- A no-PV run on the same scenarios gives the gross peak (mean 4.19 MW) and the
  gross undervoltage share.
- Mean-annual kW is approximate (sum_t dP_t/P_t x mean_t P_t); final-year kW is
  exact.

Weak feeder:
- Without PV, 34.4% of bus-hours are below 0.95 pu.
- The rule (max-exp-cap-V) relieves 1,875 undervoltage bus-hours per summer
  (7.7% of all undervoltage bus-hours). Residual relieves 1,993 (8.2%): +118
  [103, 133], with no loss of peak reduction (-0.3 kW [-0.9, 0.2]).
- Against plain PPO, residual gains +4.7 kW of peak and +120 bus-hours.
- PV alone leaves about 31.6% of bus-hours in violation, because the evening
  sag is out of PV's reach.

Stiff feeder:
- max-exp-cap - PPO = +6.6 kW [6.0, 7.1] of mean annual peak reduction, on
  about 405 kW of total reduction.
Script: scripts/21_practical_units.py. Results: results/units_*.

## D23 — Retraining the residual in the 7 settings where it tied
Both sides adapt to each setting: kappa is re-tuned on validation seeds
30000-30049, and the residual PPO is retrained on top of the re-tuned rule
(1 seed, 500k steps, as in the D13 OAT retraining). Re-tuned kappa:
- 0.5 for BESS 20%, BESS 30% and 4% growth
- 0.25 for the 5-year horizon and the 125 kW/yr budget
- 1.0 for the 375 kW/yr budget
- 0 for Bass acceptance

Retrained residual minus re-tuned rule, 100 test scenarios:
- Budget 375 kW/yr: +0.0091 [0.0045, 0.0135] (better).
- Bass: +0.0051 [0.0000, 0.0102] (borderline better, p = 0.07).
- BESS 20%, BESS 30%, 4% growth, 5-year horizon: ties.
- Budget 125 kW/yr: -0.0030 [-0.0061, 0.0002] (borderline worse, Wilcoxon
  p = 0.03).

Retraining therefore does not turn ties into wins in general. Where placement
offers little (BESS, a short horizon or a small budget), the tuned rule is
close to optimal. Re-tuning kappa alone captures much of the adaptation:
+0.0024 at the 5-year horizon and +0.0043 at the 125 kW/yr budget.

Overall, across the 15 weak-feeder settings, the best residual (frozen or
retrained) is better than the rule in 9 settings and ties in 5. Bass and the
125 kW/yr budget are borderline in opposite directions. No setting is
significantly worse by CI.
Script: scripts/20_retrain_residual.py. Results: results/retrain_*.

## D24 — Weak-feeder validation and exact re-evaluation of the voltage results
LinDistFlow vs OpenDSS on 200 random states (scripts/23_validate_feeders.py):

| Feeder | Max error | Mean error | Mean bias | Below 0.95 pu (OpenDSS / LinDistFlow) | Misclassified |
|--------|-----------|------------|-----------|----------------------------------------|---------------|
| Stiff  | 2.1e-3 pu | 0.75e-3 pu | –         | 0 / 0                                  | –             |
| Weak   | 8.8e-3 pu | 2.8e-3 pu  | +1.6e-3   | 20.2% / 17.1%                          | 3.1%          |

To check that the weak-feeder conclusions survive, scripts/24_opendss_voltage_check.py
replays five policies on the 100 test scenarios with identical draws and solves
every bus-hour in OpenDSS, about 1.4 million power flows. The policies are
max-exp-cap-V, residual s0, PPO s0, rolling-lp-V and max-exp-cap.

Results:
- OpenDSS finds more violations, and LinDistFlow overstates the absolute
  PV-attributable voltage relief by about 19% (rule: 0.133 LinDistFlow vs 0.111
  OpenDSS).
- Policy comparisons are preserved. Residual minus the rule in the voltage term
  is +0.0058 [0.0048, 0.0069] with LinDistFlow and +0.0062 [0.0051, 0.0074]
  with OpenDSS (better in 85/100 scenarios, r_rb 0.88).
- Plain PPO's voltage term is +0.0019 [0.0003, 0.0035] over the rule under
  OpenDSS, so its deficit is in capacity, not placement.

Statistics also include the matched-pairs rank-biserial effect size
(stats.rank_biserial_paired) and a seed-aware hierarchical bootstrap.

## D25 — Exact power flow with losses
Proposition 1 holds only for the lossless model. In OpenDSS
(scripts/32_losses_placement.py, mid-programme load, mean PV day), filling the
far end first instead of the substation end changes the feeder-head peak
reduction by 7.9 / 13.7 / 17.6 kW (stiff) and 18.4 / 31.9 / 41.0 kW (weak) at
0.25 / 0.5 / 0.75 MW installed; with 1 MW or more installed the net peak moves
to the evening and placement makes no difference.

All test episodes of the main policies were replayed on both feeders, solving
every bus-hour in OpenDSS (scripts/27_opendss_exact_return.py). The exact
return uses the head-power peak including losses plus the exact voltage term.
Losses raise every policy's return by about 0.02-0.03, but the comparisons hold:

| Feeder | Comparison                   | LinDistFlow | OpenDSS                                    |
|--------|------------------------------|-------------|--------------------------------------------|
| Stiff  | rule − PPO                   | +0.0156     | +0.0156 [0.0142, 0.0170], 99/100, +6.6 kW  |
| Stiff  | rule − LP                    | —           | +0.041                                     |
| Weak   | residual − voltage rule      | +0.0076     | +0.0077 [0.0052, 0.0102]                   |
| Weak   | PPO − voltage rule           | —           | −0.015                                     |
| Weak   | voltage weighting            | +0.020      | +0.029                                     |

Far-end placement also cuts losses, which is why voltage weighting gains more
under exact power flow.

## D26 — Misspecified acceptance model
In scripts/28_acceptance_misspec.py the structured policies plan with a wrong
acceptance belief while the environment keeps the true one.

- Uniform errors (all class means x0.75 or x1.25) and a wrong dispersion
  (concentration x0.5) cost the rules little:
  - The stiff rule still beats PPO by about +0.015 (99/100).
  - The residual still beats the rule by +0.007 to +0.008 on the weak feeder.
- An error that reverses the ranking of residential and commercial
  ("class-errors": residential x1.25, mixed x0.85, commercial x0.80,
  admin x1.15) breaks the exact rule: -0.125 on the stiff feeder. The KKT
  allocation is all-or-nothing on the class that looks best, so it puts the
  budget into residential (true acceptance 0.45) for years 2-8. The rolling
  LP-V is fragile as well (-0.06 to -0.07).

Fair version (scripts/29_misspec_trained.py): PPO and the residual agent were
trained in a simulator with the same wrong means (three seeds each) and
evaluated in the true environment.
- Stiff: believed-trained PPO lost 0.028 (vs PPO trained on the true model)
  and beats the believed rule by +0.082 (100/100).
- Weak: all agents lose 0.05-0.07 and tie (PPO vs rule +0.002, residual vs
  rule +0.001, both n.s.).

Conclusion: the structured advantage requires the ranking of acceptance across
classes to be about right. Under a ranking error, the stochastic softmax policy
hedges better than the all-or-nothing exact allocation. A hedged allocation is
a natural next step.

## D27 — Unaccepted offers returned to the reserve
With EnvConfig.refund_unaccepted=True and a budget of 0.165 MW/yr, the rule
installs about the same capacity as in the base case (about 1.47 MW). The
re-tuned kappa is 4.0: voltage weighting no longer costs capacity.
Results (scripts/26_refund_variant.py, 100 scenarios):

| Policy                       | Return | vs voltage rule                   |
|------------------------------|--------|-----------------------------------|
| Voltage rule (kappa = 4)     | 1.073  | —                                 |
| Residual (retrained, 5 seeds)| 1.069  | −0.004 [−0.006, −0.002], 38/100   |
| Plain PPO (5 seeds)          | 1.053  | −0.020                            |

Residual beats plain PPO by +0.016 (90/100), so "structure over algorithm"
does not depend on the lost-offer assumption. The residual's gain over the
rule, however, does: with refunds a strongly voltage-weighted rule is enough.

## D28 — Hedging and learning the acceptance model (follow-up to D26)
Both variants were chosen on tuning seeds 30000-30049 by minimax regret over
five beliefs (the true model plus the four D26 cases), with regret measured
against the unhedged rule under the true model. Tests use 100 scenarios.

Hedged rule (scripts/30_hedged_rule.py). Offers = (1 - eta) x exact allocation
+ eta x spread in proportion to weight x headroom. This does not work: it is
expensive insurance that does not cure the ranking error.
- Stiff (eta 0.7): costs 0.050 when the model is right. Under class-errors it
  gains 0.046 over the unhedged rule but still trails believed-trained PPO by
  0.036.
- Weak (eta 0.2): costs 0.016 and gives no protection.

Adaptive rule (scripts/31_adaptive_rule.py, AdaptiveCapacityPolicy). Believed
class means act as a prior worth n0 MW of offers. Each year the rule adds the
observed uptake on buses not capped by headroom and recomputes the exact
allocation. Chosen n0 = 0.02 on both feeders.
- Stiff, true model: −0.001 vs the exact rule (n.s.).
- Stiff, class-errors: +0.110 (100/100) over the unhedged believed rule, tied
  with PPO trained on the true model, and +0.028 (99/100) over believed-trained
  PPO.
- Weak, class-errors: +0.026 over the unhedged believed rule and +0.024
  (78/100) over believed-trained PPO. It costs 0.010 with the true model and
  0.022 with the optimistic belief. A stronger prior (n0 = 1) removes that
  cost but protects less.

Practical message: the structured approach needs the acceptance ranking to be
roughly right. Updating it from observed programme uptake makes it robust at
little cost.

## D29 — Substation OLTC on the weak feeder
Network33 now has an optional OLTC (EnvConfig.oltc_vmax; default None keeps
the head at 1.0 pu and reproduces all earlier results exactly). Hourly,
stateless control: 0.625% steps, line-drop compensation to an estimate of the
mean bus voltage (r_ldc, x_ldc from the nominal load proportions), setpoint
1.0 pu, head voltage limited to [2 - v_max, v_max]. v_max is the regulation
margin left after LV drop, light load and other feeders on the same busbar.
Scripts: 33_voltage_regulation.py (screen / train / eval / stats),
34_opendss_oltc_check.py (OpenDSS, OLTC acting on the exact head flow).

Without PV, the share of summer bus-hours below 0.95 pu falls from 34% (no
OLTC) to 25 / 15 / 6 / 1.2 / <0.1% for v_max = 1.01 / 1.02 / 1.03 / 1.04 /
1.05. PV never causes overvoltage (max 1.036 pu even with the full potential,
70% load and the clearest day).

The value of placement is not monotone in the margin. Voltage weighting
(kappa re-tuned per setting: 0.5, 0.5, 1.0, 0.5, 0.75, 0) gains +0.020 (no
OLTC), +0.053 (1.02; 1950 bus-hours), +0.034 (1.03), +0.002 (1.04), 0 (1.05):
a partial margin leaves many bus-hours just below the limit.

Agents trained without the OLTC do not transfer. Frozen residual: +0.008
(1.01), tie (1.02), +0.003 (1.03), -0.009 (1.04), -0.010 (1.05; its learned
priorities still favour far-end buses and it installs 0.1 MW less). Frozen
PPO -0.012 to -0.027; rolling LP-V -0.021 to -0.043.

Retrained at v_max = 1.02 (3 seeds each, 500k steps):
- residual vs re-tuned rule +0.0084 [0.0021, 0.0150], 59/100; mainly earlier
  installation (peak term +0.014) at a small voltage cost (-76 bus-hours);
  OpenDSS +0.0089 [0.0024, 0.0157].
- PPO vs rule -0.018 (OpenDSS -0.026); residual vs PPO +0.027 (OpenDSS +0.035).
- voltage weighting in OpenDSS +0.071.
Not retrained at 1.05: no undervoltage remains, so the problem reduces to the
stiff-feeder one (Proposition 1).

## D30 — Lookahead rule and perfect-information bound
scripts/35_lookahead_bound.py; RolloutCapacityPolicy in pvplan.baselines.

Lookahead rule (one-step rollout). Each year it tries kappa in {0, 0.25, 0.5,
1, 2, 4} for this year, follows the base rule afterwards on 40 futures sampled
from the model (seeds unrelated to the test scenario, shared by candidates),
and applies the best. About 30 s per episode, no training.
- Weak feeder: +0.0036 [0.0005, 0.0067] over the voltage-weighted rule, via
  timing (peak term +0.0034). Median kappa 0.25 in years 1-4, 1-2 from year 6.
  The residual agent still beats it by +0.0040 [0.0009, 0.0070], via the
  voltage term (+0.0081): placement within a year, not timing.
- OLTC 1.02: lookahead +0.0055 (n.s.) over the re-tuned rule; residual minus
  lookahead +0.0029 [-0.0035, 0.0092]: indistinguishable.

Wait-and-see bound. Per test scenario, all exogenous draws known; installs
a_it cost a_it / xi_it of budget (cumulative budget, carry-over allowed);
exact LinDistFlow return: peak epigraph plus one binary per liftable bus-hour;
new violations ignored, so it bounds every policy. HiGHS, 120 s per scenario;
best solution replayed in the simulator (difference at most ~2e-4, MIP
tolerance). Stiff feeder: pure LP, exact: 1.038; the rule attains 95% and the
clairvoyant LP 98%. Weak feeder (6,000-6,900 binaries per scenario): best
solution 1.248 on average, solver bound 1.315 (mean gap 5.1%); the residual
agent attains 89% of the wait-and-see return, the voltage-weighted rule and
the lookahead rule 89%, PPO 88%.
The gap is dominated by the value of knowing acceptance in advance (the
clairvoyant plan installs ~1.54-1.7 MW against ~1.48-1.50 MW).

## D31 — Voltage metrics and OLTC details
Reporting-only metrics added to the environment info (the reward is
unchanged; returns reproduced exactly): undervoltage deficit
mean(max(0, 0.95 - V)), counts below 0.94 and 0.96 pu, OLTC tap operations.
scripts/36_voltage_metrics.py (run / extra / timing / overvoltage / stats).

Nominal weak feeder, residual minus rule: +118 bus-hours (0.95), +120 (0.94),
+77 (0.96), +5.7 pu*h deficit [5.1, 6.4] (99/100). Peak term unchanged
(-0.0008 [-0.0021, 0.0006]), so the residual is significantly better for any
voltage weight >= 0.2. Gross deficit 456 pu*h per summer; the best policy
relieves 19% of it (8.2% of the bus-hours). 70% of undervoltage bus-hours fall
in PV hours, but only 24% could be lifted even with the full potential: the
small share is about depth, not timing (the old "evening" explanation was wrong).

OLTC: the count-based gain of voltage weighting peaks at 1.02 pu (1,950 vs 408
bus-hours), but the deficit gain stays similar (22.4 vs 18.1 pu*h): the peak
is largely a threshold effect. Fixed tap and LDC setpoints 0.99/1.01 give
identical results (tap at the limit in every undervoltage hour). Tap changes
per day 0 / 2.0 / 3.1 / 6.8 / 9.7 for limits 1.01-1.05. Max voltage with full
potential, 70% load, clearest hour: 1.036 pu.

Lookahead sensitivity (20 / 80 futures, finer grid): see
results/lookahead_summary.json. Gap closed between the unweighted rule and the
wait-and-see return: voltage-weighted rule 12%, lookahead 14%, residual 17%.

## D32 — Energy, emission and budget-efficiency accounting
The reward measures summer peak and voltage relief only. scripts/37_energy_emissions.py
replays every test episode of the main policies with identical draws (final installed
capacities match the stored results exactly) and records the installed capacity after
each annual allocation. PV energy = installed capacity of each year times the mean annual
yield of the PVGIS record (1,557 kWh/kWp, SD 33, 2005-2023); avoided CO2 with the IFI
default combined-margin factor for Iran, intermittent renewables (0.57 t/MWh, IFI
Dataset of Default Grid Factors v2.0, 2019). No reverse power flow occurs in any
simulated summer hour, so the PV output is taken as absorbed by the feeder load.

Stiff feeder: the rule delivers 13.71 GWh and avoids 7,816 t CO2 over ten years,
+0.71 GWh [0.67, 0.75] more than PPO (100/100 scenarios) and +2.45 GWh more than the
rolling LP, which offers only 1.62 of the 2.5 MW budget. Weak feeder: voltage weighting
gives up 0.20 GWh [0.12, 0.28] for +408 undervoltage bus-hours per summer; the residual
agent a further 0.40 GWh [0.34, 0.45] for +118 (about 2,000 against 300 bus-hours per GWh
given up); the residual agent delivers 0.35 GWh more than plain PPO.
