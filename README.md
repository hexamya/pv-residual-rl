# Network-aware rooftop-PV subsidy planning with residual RL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23106383.svg)](https://doi.org/10.5281/zenodo.23106383)

Multi-year allocation of a rooftop-PV subsidy budget across the 32
neighbourhoods of an IEEE 33-bus feeder. The decision is made under uncertain
acceptance, demand growth and irradiance (19 years of PVGIS data for Tehran).

This repository contains the simulator, baselines, experiment scripts, results
and trained models of the study. Trained agents are loaded from their best
checkpoints (`results/logs/<tag>/ppo_seed<k>/best_model.zip`). The study asks when network-aware placement
matters for subsidy allocation, and what reinforcement learning adds on top of
an analytic allocation rule.

## Main findings

Full numbers are in [docs/decisions.md](docs/decisions.md), D18-D32.

1. **Without line losses, the feeder-peak objective does not depend on where
   PV is installed.** Irradiance is common to all buses, so the peak term
   depends only on total installed capacity; with losses the effect of
   placement is at most 18 kW (stiff) and 41 kW (weak). On the stiff
   (impedance x0.5) feeder, a no-training rule that maximises expected
   installed capacity (`max-exp-cap`) beats PPO by +0.016 in return, also
   under exact OpenDSS power flow.
2. **Placement matters once the network binds.** On the weak (impedance x1.0)
   feeder, a voltage-weighted version of the rule (`max-exp-cap-V`) relieves
   408 more undervoltage bus-hours per summer than the unweighted rule. A
   substation tap changer with a 4-5% margin removes this effect.
3. **Structure matters more than the learning algorithm.** PPO that sets
   priorities on top of an exact allocation layer (`HybridPVDeploymentEnv`)
   beats plain PPO in 98 of 100 scenarios. The residual variant beats the tuned
   rule by +0.0076 (+118 undervoltage bus-hours per summer, about 0.5% of those
   below the limit; +5.7 p.u.h of undervoltage deficit). A lookahead (rollout)
   version of the rule recovers part of this gain without training, and the
   gain disappears when unaccepted offers return to the budget.
4. **The rules need a roughly correct ranking of acceptance.** A class-ranking
   error costs the exact rule 0.125 on the stiff feeder; updating the class
   means from observed uptake (`AdaptiveCapacityPolicy`) removes the problem.
5. **Network support costs energy.** Over the ten-year programme the
   `max-exp-cap` rule delivers 0.71 GWh more PV energy than PPO (404 t CO2
   avoided at 0.57 t/MWh) and 2.45 GWh more than the rolling LP on the stiff
   feeder. On the weak feeder, voltage weighting gives up 0.20 GWh and the
   residual agent a further 0.40 GWh for their extra undervoltage relief.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

`requirements.txt` pins the versions the stored results were produced with
(Python 3.13). The PVGIS series is cached under `data/`; no download is needed.

## Pipeline

| Script | Purpose | Main outputs |
|---|---|---|
| `03_validate_env.py` | LinDistFlow vs OpenDSS, physics checks | `results/validation_report.json` |
| `05_train_rl.py` | PPO / SAC / tranche-DQN on the stiff feeder | `results/models/nominal/` |
| `06_evaluate_all.py` | paired evaluation, stiff feeder | `results/evaluation_nominal.csv` |
| `07_sensitivity.py`, `15_sensitivity_stats.py` | OAT robustness and retraining, stiff feeder | `results/sensitivity_*` |
| `11_bass_ablation.py` | Bass-type acceptance | `results/bass_*` |
| `16_review_checks.py` | acceptance-aware rules; location invariance | `results/review_*` |
| `17_weak_feeder.py` | weak feeder: placement value, LP-V, rule tuning, PPO | `results/weak_*`, `models/weak/` |
| `18_hybrid.py` | hybrid and residual PPO over the allocation layer | `results/hybrid_*`, `models/weak_{hybrid,residual}/` |
| `19_hybrid_robustness.py` | 14 weak-feeder perturbations, frozen policies | `results/robust_*` |
| `20_retrain_residual.py` | rule re-tuning and residual retraining per setting | `results/retrain_*` |
| `21_practical_units.py` | kW and undervoltage bus-hours | `results/units_*` |
| `23_validate_feeders.py` | LinDistFlow vs OpenDSS on both feeders | `results/validation_feeders.json` |
| `24_opendss_voltage_check.py` | OpenDSS replay of the voltage term (weak feeder) | `results/opendss_voltage_check*` |
| `26_refund_variant.py` | unaccepted offers returned to the budget | `results/refund_*`, `logs/weak_refund*/` |
| `27_opendss_exact_return.py` | exact return with losses, both feeders | `results/opendss_exact_*` |
| `28_acceptance_misspec.py` | policies planning with a misspecified acceptance model | `results/misspec_*` |
| `29_misspec_trained.py` | agents trained in a misspecified simulator | `results/misspec_trained_*`, `logs/misspec_*/` |
| `30_hedged_rule.py`, `31_adaptive_rule.py` | hedged and adaptive (uptake-learning) rules | `results/hedge_*`, `results/adaptive_*` |
| `32_losses_placement.py` | effect of placement on the head peak with losses | `results/losses_placement.json` |
| `33_voltage_regulation.py` | substation OLTC sweep; retraining at 1.02 p.u. | `results/oltc_*`, `logs/oltc_102_*/` |
| `34_opendss_oltc_check.py` | OpenDSS replay with the OLTC | `results/opendss_oltc*` |
| `35_lookahead_bound.py` | rollout lookahead rule and wait-and-see bound | `results/rollout_*`, `results/bound_*`, `results/lookahead_*` |
| `36_voltage_metrics.py` | undervoltage deficit, thresholds, tap operations, timing | `results/vmetric_*`, `results/uv_timing.json`, `results/oltc_overvoltage.json` |
| `37_energy_emissions.py` | PV energy, avoided CO2 and installed capacity per MW offered (replay of the test episodes) | `results/energy_*` |

All policies are evaluated on the same seeded scenarios (common random numbers):
- stiff and weak test sets: 10000-10099
- stiff-feeder OAT: 40000-40049
- rule and LP tuning: 30000-30049
- PPO validation: 999999

The `max-exp-cap` rows in `weak_evaluation.csv` and `weak_summary.csv` come from
an earlier greedy solver and are superseded by `hybrid_*` (D19).

## Package

```
src/pvplan/
  network.py      IEEE 33-bus + LinDistFlow sensitivity matrices; optional substation OLTC
  profiles.py     neighbourhood classes, load archetypes, rooftop potential
  pvgis.py        PVGIS processing
  uncertainty.py  acceptance / growth / weather / noise sampler
  environment.py  planning core; continuous, tranche and hybrid Gym envs; reporting-only voltage metrics
  allocation.py   exact expected-capacity allocation (KKT), voltage-relief weights
  baselines.py    heuristics, rolling LP (+ undervoltage term), oracle, acceptance-aware,
                  hedged, adaptive and rollout (lookahead) rules
  evaluation.py   paired evaluation protocol
  stats.py        Wilcoxon, Cliff's delta, matched-pairs rank-biserial, bootstrap CIs
  opendss_validate.py  OpenDSS circuit and snapshot solver
```

## Citation

The code is archived on Zenodo. To cite all versions, use the concept DOI:

> Jafari H, Sahebi H. pv-residual-rl: network-aware rooftop PV subsidy planning with an
> exact allocation layer and residual reinforcement learning [software]. Zenodo; 2026.
> https://doi.org/10.5281/zenodo.23106383

Each version also has its own DOI on the Zenodo record (v1.0.0:
https://doi.org/10.5281/zenodo.23106384). Version 1.1.0 adds the energy, emission
and budget-efficiency accounting (`37_energy_emissions.py`, D32). Citation metadata
is also in [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
