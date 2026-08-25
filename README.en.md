# Physics-Prior-Guided Longitudinal Phase Space Super-Resolution

[中文版 →](README.md)

![Version](https://img.shields.io/badge/version-0.x.x-yellow)
![Status](https://img.shields.io/badge/status-spec%20draft-orange)
![Spec](https://img.shields.io/badge/spec-v0.1%20draft-lightgrey)

> Paired-data simulation study on recovering high-resolution fine structures of the longitudinal phase space $(z, \delta)$ from low-resolution observations in FEL beam diagnostics, guided by physical priors.

---

## Overview

In free-electron laser (FEL) beam diagnostics, the longitudinal phase space $(z, \delta)$ is the central object for beam-quality assessment. TDS (transverse deflecting structure) + spectrometer setups output low-resolution observations, while high-resolution fine structures (compression folds, thin spikes, current peaks, local energy-spread variation) are physically critical but not resolvable from the low-resolution image alone.

This project generates strictly paired $(H, L, P)$ data via a lightweight longitudinal phase space simulator (independent of elegant / Ocelot), trains super-resolution neural networks, and asks **one core question**: under low-resolution observation, can physical priors help the network recover the high-resolution longitudinal phase space more accurately and more consistently with physics?

### Approach

- **Three-arm matched comparison** (identical data, loss, backbone, training config, evaluation metrics):
  - A: No prior (baseline) — $\hat{H} = \mathrm{NonNeg}(L_{\mathrm{up}} + G_0(L_{\mathrm{up}}))$
  - B: Image prior + residual — $\hat{H} = \mathrm{NonNeg}(P_2 + G_1(L_{\mathrm{up}}, P_2))$
  - C: Parametric prior + FiLM — $\hat{H} = \mathrm{NonNeg}(L_{\mathrm{up}} + G_2(L_{\mathrm{up}} \mid c_{\mathrm{prior}}))$
- **Training objective**: space-frequency hybrid loss $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{space}} + \lambda \mathcal{L}_{\mathrm{spec}}$, with $\lambda$ **frozen at 1.0** (pre-registered, no proxy selection).
- **Starting complexity**: Level 1 (non-Gaussian profile + asymmetry + local thickness variation + 3rd-order center line + 2nd/3rd-order compression folds).
- **Degradation**: $256 \rightarrow 64$ downsampling + Gaussian blur + Gaussian noise, with calibrated values ($\sigma_K$ / $\sigma_n$ / $\sigma_{\mathrm{smooth}}$) recorded in `config.yaml` and `99_change_log.md`.

---

## Repository layout

```text
.
├── README.md                                              # Chinese version (default)
├── README.en.md                                           # English version (this file)
├── LICENSE                                                # MIT
├── .gitignore
└── docs/
    ├── specs/                                             # Spec set (v0.1 draft, pending review)
    │   ├── README.md                                      # Spec index
    │   ├── 00_master_spec.md                              # Master: goals, symbols, milestones
    │   ├── 10_research_plan.md
    │   ├── 20_physics_generator_spec.md
    │   ├── 30_degradation_spec.md
    │   ├── 40_prior_spec.md
    │   ├── 50_network_spec.md
    │   ├── 60_training_spec.md
    │   ├── 70_evaluation_spec.md
    │   ├── 80_experiment_matrix.md
    │   ├── 90_delivery_spec.md
    │   ├── 99_change_log.md
    │   └── archive/
    │       └── chat-超分辨率增强模拟1.txt                 # Archived original conversation

# To be created at implementation stage:
#   src/                    # Source code (generators / models / training / evaluation)
#   data/<version>/         # Datasets
#   results/<EXP>.../       # Experiment outputs
#   final_report.md         # Final research report
```

---

## Current status

| Item | Status |
|---|---|
| Spec set | **v0.1 draft** (pending final user review) |
| Version | **0.x.x** (to be bumped to 1.x.x only after all milestones complete, the report is produced, and the user signs off) |
| Source code | Not started (will begin after M0 freeze) |
| Dataset | Not generated |
| Final report | Not produced |

### Versioning convention

- **0.x.x (current)**: v0.1 draft specs, no code, no executed pipeline.
- **1.x.x (target)**: All milestones (M0 → M6) complete, `final_report.md` produced, user sign-off → bump to 1.0.0; further increments 1.x.y track change magnitude.

### Milestones

| Milestone | Name | Entry condition | Exit criterion |
|---|---|---|---|
| M0 | Spec freeze | User review (incl. cross-doc consistency check) | v1.0 frozen; version history recorded |
| M1 | Basic simulator | M0 | `f(x) → (H, L, P)` operational; ACs pass |
| M2 | Dataset generation | G0 data-validity gate | Size / partition / reproducibility met |
| M3 | No-prior network (baseline) | G1(a) | Scheme A converges; baseline metrics recorded |
| M4 | Prior-bearing networks + prior-gain verdict | G2 | Three-class statistical verdict produced |
| M5 | Physics evaluation + ablation | G3 | Gain decomposition + direction verdict + hallucination verdict |
| M6 | Final report + go/no-go decision | `90` [S5] N1–N8 | `final_report.md` passes acceptance |

Full phase–gate–milestone–experiment mapping: [`docs/specs/80_experiment_matrix.md`](docs/specs/80_experiment_matrix.md) § [S10].

---

## Reading order

1. [`docs/specs/README.md`](docs/specs/README.md) — Spec index;
2. [`docs/specs/00_master_spec.md`](docs/specs/00_master_spec.md) — Master spec (must-read);
3. Read the relevant sub-spec for the current task (10–90);
4. Any change: first record in [`docs/specs/99_change_log.md`](docs/specs/99_change_log.md); only after approval, edit; once landed, mark Implemented.

---

## License

[MIT](LICENSE).

---

## Contact

TBD.