# Pull Request

> 完整开发约定见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

## Summary<!-- 1-3 sentences: what this PR does and why -->

<!--
例：
This PR implements the FiLM modulation at the bottleneck and decoder layers
of model C, as required by spec 50 [S10] C5. Verified with EXP-01a sanity check.
-->

## Test plan<!-- How was this tested? -->

- [ ] Unit tests pass
- [ ] Smoke test (`python -m src.train --smoke`) passes
- [ ] EXP-01a sanity: loss decreases, output shape 256×256, `Ĥ ≥ 0`
- [ ] EXP-01b/c/d 标定项（如适用）：σ_K / σ_n / σ_smooth 标定通过
- [ ] EXP-0X 完整实验（如适用）：见 `results/<EXP>/summary.json`

## Spec compliance<!-- Which spec claims does this implement or modify? Cite section + claim id. -->

| Spec | Section | Claim | Implements / Modifies |
|---|---|---|---|
| 50 | [S10] | C5 | Implements |
| 60 | [S4] | C4 | Implements |

（如不涉及 spec 变更，填 "N/A — pure implementation, no spec change"）

## Change log<!-- Refs 99_change_log.md entry -->

- [ ] 此 PR **不**涉及 spec 变更（仅实现层） → 填 "N/A"
- [ ] 此 PR 涉及 spec 变更 → 已在 `99_change_log.md` 登记，对应行号：`Refs 99:<行号>`
- [ ] 此 PR 是 R2 实现级重跑 → 已在 `99_change_log.md` 登记 R2 标签，对应行号：`Refs 99:R2-<日期>`

## Code / data / spec version<!-- Reproducibility triple (per CONTRIBUTING.md §7) -->

- `code_version`：`<commit hash>`（本分支 HEAD）
- `data_version`：`<data/<version>/>`（如改动）
- `spec_version`：`<git describe --tags>`（如改动）

## Files changed<!-- Brief summary of touched files -->

<!-- 例：
- src/models/c_unet.py — add FiLM modulation
- src/train.py — pass c_prior through to model C
- config.yaml — record code_version
- results/EXP-01_Main/summary.json — sanity check passed
-->

## Checklist<!-- Reviewer-facing pre-submission checklist -->

- [ ] Title 与第一个 commit 符合 Conventional Commits
- [ ] Commit footer 引用了对应 spec 行号或 99 变更日志
- [ ] 分支命名符合 CONTRIBUTING.md §2
- [ ] `config.yaml` 三元组（code/data/spec version）已更新（如适用）
- [ ] `metrics.csv` / `summary.json` / `visuals/` 已生成（如适用）
- [ ] 无 force-push 历史
- [ ] 无引入 `c_high` 到方案输入（`00` 全局约束 3）
- [ ] 三方案公平性保持（`00` 全局约束 4）
- [ ] 无物理损失进入训练（`00` 全局约束 5，除非 spec 变更已 Approved）
- [ ] `git status` 干净，no debug files / `__pycache__/` / `.ipynb_checkpoints/`

## Reviewer notes<!-- Optional -->

<!-- 任何审查者需要知道的额外信息 -->

---

<!-- 本模板由 .github/PULL_REQUEST_TEMPLATE.md 自动加载；修改本文件本身请走 PR。 -->