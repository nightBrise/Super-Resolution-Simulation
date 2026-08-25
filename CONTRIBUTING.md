# Contributing / 开发约定

> 本文档是项目的**开发约定**（git 工作流、PR 流程、commit 规范），不构成研究契约。研究契约见 [`docs/specs/`](docs/specs/)；任何对 spec 的修改须先登记到 [`docs/specs/99_change_log.md`](docs/specs/99_change_log.md) 并经用户批准。

## 1. Git 工作流：**GitHub Flow**

来源：[GitHub Docs — GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)。

### 1.1 主分支

- **主分支**：仅 `main`；受分支保护规则约束（见 [§ § 4](#4-主分支保护纪律)）。
- **不允许** `develop` / `release/*` / 长生命周期 feature 分支——这是 GitHub Flow 而非 Gitflow 的核心区别。

### 1.2 工作流程（5 步）

```text
1. Create branch  → 在 main 之上开分支
2. Make changes   → 在分支上提交独立完整的 commits
3. Open PR        → push 分支并开 Pull Request
4. Review & Merge → 至少 1 个 approve；CI 通过后 squash merge
5. Delete branch  → 合并后立即删除功能分支
```

### 1.3 分支寿命与合并频率

- 分支寿命 ≤ 1 周；超出须在 PR 描述中说明。
- 每天 ≤ 3 次合入 main（研究项目节奏，不必过度追求高频）。

---

## 2. 分支命名规范

### 2.1 通用前缀

| 前缀 | 用途 | 例子 |
|---|---|---|
| `feat/` | 新功能 | `feat/prior-fiilm-injection` |
| `fix/` | Bug修复 | `fix/sigma-k-calibration-overflow` |
| `docs/` | 文档（不影响代码） | `docs/update-readme-bilingual` |
| `refactor/` | 重构（不增加功能） | `refactor/unet-channel-width` |
| `test/` | 仅测试 | `test/exp02-baseline` |
| `chore/` | 杂项（依赖、配置） | `chore/bump-torch-2.4` |
| `experiment/` | 实验性工作（不保证合入 main） | `experiment/lr-sweep` |

### 2.2 项目特殊前缀：`r2-fix/`

实现级重跑修复（对应 spec `80_experiment_matrix.md` § [S11] R2 风险登记）：

```
r2-fix/<YYYY-MM-DD>-<N>
```

- `YYYY-MM-DD`：触发日期
- `N`：当日序号

**例子**：`r2-fix/2026-09-15-01`

**R2 重跑纪律**（与 spec 同步）：

1. R2 触发 → 在 `99_change_log.md` 登记 `Proposed` 行（带 `R2` 标签）；
2. 用户批准 → `Approved`；
3. 创建 `r2-fix/...` 分支；
4. 仅做实现层修复（不能改 spec/超参/数据）；
5. Commit message footer 引用 `99_change_log.md` 行号（如 `Refs 99:R2-2026-09-15`）；
6. 修复后**先在代理尺度复现归因**，再全量重跑；
7. 合并回 `main`（squash merge），删除分支，patch 文件保存于 `results/<EXP>/patches/`；
8. 全局上限 1 次；仍失败升级 `99` + 用户批准。

### 2.3 命名风格

- 全小写、用 `-` 连字符、不超过 60 字符；
- 名词在前、动词在后（如 `feat/unet-add-skip-connection` 而非 `feat/skip-connection-added`）。

---

## 3. Commit Message 规范：**Conventional Commits**

事实标准：[Conventional Commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)。

### 3.1 格式

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### 3.2 Type 取值

| Type | 含义 | SemVer 对应 |
|---|---|---|
| `feat` | 新功能 | MINOR |
| `fix` | Bug 修复 | PATCH |
| `docs` | 仅文档 | — |
| `style` | 格式（不影响逻辑） | — |
| `refactor` | 重构 | — |
| `test` | 仅测试 | — |
| `chore` | 构建/依赖 | — |
| `perf` | 性能改进 | PATCH |
| `revert` | 回滚 | — |

### 3.3 Scope（可选）

项目模块名：`model-a` / `model-b` / `model-c` / `data-gen` / `degradation` / `prior` / `train` / `eval` / `docs` / `spec`。

### 3.4 Description 规则

- 不超过 72 字符；
- 祈使语气（"add" 而非 "added" / "adds"）；
- 不大写首字母、不加句号。

### 3.5 Footer 规范

至少含一种：

- **关联 issue/PR**：`Refs #12`
- **关联 spec**：`Refs spec/50[S10]` 或 `Refs spec/80[S11]R2`
- **关联变更日志**：`Refs 99:R4-2026-09-01`（指向 `99_change_log.md` 中行号或锚点）
- **破坏性变更**：`BREAKING CHANGE: <说明>`
- **R2 重跑**：见 [§ 2.2](#22-项目特殊前缀r2-fix)

### 3.6 完整例子

feat(model-c): add FiLM modulation at bottleneck and decoder

Implements spec 50 [S10] C5. Tested with EXP-01 sanity check;
output shape verified at 256x256.

- Refs spec/50[S10]
- Refs spec/60[S4]C4
- Refs 99:R4-2026-09-01

```

---

## 4. 主分支保护纪律

### 4.1 GitHub 分支保护规则（在 repo Settings → Branches 配置）

✅ Require a pull request before merging
✅ Require approvals: 1
✅ Require linear history（强制 squash merge 或 rebase）
❌ Allow force pushes（必须关闭）
❌ Allow deletions（必须关闭）
✅ Rules applied to everyone including administrators

### 4.2 纪律声明（与 spec 集镜像）

- **Spec 集冻结**（`00 [S9] C2`）↔ **`main` 分支纪律**：
  - 冻结后的 Spec 不得在实现阶段修改 → 走 `00 [S9]` 重大需求变更；
  - 冻结后的 `main` 不得 force-push 或 reset → 走 PR + squash merge。
- **Spec 修改** ↔ **代码修改**：二者 SHALL 同步写入 `99_change_log.md`，互为镜像。

### 4.3 工作纪律（自我约束）

虽然 GitHub 免费版允许仓库所有者绕过保护规则，但本项目要求自我执行：

- ❌ 直接 `git push origin main`（即使能成功）
- ✅ 创建分支 → push → 开 PR → review → merge
- ✅ 每个 commit 是独立完整的单元（便于单独 revert）

---

## 5. PR 流程

### 5.1 创建 PR

- **Title**：与第一个 commit 同形（Conventional Commits）；
- **Body**：使用 [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md)；
- **Draft**：未完成的 PR 标记为 Draft。

### 5.2 Review 检查清单

审查者（reviewer）至少检查：

- [ ] Title 与 commit 历史符合 Conventional Commits
- [ ] 代码与声明的 spec claim 一致
- [ ] 训练/测试影响范围明确（哪个 EXP / 哪个里程碑）
- [ ] `config.yaml` 三元组（code/data/spec version）若改动则已更新
- [ ] commit footer 引用了对应的 spec 行号或 99 变更日志
- [ ] 无 force-push 历史
- [ ] 分支命名符合规范

### 5.3 合并方式

- **默认 squash merge**：每个 PR 一个 commit 进入 `main`（commit 历史干净、便于审计每个 PR 引入的改动）
- **Merge commit**：仅在以下情况使用——PR 包含大量独立 commit 各自代表一个完整单元（如大量 backport）
- **Rebase merge**：与 squash 类似但保留原始 commit 列表

### 5.4 合并后

- 自动删除分支（GitHub 设置中勾选）；
- 验证 `main` commit hash 与本地一致。

---

## 6. Tag / Release 流程

### 6.1 版本号（SemVer）：`MAJOR.MINOR.PATCH`

| 触发事件 | 增量位 | 对应 spec 操作 |
|---|---|---|
| 重大需求变更（`00 [S9] 规则 6`） | MAJOR | 废弃旧 spec，按 write-spec 流程立新 spec |
| 普通 spec 变更（`99 [S5]`） | MINOR | 修改 spec + 同步登记 99 |
| 实现级修复（R2 重跑） | PATCH | R2 修复 + 同步登记 99 |
| 文档/CI 改动 | 不打 tag | 不涉及 spec |

### 6.2 项目版本号约定（特殊）

- **当前**：`0.x.x`（spec v0.1 草案，代码未实现）
- **目标**：`1.x.x`（spec v1.0 冻结 + M0–M6 全部跑通 + `final_report.md` 验收后）
- **首次升级**：M0 通过时打 `v1.0.0` 附注 tag

### 6.3 打 tag 命令

```bash
git tag -a v1.0.0 -m "Spec set v1.0 frozen; M0 milestone complete"
git push origin v1.0.0
```

### 6.4 GitHub Release

- 基于 tag 创建 Release；
- 自动从 commit 历史生成 release notes（Conventional Commits 工具可解析）；
- 关键 Release 手动撰写 summary（说明对应里程碑）。

---

## 7. 可复现性三元组：code / data / spec version

每个实验的 `config.yaml` 必含三个字段：

| 字段 | 含义 | 来源 |
|---|---|---|
| `code_version` | 运行的 commit hash | `git rev-parse HEAD` |
| `data_version` | 数据集版本号 | `data/<版本>/manifest.json` |
| `spec_version` | spec 集版本号 | `git describe --tags` |

**强制约定**（`90 [S5] N8` 预注册对账）：

- 三个字段必须与 `final_report.md` / `config.yaml` 中报告的版本一致；
- 任何不一致视为验收失败。

**自动写入示例**（推荐在训练脚本顶部）：

```python
import subprocess

CODE_VERSION = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
SPEC_VERSION = subprocess.check_output(["git", "describe", "--tags"]).decode().strip()
```

---

## 8. 与 spec 集的协同

### 8.1 spec 是研究契约，不是开发手册

| 范围 | 位置 |
|---|---|
| **研究目标、实验设计、公平性约束、评估判据** | `docs/specs/*.md` |
| **Git 工作流、commit 规范、PR 流程、tag 流程** | 本文件（CONTRIBUTING.md） |
| **PR 模板** | `.github/PULL_REQUEST_TEMPLATE.md` |
| **CI/CD 配置** | `.github/workflows/`（如未来加入） |
| **审查角色（CODEOWNERS）** | `.github/CODEOWNERS`（如未来加入） |

### 8.2 修改 spec 的纪律（不是 git 流程）

- 改 spec → 先登记到 `99_change_log.md`（Proposed）→ 用户批准（Approved）→ 修改 → 置 Implemented；
- 不允许"边写代码边改 spec 不登记"；
- 详见 spec `00 [S9]`。

### 8.3 修改代码的纪律（git 流程）

- 按本文件 [§ 2–6](#2-分支命名规范)；
- commit footer 关联 spec / 99 行号；
- PR 模板填写 spec compliance 节。

---

## 9. 快速参考卡

```bash
# 新功能
git checkout main
git pull
git checkout -b feat/my-feature
# ... work, commits ...
git push -u origin feat/my-feature
# 开 PR，merge 后删除分支

# R2 实现级修复
git checkout -b r2-fix/2026-09-15-01
# ... 仅实现层修复 ...
git commit -m "fix(model-c): correct FiLM gamma overflow

Fix overflow when gamma computation underflows on small batch.
Verified by re-running EXP-01c calibration script.

- Refs 99:R2-2026-09-15
- Refs spec/80[S11]R2"
git push -u origin r2-fix/2026-09-15-01
# 开 PR，merge 后删除分支；保存 patch 文件到 results/<EXP>/patches/

# 打 tag（M0 冻结时）
git tag -a v1.0.0 -m "Spec set v1.0 frozen; M0 milestone complete"
git push origin v1.0.0
```

---

## 10. 后续可扩展（不在本文件范围）

- CI/CD 配置（`.github/workflows/`）：测试、lint、训练 smoke test；
- `CODEOWNERS`：指定审查者（如未来多人协作）；
- Issue 模板（已在 `.github/ISSUE_TEMPLATE/` 占位）；
- Release notes 自动化（基于 Conventional Commits + GitHub Actions）。

---

**最后更新**：与 spec v0.1 草案同步；如 spec 变更影响 git 流程约束，同步更新本文件并记录于 `99_change_log.md`。