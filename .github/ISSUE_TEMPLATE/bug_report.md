---
name: Bug report
about: 报告代码、实验、文档中的错误
title: "[BUG] <简短描述>"
labels: bug
---

## 描述<!-- What happened? -->

<!-- 清晰地描述 bug -->

## 复现步骤<!-- How to reproduce? -->

1.
2.
3.

## 期望行为<!-- What did you expect? -->

## 实际行为<!-- What actually happened? -->

## 环境<!--

- 代码 commit hash：`git rev-parse HEAD`
- 数据集版本：`data/<版本>/manifest.json`
- spec 版本：`git describe --tags`
- 操作系统：
- Python / PyTorch / CUDA 版本：

## 相关文件<!-- Which files / experiments are affected? -->

<!--
例：
- src/models/c_unet.py
- results/EXP-02_Main/summary.json
- docs/specs/60_training_spec.md [S2]
-->

## Spec 合规性<!-- Is this a spec violation or an implementation bug? -->

- [ ] 这是 spec violation（spec 本身有问题）→ 需走 `99_change_log.md` 流程
- [ ] 这是 implementation bug（代码与 spec 不一致）→ 直接修代码
- [ ] 不确定

## 优先级<!-- Priority -->

- [ ] Blocker（阻塞当前阶段）
- [ ] High（影响下一里程碑）
- [ ] Medium（一般）
- [ ] Low（可选改进）

## 额外上下文<!-- Anything else? -->

<!-- 截图、错误堆栈、相关 issue 引用 -->