# Test Report: M2 数据集生成（L0+L1+Acceptance）

- 运行时间（UTC）：2026-08-25T18:09:20Z
- 代码版本（commit8）：`1aa74c3d`（生成时 HEAD；M2 代码未提交，见 B 类登记）
- Spec 版本：v1.0
- 环境：`sr-sim` conda（python 3.11 / torch 2.4.0 / numpy 1.26.4 / pytest 9.1.1），check_env 通过

## 结果汇总

| 套件 | 结果 | 耗时 |
|---|---|---|
| L0 单元 + L1 集成（tests/unit tests/integration）| 111 passed, 0 failed | 97.5s |
| L3 acceptance（tests/acceptance -m acceptance，M1+M2）| 34 passed + 1 xfailed（M1 AC14 SSIM B 类待裁定，OQ-40-02），0 failed | 66.0s |
| 其中 M2 绑定（-m "acceptance and m2"）| 10 passed, 0 failed | 45.1s |

逐用例耗时见 `junit_l0l1.xml` 与 `junit_acceptance.xml`（--durations 摘要：集成
test_parallel_serial_bitwise 42.4s、test_regeneration_bitwise 22.4s、acceptance
test_g0_b_probe_method 43.0s——均在预算内）。

## M2 关键用例（05 [S5] 绑定）

- test_seeds（unit，8 用例）：SeedSequence 派生可复现、分支独立、★
  test_no_self_random_source（禁用全局 np.random 后生成函数仍正常）。
- test_dataset_builder（integration，13 用例）：manifest 三元组/γ 块/标定值、
  HDF5 float32+gzip4+按样本切分、重生成逐位一致、★ test_no_c_cross_split、
  ★ test_gamma_block（固定总体分位数）、并行/串行逐位一致、1:1、OOD 极端参数。
- test_m2_dataset（acceptance，10 用例）：G0(a) W8 覆盖 0.74 ≥ 0.6、G0(b) 探针
  min(s_x)=0.0 < 0.5、G0(c) SNR_hf 中位数 dev1 0.076 / v1 0.072 < 0.1、
  规模计数 dev1 2000/500/250/250 与 v1 20000/2000/1000/1000、manifest 三元组、
  γ 块实数据校验、OOD 500 不相交、种子派生与 manifest 一致。

## 备注

- 断言只覆盖协议/契约/不变量与规格定死的门禁阈值（05 [S1]），不断言研究结果。
- 数据集产物与 G0 评估见 `results/M2_dataset/` 与 `data/dev1/`、`data/v1/`。
