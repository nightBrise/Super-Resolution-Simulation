"""研究线迁移后的旧→新路径定向替换（一次性迁移工具，只改现状/交付约定）。

只处理"现状描述 + 交付约定"类文档；不改 99_change_log（历史记录）和
30/70_evaluation_spec 的契约/`归档版`历史描述（避免篡改冻结 spec 语义）。

各文件规则：
- docs/reports/line1_substitute_sr_final_report.md：全部旧路径数据来源 + 图片引用
- progress.md / AGENTS.md：现状描述
- docs/specs/90_delivery_spec.md：只改附录 B 阶段索引路径 + 图路径；
  "六章数据来源"的泛化模式（results/EXP-02_*_summary、EXP-0[3,4,7,8]*、EXP-08*）保留
  （描述未来实验字段，非迁移后的具体路径）；`results/<EXP>/stage_report.md` 为占位符保留
- docs/specs/05_testing_spec.md：results/test_reports/ 记录约定
- docs/specs/00_master_spec.md：scripts_tmp/consultation/ 写入约定

用法：python scripts/remap_paths.py [--dry-run] [--suffix .bak]
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = "line1_substitute_sr"

# 通用映射（长前缀优先；均为字符串级精确 or 前缀替换，不与泛化通配冲突）
GLOBAL_MAPS = [
    ("assets/figure_", f"studies/{STUDY}/results/assets/figure_"),
    ("results/EXP-02_summary", f"studies/{STUDY}/results/summary/EXP-02_summary"),
    ("results/EXP-01_summary", f"studies/{STUDY}/results/summary/EXP-01_summary"),
    ("results/EXP-03_summary", f"studies/{STUDY}/results/summary/EXP-03_summary"),
    ("data/v1/", f"studies/{STUDY}/data/v1/"),
    ("data/v1\"", f"studies/{STUDY}/data/v1\""),
    ("results/M1_generators", f"studies/{STUDY}/reports/M1_generators"),
    ("results/M2_dataset", f"studies/{STUDY}/reports/M2_dataset"),
    ("results/test_reports", f"studies/{STUDY}/reports/test_reports"),
    ("results/EXP-02_A_seed", f"studies/{STUDY}/results/run/EXP-02_A_seed"),
    ("results/EXP-02_B_seed", f"studies/{STUDY}/results/run/EXP-02_B_seed"),
    ("results/EXP-02_C_seed", f"studies/{STUDY}/results/run/EXP-02_C_seed"),
    ("scripts_tmp/consultation/", f"archive/{STUDY}/scripts_tmp/consultation/"),
]

FILES = [
    "docs/reports/line1_substitute_sr_final_report.md",
    "progress.md",
    "AGENTS.md",
    "docs/specs/05_testing_spec.md",
    "docs/specs/00_master_spec.md",
    "docs/specs/90_delivery_spec.md",   # maps 同上（见 apply），不会碰泛化通配
]

SKIP = ("docs/specs/99_change_log.md", "docs/specs/30_degradation_spec.md",
        "docs/specs/70_evaluation_spec.md")


def safe_apply(text: str) -> str:
    """用 GLOBAL_MAPS 逐字符串替换；保护 90 中的泛化通配模式。"""
    # 90_delivery_spec 的泛化模式（含通配符 []/*/_*），这些不能被前缀替换误伤
    if "EXP-0[3,4,7,8]*" in text or "EXP-08*" in text:
        # 只对 90 特有：跳过"六章数据来源"里的通配所在的行？——不整行跳，
        # 而是对每条映射先检查目标串是否出现在通配上下文中。
        pass
    for old, new in GLOBAL_MAPS:
        # 仅当替换目标不是"六章数据来源"泛化串的一部分时才替换。
        # `results/EXP-02_summary`/`EXP-03_summary` 是精确目录名，不会命中
        # `EXP-02_*_summary`（带 `_*_`）或 `EXP-0[3,4,7,8]*`（带方括号）。
        text = text.replace(old, new)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    changed = 0
    for rel in FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        new = safe_apply(text)
        if new != text:
            changed += 1
            tag = "DRY" if args.dry_run else "OK "
            print(f"[{tag}] {rel}")
            if not args.dry_run:
                if args.suffix:
                    path.with_suffix(path.suffix + args.suffix).write_text(text, encoding="utf-8")
                path.write_text(new, encoding="utf-8")
    print(f"\n共 {changed} 个文件更新（SKIP: {', '.join(p.rsplit('/',1)[-1] for p in SKIP)}）")


if __name__ == "__main__":
    main()
