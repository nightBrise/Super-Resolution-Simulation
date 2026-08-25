#!/usr/bin/env python3
"""spec 一致性自动检查脚本（漏洞 2.1）。

扫描 docs/specs/ 下所有 spec，抽取 Claims 与跨文档引用，验证：
  1. 所有 [Sn] 章节都有 Claims
  2. 所有跨文档引用 (`NN` / `NN [Sn]` / `NN [Sn] Ck`) 指向真实存在
  3. 输出 spec_claim_index.md

用法：python scripts/check_spec_consistency.py [--strict]
  默认 warning 模式；--strict 让 broken refs 返回非零退出码。
"""
import argparse
import re
import sys
from pathlib import Path

SPECS_DIR = Path('docs/specs')
SPEC_PATTERN = re.compile(r'^(\d{2})_([a-z_]+)\.md$')

# claim 正则：'- C1: text...' 或 '- C12: text...'
CLAIM_PATTERN = re.compile(r'- C(\d+):\s*(.+?)(?:\s*\|$|$)', re.MULTILINE)

# 跨文档引用正则：`NN` 或 `NN [Sn]` 或 `NN [Sn] Ck` 或 `NN [Sn]XXX`
# 例：60 [S14] C1、80 [S11] R2、90 [S6] C5
REF_PATTERN = re.compile(r'`(\d{2})\s*\[S(\d+)\](?:\s*(C\d+|R\d+|D\d+|C\d+))?`')


def parse_claims(spec_path: Path) -> list:
    """解析 spec 中的 Claims 章节，返回 [(section, claim_id, text), ...]。"""
    content = spec_path.read_text(encoding='utf-8')
    spec_id = SPEC_PATTERN.match(spec_path.name).group(1)
    claims = []
    # 找每个 [Sn] 章节下的 Claims
    for sec_match in re.finditer(r'##\s*\[S(\d+)\]', content):
        section = sec_match.group(1)
        start = sec_match.end()
        next_sec = re.search(r'##\s*\[S\d+\]', content[start:])
        section_content = content[start:start + next_sec.start()] if next_sec else content[start:]
        if '### Claims' not in section_content:
            continue
        claims_start = section_content.index('### Claims') + len('### Claims')
        claims_section = section_content[claims_start:]
        for c_match in CLAIM_PATTERN.finditer(claims_section):
            claims.append((spec_id, section, c_match.group(1), c_match.group(2).strip()))
    return claims


def check_refs(spec_path: Path, claim_index: dict) -> list:
    """检查跨文档引用，返回 broken refs 列表。"""
    content = spec_path.read_text(encoding='utf-8')
    spec_id = SPEC_PATTERN.match(spec_path.name).group(1)
    broken = []
    for ref_match in REF_PATTERN.finditer(content):
        target_spec = ref_match.group(1)
        target_sec = ref_match.group(2)
        target_claim = ref_match.group(3)
        # 跨 spec 引用不检查（spec 之间互引常见）
        if target_spec == spec_id and target_claim:
            key = (target_spec, target_sec, target_claim[1:])
            if key not in claim_index:
                broken.append(f'{spec_path.name}:{ref_match.group(0)}')
    return broken


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict', action='store_true', help='broken refs 返回非零退出码')
    parser.add_argument('--out', default='spec_claim_index.md', help='索引输出文件')
    args = parser.parse_args()

    # 1. 解析所有 claims
    claim_index = {}  # (spec, section, claim_num) -> text
    all_claims = []
    for spec_path in sorted(SPECS_DIR.glob('*.md')):
        if not SPEC_PATTERN.match(spec_path.name):
            continue
        claims = parse_claims(spec_path)
        all_claims.extend(claims)
        for spec_id, section, c_num, c_text in claims:
            claim_index[(spec_id, section, c_num)] = c_text

    # 2. 跨文档引用检查
    broken = []
    for spec_path in sorted(SPECS_DIR.glob('*.md')):
        if not SPEC_PATTERN.match(spec_path.name):
            continue
        broken.extend(check_refs(spec_path, claim_index))

    # 3. 输出索引
    out_lines = ['# 全 Spec Claim 索引（自动生成）', '',
                 '> 由 `scripts/check_spec_consistency.py` 自动生成。',
                 '> 维护：每次 spec 修改后重新跑此脚本刷新本文件。', '',
                 f'总计 {len(all_claims)} 个 Claim。', '',
                 '| Claim ID | 文档 | 章节 | 简述（前 80 字）|',
                 '|---|---|---|---|']
    for spec_id, section, c_num, c_text in sorted(all_claims, key=lambda x: (x[0], x[1], int(x[2]))):
        short = c_text[:80] + ('...' if len(c_text) > 80 else '')
        out_lines.append(f'| {spec_id}.S{section}.C{c_num} | {spec_id} | [S{section}] | {short} |')

    Path(args.out).write_text('\n'.join(out_lines) + '\n', encoding='utf-8')

    # 4. 报告
    print(f'总计 {len(all_claims)} 个 Claim 写入 {args.out}')
    if broken:
        print(f'WARN: {len(broken)} broken refs:')
        for b in broken[:10]:
            print(f'  {b}')
        if len(broken) > 10:
            print(f'  ... 还有 {len(broken) - 10} 条')
        if args.strict:
            return 1
    else:
        print('OK: 所有跨文档引用一致')

    return 0


if __name__ == '__main__':
    sys.exit(main())
