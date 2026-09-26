#!/usr/bin/env python3
"""Create/check a portable SHA256 inventory and export the final folder."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT / '验收与清单/最终文件清单.csv'
EXCLUDED = {'.venv', '__pycache__', '.mplcache', '.pytest_cache', '.git'}


def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and p != INVENTORY
                  and not (set(p.relative_to(ROOT).parts) & EXCLUDED)
                  and not p.name.startswith('fontlist-v') and p.suffix != '.pyc')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_manifest():
    with INVENTORY.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['路径', '字节数', 'SHA256'])
        for p in files():
            writer.writerow([p.relative_to(ROOT).as_posix(), p.stat().st_size, sha(p)])


def check():
    with INVENTORY.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    expected = {r['路径'] for r in rows}
    actual = {p.relative_to(ROOT).as_posix() for p in files()}
    errors = [f'文件列表差异：{sorted(expected ^ actual)}'] if expected != actual else []
    for r in rows:
        path = ROOT / r['路径']
        if not path.is_file() or path.stat().st_size != int(r['字节数']) or sha(path) != r['SHA256']:
            errors.append(r['路径'])
    if errors:
        raise RuntimeError('交付清单核对失败：' + '\n'.join(errors))
    print(json.dumps({'status': 'passed', 'files_verified': len(rows), 'bytes': sum(int(r['字节数']) for r in rows)}, ensure_ascii=False))


def package():
    check()
    target = ROOT.parent / '三题最终版.zip'
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in [*files(), INVENTORY]:
            z.write(p, (Path('最终版') / p.relative_to(ROOT)).as_posix())
    with zipfile.ZipFile(temporary) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('ZIP CRC核验失败：' + bad)
    temporary.replace(target)
    target.with_suffix('.zip.sha256').write_text(f'{sha(target)}  {target.name}\n', encoding='utf-8')
    print(f'已打包并完成ZIP CRC核验：{target} ({target.stat().st_size} bytes)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--zip', action='store_true')
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
    if args.check or not (args.write_manifest or args.zip):
        check()
    if args.zip:
        package()
