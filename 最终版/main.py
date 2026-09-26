#!/usr/bin/env python3
"""Portable entry point for the organized final deliverable."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE / '完整工程'


def interpreter():
    if sys.prefix != sys.base_prefix:
        return Path(sys.executable)
    for base in (HERE, PROJECT, HERE.parent):
        for suffix in ('bin/python', 'Scripts/python.exe'):
            candidate = base / '.venv' / suffix
            if candidate.is_file():
                return candidate
    return Path(sys.executable)


def main():
    parser = argparse.ArgumentParser(description='三题最终版：默认验收；--stage all 完整重新计算。可从任意目录启动。')
    parser.add_argument('--stage', choices=('verify', 'all', 'q1', 'q2', 'q3', 'report', 'figures', 'integrity'), default='verify')
    args = parser.parse_args()
    if args.stage == 'integrity':
        return subprocess.call([str(interpreter()), '-B', str(HERE / 'release.py'), '--check'], cwd=HERE)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / 'c-final-mpl'))
    command = ['Q1/main.py'] if args.stage == 'figures' else ['run_optimization.py', '--stage', args.stage]
    print(f'工程目录：{PROJECT}\n解释器：{interpreter()}\n阶段：{args.stage}', flush=True)
    result = subprocess.run([str(interpreter()), '-B', *command], cwd=PROJECT, env=env)
    if result.returncode:
        print('运行未通过，请查看完整工程/优化计划及各题运行日志。', file=sys.stderr)
        return result.returncode
    if args.stage == 'all':
        result = subprocess.run([str(interpreter()), '-B', 'Q1/main.py'], cwd=PROJECT, env=env)
        if result.returncode:
            return result.returncode
    print('阶段完成。若更新了输出，可运行 release.py --write-manifest 更新交付文件清单。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
