"""Immutable, self-contained frozen fits for reconstruction and audits."""
from pathlib import Path
import json
import shutil
from .common import ROOT, sha

REFERENCE = ROOT / 'q3_result/reference'
MANIFEST = Path(__file__).with_name('reference_manifest.json')


def verify_reference(directory=None):
    directory = REFERENCE if directory is None else Path(directory)
    expected = json.loads(MANIFEST.read_text(encoding='utf8'))['retained_files']
    actual = {p.name for p in directory.iterdir() if p.is_file()}
    if actual != set(expected):
        raise ValueError(f'冻结参考文件缺失或多出：{sorted(actual ^ set(expected))}')
    for name, digest in expected.items():
        if sha(directory / name) != digest:
            raise ValueError(f'冻结参考已改变：{name}')
    return expected


def prepare_reference(out):
    """Copy only to a new output; never overwrite frozen inputs."""
    expected = verify_reference()
    destination = Path(out) / 'reference'
    if destination.resolve() != REFERENCE.resolve():
        if destination.exists():
            verify_reference(destination)
        else:
            shutil.copytree(REFERENCE, destination)
    return expected
