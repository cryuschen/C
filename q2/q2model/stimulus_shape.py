"""Extract the pictured cue and encode its mirrored edge arrangement.

The Word problem embeds one right-pointing screenshot. The left cue is created by
reflecting only the triangle around the shared fixation circle. This preserves
the observed cue size/color and separates it from the common circle input.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, maximum_filter


DOCX = '服务于脑机接口与精神性疾病诊断的脑电图计算模型.docx'
MEDIA = 'word/media/image12.png'


def cue_images(root: Path):
    with ZipFile(Path(root) / DOCX) as archive:
        raw = archive.read(MEDIA)
    image = np.asarray(Image.open(BytesIO(raw)).convert('RGB')).astype(np.float64)
    blue = np.maximum(0., image[:, :, 2] - np.maximum(image[:, :, 0], image[:, :, 1])) / 255
    mask = blue > .2
    components, count = label(mask)
    pieces = sorted([(int(np.sum(components == j)), j) for j in range(1, count + 1)], reverse=True)
    if len(pieces) < 2 or pieces[0][0] < 80 or pieces[1][0] < 40:
        raise ValueError('Cannot identify the triangle and fixation circle in the problem image')
    top, bottom = sorted((pieces[0][1], pieces[1][1]),
                         key=lambda j: np.mean(np.nonzero(components == j)[0]))
    triangle = np.where(components == top, blue, 0.)
    circle = np.where(components == bottom, blue, 0.)
    center_x = int(round(np.mean(np.nonzero(circle)[1])))
    left_triangle = np.zeros_like(triangle)
    ys, xs = np.nonzero(triangle)
    reflected_x = 2 * center_x - xs
    inside = (reflected_x >= 0) & (reflected_x < triangle.shape[1])
    left_triangle[ys[inside], reflected_x[inside]] = triangle[ys[inside], xs[inside]]
    if not np.isclose(triangle.sum(), left_triangle.sum(), rtol=0, atol=1e-10):
        raise ValueError('Mirroring lost triangle pixels')
    return dict(right=triangle + circle, left=left_triangle + circle,
                right_triangle=triangle, left_triangle=left_triangle,
                circle=circle, center_x=center_x,
                source_size=image.shape[:2], source_media=MEDIA)


def oriented_edges(image: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(image, .8)
    gx = gaussian_filter(smooth, 1., order=(0, 1))
    gy = gaussian_filter(smooth, 1., order=(1, 0))
    angles = np.arange(8) * np.pi / 8
    return np.stack([np.abs(gx * np.cos(a) + gy * np.sin(a)) for a in angles])


def sites(template: np.ndarray, n: int = 18) -> list[tuple[int, int, int]]:
    response = oriented_edges(template)
    peak = response.max(0)
    ranked = np.argsort(peak.ravel())[::-1]
    selected = []
    width = template.shape[1]
    for flat in ranked:
        y, x = divmod(int(flat), width)
        if peak[y, x] < .025 * peak.max():
            break
        if any((y - yy) ** 2 + (x - xx) ** 2 < 9 for yy, xx, _ in selected):
            continue
        orientation = int(np.argmax(response[:, y, x]))
        selected.append((y, x, orientation))
        if len(selected) == n:
            break
    if len(selected) < n:
        raise ValueError('Too few template edge sites')
    return selected


def shape_inputs(cues: dict):
    names = ('left', 'right')
    triangle = (cues['left_triangle'], cues['right_triangle'])
    templates = [sites(item) for item in triangle]
    # One-pixel tolerance; two pixels would pool across much of this 13-pixel cue.
    pooled = [maximum_filter(oriented_edges(item), size=(1, 3, 3))
              for item in triangle]
    activation = np.zeros((2, 2))
    for i, response in enumerate(pooled):
        for j, selected in enumerate(templates):
            values = np.array([response[o, y, x] for y, x, o in selected])
            activation[i, j] = np.exp(np.mean(np.log(values + 1e-5)))
    normalization = np.maximum(np.diag(activation), 1e-10)
    activation /= normalization[None, :]
    # A shared, direction-neutral fixation input is supplied separately.
    common = np.ones(2)
    return dict(inputs=np.column_stack([activation, common]),
                activation=activation, sites=np.array(templates), names=names,
                selectivity=(activation[:, 1] - activation[:, 0]) /
                            (activation[:, 1] + activation[:, 0] + 1e-9))
