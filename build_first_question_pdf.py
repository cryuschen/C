#!/usr/bin/env python3
"""将第一问 V7 已复核数值和图表排成可提交的独立 PDF。"""
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / 'eeg_v7_results'
OUT = ROOT / 'output/pdf/第一问_脑电预处理与视觉响应建模_终稿.pdf'
FONT = '/usr/share/fonts/truetype/winfonts/NotoSansSC-VF.ttf'
pdfmetrics.registerFont(TTFont('CN', FONT))
NAVY = colors.HexColor('#16344a')
TEAL = colors.HexColor('#007f89')
MUTED = colors.HexColor('#546779')
PALE = colors.HexColor('#e9f4f5')
RULE = colors.HexColor('#d8e2e7')


def sty(name, size=10, leading=None, color=NAVY, space=0, align=TA_LEFT):
    return ParagraphStyle(name, fontName='CN', fontSize=size,
                          leading=leading or size * 1.55, textColor=color,
                          spaceAfter=space, alignment=align, wordWrap='CJK')


TITLE = sty('title', 23, 34, NAVY, 16)
SUB = sty('sub', 13, 21, TEAL, 13)
H1 = sty('h1', 15, 23, NAVY, 10)
H2 = sty('h2', 11, 17, TEAL, 7)
BODY = sty('body', 10.4, 18, NAVY, 8)
SMALL = sty('small', 9, 15, MUTED, 6)
CAP = sty('cap', 9, 15, MUTED, 8)
CELL = sty('cell', 9.1, 14, NAVY)
CELL_HEAD = sty('cell-head', 9.1, 14, colors.white)


def p(text, style=BODY):
    return Paragraph(escape(text).replace('\n', '<br/>'), style)


def table(rows, widths, header=True):
    data = []
    for i, row in enumerate(rows):
        data.append([p(str(v), CELL_HEAD if i == 0 and header else CELL) for v in row])
    t = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1 if header else 0)
    commands = [
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, -1), (-1, -1), .5, RULE),
    ]
    if header:
        commands.append(('BACKGROUND', (0, 0), (-1, 0), NAVY))
        if len(rows) > 1:
            commands.append(('ROWBACKGROUNDS', (0, 1), (-1, -1),
                             [colors.white, colors.HexColor('#f4f8fa')]))
    t.setStyle(TableStyle(commands))
    return t


def figure(path, width=172 * mm, max_height=202 * mm):
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(width / w, max_height / h)
    return Image(str(path), width=w * scale, height=h * scale, hAlign='CENTER')


def bullets(*items):
    return [p('• ' + item) for item in items]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.line(19 * mm, 18 * mm, 191 * mm, 18 * mm)
    canvas.setFont('CN', 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(19 * mm, 13 * mm, '第一问：脑电预处理与视觉响应建模  |  V7 开发集实验')
    canvas.drawRightString(191 * mm, 13 * mm, str(doc.page))
    canvas.restoreState()


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=19 * mm, rightMargin=19 * mm,
                            topMargin=19 * mm, bottomMargin=24 * mm,
                            title='第一问：脑电预处理与视觉响应建模',
                            author='第一问建模实验', subject='Fz/F3/F4，任务一与任务二')
    story = []
    add = story.append

    add(p('第一问｜脑电预处理、伪影校正与有效视觉响应拟合', TITLE))
    add(p('基于 Fz / F3 / F4 原始记录的可复核建模实验', SUB))
    add(p('研究目标', H1))
    add(p('针对受试者 A、B 的项目一与项目二，提取视觉提示事件，筛查严重坏试次，对原始额区脑电进行有限校正，并分别刻画左、右提示下的平均响应与曲线。评价同时考察伪影代理量和方向相关波形的保留。'))
    add(p('核心结果', H1))
    add(table([
        ['数据组', '可用试次', '代理 MAE 降幅', 'SNR 代理增量', '左右差分幅度比'],
        ['A · 项目一', '90 / 100', '10.0%', '+0.217 dB', '0.934'],
        ['A · 项目二', '95 / 100', '8.2%', '+0.027 dB', '0.952'],
        ['B · 项目一', '94 / 100', '5.4%', '+0.103 dB', '0.942'],
        ['B · 项目二', '92 / 100', '9.8%', '+0.481 dB', '0.813'],
    ], [86, 78, 104, 106, 114]))
    add(Spacer(1, 8 * mm))
    add(p('解读原则', H2))
    story += bullets(
        'V7 四组的代理 MAE 均低于预处理，SNR 代理量均提高；幅度改善有限，不作“完全去伪影”解释。',
        '左右差分幅度比接近 1 仅表明与处理前更相近；处理前可能包含方向相关眼动。',
        '四组 1 倍、2 倍半合成扰动的恢复误差比 V6 略低；0 倍和轻扰动仍出现背景改动。',
    )
    add(p('结论定位', H2))
    add(p('这是同一批四组数据上的开发集折外实验，不是独立受试者泛化结果。数据只有三个额区通道，缺少眼电和独立无伪影脑电真值，不能证明神经源无损恢复。'))
    add(PageBreak())

    add(p('1  数据来源与处理流程', H1))
    add(table([
        ['环节', '实施与可复核边界'],
        ['输入', '只使用四份原始 MAT 文件中的 Fz、F3、F4 与 VisCue（−1 左、+1 右）；不用设备已处理的 Decon 通道。'],
        ['固定预处理', '256 Hz；连续信号 60 Hz 陷波、0.1–30 Hz 零相位带通；−250–0 ms 中位数基线；试次窗约 −250–797 ms。'],
        ['质量筛查', '沿用原工程阈值筛查严重极值、峰峰值和跳变；各阶段共用同一可用试次。'],
        ['折外建模', '五折按方向分层；每折仅用训练试次建立低伪影方向模板、评估候选、选择 V7 分支。'],
        ['响应与拟合', '分别计算左右方向 Fz/F3/F4 的 ERP；固定 50 ms 内部结点的三次样条描述 50–750 ms 波形。'],
    ], [86, 402]))
    add(Spacer(1, 7 * mm))
    add(p('V7 校正定义', H2))
    add(p('设 x 为预处理试次，v4 为该外折训练选出的 V4 候选输出，δ = x − v4。若训练折选中“保守双分量”，V7 = x − 0.25δ + 0.10m_y，其中 m_y 为同方向训练试次的平均 δ；其他候选则 V7 = x − 0.15δ。训练均值只来自外折训练集，不以测试试次重估。'))
    add(p('0.15、0.25、0.10 是在查看四组开发结果后选定的保守比例；本 PDF 中的折外数字不应解释为独立调参验证。'))
    add(p('指标口径', H2))
    add(p('代理参考是当前外折训练集中同方向低伪影半数试次的逐点中位数。MAE 在 250–500 ms 比较测试 ERP 与对应训练代理参考；SNR 代理量用 0–500 ms 的 ERP 功率与试次间残差功率之比。二者都不是独立清洁脑电真值。'))
    add(p('仅当 250–500 ms 内存在合格局部正峰才报告峰潜伏期。三个电极均位于额区，因此正峰只称“额区正向响应”，不声称在中央或顶区观测到典型 P300。'))
    add(PageBreak())

    add(p('2  校正幅度与特征保留的联合评价', H1))
    add(figure(RESULT / '汇总与说明/去噪与特征保留联合评价.png', max_height=190 * mm))
    add(Spacer(1, 3 * mm))
    add(p('图 1｜每行对应一组数据。左列为相对预处理的代理 MAE 降幅，中列为 SNR 代理值增量，右列为 250–500 ms 左减右 ERP 差分幅度比。三个量有不同含义，不合成单一综合分数。B 项目二的差分保留比仅 0.813，仍需把特征损失列为代价。', CAP))
    add(PageBreak())

    add(p('3  分方向三通道响应：受试者 A · 项目一', H1))
    add(figure(RESULT / '受试者A_项目一/左右刺激三通道ERP.png', max_height=180 * mm))
    add(Spacer(1, 3 * mm))
    add(p('图 2｜左、右刺激分列；每通道内纵轴共享。灰蓝虚线为预处理，浅紫点线为 V6，青色实线为 V7。浅青色阴影是 V7 逐时间点试次重采样 95% 区间，不是整条曲线的同时置信带。部分额区响应有明显慢变化；曲线相似不能单独证明眼动已除净。', CAP))
    add(PageBreak())

    add(p('4  高慢变场景：受试者 B · 项目二', H1))
    add(figure(RESULT / '受试者B_项目二/左右刺激三通道ERP.png', max_height=180 * mm))
    add(Spacer(1, 3 * mm))
    add(p('图 3｜项目二的晚期缓慢偏移大，V7 对幅度的调整可见，但无法仅凭三个额区电极判断这是眼动伪影还是神经慢电位。本组代理参考相关仍为负值（均值 −0.086），因此不能称其形状已恢复到可信真值。', CAP))
    add(PageBreak())

    add(p('5  试次分布与曲线拟合检查', H1))
    add(figure(RESULT / '受试者B_项目二/Fz热力图_细节视图.png', max_height=100 * mm))
    add(p('图 4｜Fz 试次×时间热力图，三阶段共用色域；细节图按图内所示分位截色约 2% 像素，用于看弱结构。完整色域图另存于结果目录，判断大伪影时应同时查看。', CAP))
    add(figure(RESULT / '受试者A_项目一/分方向样条拟合与残差.png', max_height=100 * mm))
    add(p('图 5｜六条方向×通道 ERP 的三次样条及残差。拟合残差衡量平滑描述程度，不是未知试次预测误差；有正向面积但缺少窗内局部峰时，不填 0 ms 峰潜伏期。', CAP))
    add(PageBreak())

    add(p('6  已知注入扰动的闭环检查', H1))
    add(figure(RESULT / '半合成验证/四组多幅度半合成验证.png', max_height=150 * mm))
    add(p('图 6｜在外折测试试次注入眨眼样、扫视样、运动样扰动，污染前的实测预处理波形为计算目标。各组除以自身背景 RMS 后显示完整试次恢复误差。0 倍处未校正误差为零，V7 的非零值表示背景被改动。', CAP))
    add(table([
        ['数据组', '1 倍：V6 → V7', '2 倍：V6 → V7', '0 倍：V7'],
        ['A · 项目一', '0.338 → 0.337', '0.631 → 0.623', '0.124'],
        ['A · 项目二', '0.673 → 0.662', '1.322 → 1.288', '0.100'],
        ['B · 项目一', '0.470 → 0.464', '0.917 → 0.897', '0.090'],
        ['B · 项目二', '0.599 → 0.588', '1.177 → 1.147', '0.110'],
    ], [110, 148, 148, 82]))
    add(p('A 项目一在 0.5 倍轻扰动时 V7 为 0.206，高于未校正的 0.174；不能说所有污染强度都获益。重复注入的场景不是独立受试者，污染前实测背景也可能已有伪影。', SMALL))
    add(PageBreak())

    add(p('7  结果边界与复现说明', H1))
    story += bullets(
        '赛题要求分别处理两个项目的视觉响应、抑制运动和眨眼等伪影并尽量保留波形特征。这里提供了额区分方向响应的可复核实现；方向与视觉形状在现有标记下可能混杂，不能证明更细的形状神经表征。',
        '任务一和任务二都呈现试次级大振幅与晚期慢变。V3/V4 可大幅降低代理误差，却更明显改变左右差分；V7 在噪声代理量和处理前形状之间作保守取舍。',
        '真实数据没有 EOG 或无伪影 EEG 真值。预处理、V3、V4、V6、V7 的数字在同一试次与同一训练代理参考上计算，但这个代理参考参与模型构造。',
        '四组结果经过开发过程观察；五折只能保证单次外折测试试次不参与该折模板和策略估计，不能消除开发集调参偏倚，也不能推断人群或临床效果。',
        '项目一/二的同受试者曲线对比仅描述条件差异；已知 VisCue 的离线曲线不是未知刺激分类或诊断模型。',
    )
    add(Spacer(1, 5 * mm))
    add(p('提交与复核文件', H2))
    add(p('源码：EEG_P300_artifact_correction_v7.py；依赖：V3、V4 模块与 requirements-v4.txt；四份 MAT：data/；实验报告：eeg_v7_results/第一问终稿实验报告.md；指标定义：指标字典_V7.md。每组有试次审计、逐折参考清单、逐方向指标、重采样区间和可复核波形；运行清单保存源码、输入和数值 CSV 的 SHA256。'))
    add(p('完整运行：python EEG_P300_artifact_correction_v7.py --output eeg_v7_results。复算测试：python -m unittest discover -s tests -v。建议以仓库 README_V7.md 指定的 Python 环境为准。', SMALL))
    add(p('方法依据', H2))
    add(p('Tanner 等关于不恰当高通滤波对认知 ERP 的影响：PMC4506207。Luck 等关于 ERP 伪影校正和剔除的评价框架：PMC11021170。此处引用用于说明滤波和评价边界，不作为本数据集有效性的外部验证。', SMALL))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == '__main__':
    build()
