# pdf-toolbox · PDF 工具箱

免费的本地 PDF 编辑工具。核心能力是**直接修改 PDF 原文**——点一下就能改字，
改长了后面的内容自动重排，而不是贴个文本框盖在上面。

扫描件（没有文字层的图片型 PDF）也能改：用 OCR 建立可编辑的文本层，
修改时按原字的位置与字体重绘。

所有处理都在本机完成，不联网、不上传文件。

## 这个项目有什么不同

大多数"PDF 编辑"工具要么在原文上盖一个文本框，要么整页重建版式。
这个项目的做法是**在字符层面操作**：

1. **字符级坐标表 → 规则反推结构 → 改结构 → 重算坐标**。
   每个字符的包围盒、基线、字体、字号、颜色都从 PDF 里精确取出，
   再聚合成行、块、流、图层五层结构。
2. **最小位移原则**：重排后位置没变的元素**保留原文件字节不动**，
   只重画真正需要移动的。保真度因此从"全页重建"抢回"局部重建"。
3. **活字印刷式编辑**：段落内已有的字直接复用原字形（矢量或像素），
   没出现过的字才走字体匹配。这是图像型 PDF 保真的关键机制。
4. **图层分离**：页眉页脚、页码、水印、印章钉死原位，永不参与重排 ——
   改字不会把印章顶飞。

## 下载即用（推荐给不写代码的人）

到本仓库的 **[Releases](../../releases)** 页下载 `PDF工具箱.exe`，双击运行即可 ——
单文件、免安装、不需要 Python 环境。Windows 10/11 64 位，启动约 3 秒。

> 未签名的单文件 exe 可能被杀毒软件拦一次，选择"仍要运行"即可。
> 源码全部在本仓库，可以自行核对，也可以自己打包：
> `python -m PyInstaller --clean pdfbox.spec`

打包进去的东西：OCR 模型（15.6MB，扫描件功能靠它）、PySide6、PyMuPDF、
以及转换用的 docx/xlsx/pptx 库。`Office → PDF` 需要本机装有 Office，
没装时那条菜单会明确报错，不影响其它功能。

## 从源码运行（开发者）

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install PyMuPDF fontTools numpy Pillow PySide6 rapidocr-onnxruntime
```

编辑中文文档需要系统装有中文字体（Windows 自带宋体/黑体/微软雅黑，通常无需额外安装）。
格式转换里"Office → PDF"需要本机装有 Office（走 COM 自动化）。

## 快速开始

```bash
python -m pdfbox                    # 启动桌面界面
python -m pdfbox --help             # 看全部子命令
```

界面里：工具栏那一排是**编辑动作**（打开/保存/撤销/缩放/显示结构框），
菜单栏里是**文档级动作**：

| 菜单 | 内容 |
|---|---|
| 页面 | 页面列表、删除页、上移下移、调整顺序、旋转、提取、插入、合并 |
| 工具 | 分析图像占用、压缩文档、转换为 Word/Excel/PPT、导出图片、其他文档转 PDF |

### 命令行

```bash
# 看文档类型、页数、每页块数
python -m pdfbox --cli info 文件.pdf

# 改字：把匹配到的块整块替换；--replace-only 只替换片段
python -m pdfbox --cli edit 合同.pdf --match "金额" --to "¥5000" -o 改后.pdf

# 页面：删除 / 重排 / 旋转 / 提取 / 插入 / 合并
python -m pdfbox --cli pages 文件.pdf --delete 2,5-7 -o 输出.pdf
python -m pdfbox --cli pages 文件.pdf --order 5,1-4 -o 输出.pdf

# 压缩（先 --analyze 看每张图的有效 DPI）
python -m pdfbox --cli compress 扫描件.pdf --analyze
python -m pdfbox --cli compress 扫描件.pdf --preset balanced

# 格式转换
python -m pdfbox --cli convert 文件.pdf --to docx --verify
```

## 保真度：哪些是真正无损的

| 元素 | 保真级别 |
|---|---|
| 固定层（页眉页脚/水印/印章/背景） | 字节级 100%，永不参与重排 |
| 重排中位置未变的流式元素 | 字节级 100%（最小位移原则） |
| 被重排顶下去的元素 | 按原始几何高保真重建，非字节级 |
| 图片型 PDF 的编辑 | 覆盖式重绘，原像素无法真正移除 |

第三行是这套方案的固有成本，但它的偏差是**局部且可预期**的
（只在你改动的下游），不像整页重建那样全页漂移。

## 验证

这个项目的每个关键结论都有可复跑的实测支撑，而不是"看起来对"。

```bash
python -m tools.regression              # 全量回归，12 项
python tools/make_form_sample.py        # 生成合成测试样本
python tools/make_truth_sample.py       # 生成带标准答案的逐字框样本
python tools/verify_char_boxes.py       # 逐字框精度（有真值）
python tools/locate_chars.py --calibrate  # 标定独立测量工具的精度
```

**逐字框精度**（用户要求"上下左右边框都贴着单个字本身"）：

| 指标 | 值 |
|---|---|
| 真值样本 138 个字，四边误差 ≤1pt | **138 / 138（100%）** |
| 四边均值 | 中位 0.29pt、90 分位 0.42pt、最大 0.80pt |

真值环境的做法：用已知字体在已知坐标写字，逐字单独渲染量出真实墨迹框，
再光栅化成 1.55 像素/pt 的位图 —— 既像扫描件，又知道真相。

**压缩**：逐张图按**有效 DPI**（铺在页面上的每英寸像素数）决策，
把"降采样损失"与"编码损失"分开实测，三层校验（单图两种损失 /
整页渲染逐像素对比 / 文字层字节指纹）。实测 balanced 档
扫描件 2030→148KB（7%）、整页 PSNR 44.1dB，文字层字节一致。

**格式转换**：PDF→Office 用本工程自己的版面模型重建（每处属性都能追到源 PDF
的哪个元素）；Office→PDF 交给本机 Office 渲染。PDF→Word 带页数自校正
（构建→用 Word 实际渲染量页数→按比例微调字号重建）。

## 已知限制

- **流式排版无法复现绝对定位版面**：同一行并排的文本块、分栏、页眉页脚、
  叠放的图片，PDF→Word 时只能顺序堆叠。需要逐页完全一致时请直接用
  「编辑 PDF」改字，不要中转 Word。
- **短行（页脚那种）的字格推算偏弱**：字符少、字宽比例约束不足，
  实测页脚 `第X页/共Y页` 这类行的框会偏宽。正文不受影响。
- **低分辨率扫描件上，字间空隙不足一个源像素时**，边界只能靠字体度量推断，
  这类字的位置精度受限于扫描分辨率本身。
- **扫描件编辑做不到字节级无损**（覆盖式重绘的固有限制）。
- **加密 PDF** 需要密码。

## 测试样本

`tests/fixtures/` 里的样本**全部是合成的**（`tools/make_form_sample.py` 生成），
内容虚构，不含任何真实文档 —— 真实文档常含个人信息，不能随源码分发。
合成样本刻意保留了真实扫描件的技术特征：1.55 像素/pt 的分辨率、
扫描噪声、表单填写栏的下划线、中西混排、密集数字串、全角标点。

## 目录结构

```
pdfbox/
├── core/                    核心引擎（与界面无关，CLI 和 GUI 共用）
│   ├── model.py             五层结构模型（字符/行/块/流/图层）
│   ├── structure.py         文本型 PDF 的结构提取
│   ├── classify.py          规则引擎：类型判定 + 图层判定 + 置信度
│   ├── fonts.py             字体匹配与度量验证
│   ├── editor.py            擦除 + 重画内核
│   ├── reflow.py            重排引擎 + 最小位移原则
│   ├── raster.py            图片型 PDF 的墨迹实测与覆盖重绘
│   ├── raster_typeset.py    图片型 PDF 的活字印刷式编辑
│   ├── scan_match.py        扫描件模板匹配（独立于切分的测量手段）
│   ├── ocr.py               OCR 引擎与缓存
│   ├── session.py           文档会话（统一两条路径 + 撤销 + 页面操作）
│   ├── pages.py             页面删除/重排/旋转/提取/插入/合并
│   ├── compress.py          图像级压缩（两种损失分别实测 + 三层校验）
│   ├── convert.py           PDF↔Word/Excel/PPT/图片、Office→PDF
│   └── verify*.py           像素比对、固定层校验、转换质量验证
├── ui/                      桌面界面（PySide6）
├── cli.py                   命令行工具
└── __main__.py              python -m pdfbox 入口
tools/                       样本生成器与验证工具
tests/fixtures/              合成测试样本
```

## 许可证

[MIT](LICENSE)
