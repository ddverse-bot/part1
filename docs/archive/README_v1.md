# octreg v1 — OCT 组织块 → MRI 的无标注配准（2026-08-20）

代码：`octreg/octreg/`（包）与 `octreg/scripts/`（流程脚本）。运行环境：AutoDL 服务器 `~/autodl-tmp/oct-mri-registration/`（conda env `octmri`，RTX 4090 24 GB）。数据：Costantini 2023（DANDI:000026）多个 subject 的 OCT block + 全半球 ex-vivo MRI（见 §2）。结果、图、变换在服务器 `work/runs/`，本地 `octreg/results/final/` 只放 JSON 摘要和图。

这一版的原则：**只用真实数据，不用任何标注、mask 或位置先验；方法保持最简，只保留逻辑上必要、实验上有效的东西**。手工标注（MRI 血管、GM/WM 标签）只用于评价。

## 0. 2026-09-13 更新：v1.1 与 I58 脑干对的结论

完整技术记录在 [`docs/v11/REPORT_I58_v11.md`](docs/v11/REPORT_I58_v11.md)（探针记录 P1–P10、规范、审查记录、证据图都在 `docs/v11/`）。要点：

- **v1.1 新增（默认关闭或按数据规则触发，v1 默认路径逐位不变，DANDI I46/I55 回归 `regress.py` 各 26 项 0 失败）**：`prep_subject.py --oct-mask auto`（纹理 watershed 标本 mask，琼脂糖包埋块专用：I58 由 28.9 → 18.0 cm³，MRI 组织 13.3）、`--destripe on`（自动检测切片轴并沿该轴归一化卷积平场：I58 的切片轴是数组轴 2，周期 0.30 mm = 15 层）、`--skip-vessels`；`register.py --init-transform`（位置先验路径，PI 8/23 明确先验是有的）、血管阶段验收门/自动跳过、`--fine on`（符号校正的 3 mm 拉平 masked NCC，以起始位姿为基准优化增量，带重启/多相似度/split-half/逆一致性/U 验收门）；`qc_fine.py`（边界一致、MI 景观、棋盘格）、`regress.py`。**注意 `--fine on` 未通过皮层验证**：在 I46/I55 上它通过了自己全部验收门（8/8 重启收敛、U 0.4–1.1 mm），最终位姿却偏离标注最优 0.7–1.0 mm（I46 血管中位距离 123 → 165 µm、深度拉伸 1.219 → 1.31；I55 GM Dice 0.930 → 0.916），所以保持默认关闭，自洽性 U 只能作为自洽性而不是精度来报告；任何新的 fine 目标函数都必须先在 I46/I55 上通过 `regress.py --tol fine`。 v1.1b（本次提交）只加可选项，默认仍等于 v1.1：`prep_subject.py --mask-fill-pockets/--mask-fill-holes/--mask-rim-required/--mask-erode-mm` 与 `oct150_mask_core.npy`；`register.py --fine-fixed-mask`（P7 的固定点集 + 冻结权重）、`--fine-fixed-weight`、`--fine-restart-mm/-deg/-logscale/-tol-mm`、`--fine-exclude-fov-mm`、`--fine-dof {rigid,similarity,affine}`；I58 上 CPU 验证的保守 polish 仍不收敛（报告附录 C.2），I46/I55 上的 GPU 校准（`scripts/run_v11_polish.sh`）待跑。
- **I58 结果**：prep 全部成立；本体级结构位姿跨种子/去条纹可复现（0.31 mm/1.1°、0.35 mm/1.4°），边界一致中位数正向 2.02 mm / 反向 1.05 mm；fine 阶段六次运行全被自己的验收门拒绝（8 次重启 0 次收敛）。**原因是数据不是算法**：MRI 裁剪与 OCT 块不是同一组刚性摆放的组织——MRI 小脑叶在 OCT 里没有刚性对应（反向全局搜索分低于极性翻转空模型），而 OCT 深端的折叠块经 0.04 mm 特写证实是撕裂的小脑叶片、位于本体另一侧；OCT mask 还含约 5.5 cm³ 条纹琼脂糖壳与片层空腔；MRI 裁剪在三个面截断标本。轮廓距离、MI、分块配准、对称伙伴都试过，均不能钉住位姿。这对数据无标注能支持的是本体级位姿（约 2 mm 边界一致、数度旋转模糊），不是亚毫米。
- **2026-09-14：Xiangrui 数据一键跑通**：服务器上 `bash octreg/scripts/run_xiangrui_i58.sh` 从两份原始 NIfTI 出发完成 prep → 配准（不含 fine）→ QC → 按原始 NIfTI 坐标系导出（4×4、FreeSurfer LTA、双向叠加图）→ summary.json，27.6 min，内存峰值 32 GB（60 s 采样），结果与 R5 逐位一致。强度重归一化/tile 平场、排除小脑与深端叶片两条路都按事先写好的判据测过，均未采用：前者在 P_R5 处的局部目标上看不到可测收益（没有重跑搜索和重启），后者让位姿跳进身体边界更差的盆地（代码以补丁存档在 `docs/v11/p11/`，未合并）。**OCT 相对 MRI 的手性数据定不下来**：register.py 找到的另一手性位姿正向外轮廓更好（rim 正向 1.57 vs 2.02 mm，反向 1.08 vs 1.05 mm，y+ 面 1.48 vs 4.96 mm），但内部类别相关更低（NCC 0.113 vs 0.124，重启 1/12 vs 6/12）；而同一手性下另一个位姿（离它 30.6 mm，各向同性缩小 5.4 %）NCC 为 0.136，所以类别相关也分不出手性。按文件头字面理解（OCT LPI，是 FreeSurfer matlab 写的简单翻转，MRI RIA），报告位姿是镜像、另一候选是正常手性，但前提是两个文件头都记录了真实物理方向，这一点没有核实。两套候选都已导出，需人眼在 freeview 里判断或给 3 对地标。详见报告附录 C.4。
- **需要 Xiangrui 回答**：未裁剪的 MRI 或裁剪框来源；解剖/包埋顺序（小脑是否被分离过）；OCT 哪个数组轴是光学深度轴、是否做过折射率深度校正（所有度量都偏好沿数组轴 0 拉伸 1.1–1.4×）；本体上约 20 对地标点用于 TRE。
- **README 其余部分描述 v1（2026-08-20），仍然有效；§4/§6 提到的 `scripts/validation_i46/`、`scripts/legacy/` 目录并不存在，I46 验证脚本平铺在 `scripts/` 下。**

## 1. 方法（四步）

```text
OCT (任意分辨率、任意朝向、任意手性)          MRI (任意分辨率；整半球或裁剪块)
  ├─ 逐层 slab 归一化 → 池化到 0.15 mm          ├─ 组织 mask（空气/液体 vs 组织的直方图谷）
  ├─ 组织 mask → 两类强度划分 [亮, 暗]           ├─ 局部均值拉平（σ=10 mm，去偏置场）→ Otsu 两类 [WM, GM]
  └─ 24–35 µm 级 Frangi 暗管 → 自动血管分割      └─ 局部中值暗斑图（血管在 ex-vivo FLASH 里是暗点）
① 0.6 mm 全局搜索：masked NCC over 全部平移 × 8000 旋转 × {正常, 镜像}（FFT，整半球 150 s）
② 结构精配准：rigid → similarity → affine，0.6 → 0.3 → 0.15 mm（尺度先验），对 OCT 两种类极性各做一遍，取精配准 NCC 高者
③ 血管精配准：第三通道 = MRI 暗斑 vs OCT 自动血管密度，三通道 NCC（权重 1,1,2）仿射，尺度范围放宽
④ 评价（只在有标注时）：手工 MRI 血管 → 最近 OCT 血管距离（含随机平移对照）、OCT 类 vs 手工标签 Dice、扰动重启、深度轴尺度
```

- 两侧的结构表示都是**不需要训练**的两类强度划分；MRI 侧先用局部均值拉平消除偏置场（整半球上这一步把 WM/GM 划分相对手工标签的 Dice 从 0.66/0.62 提到 0.90/0.93）。作为消融保留了一个在 I46 手工标签上训练的 3D U-Net parser（`--features parser`）。
- 类极性（OCT 里亮的是 WM 还是 GM）是成像协议的属性，`--oct-wm-bright yes|no` 给定；`auto` 对两种极性各跑一遍 ①②，取精配准 NCC 高者。0.6 mm 的搜索分数分不开两种极性（皮层"GM 在外、WM 在内"的两类图换类之后在别处也能匹配得同样好），精配准 NCC 能分开（I46：0.757 vs 0.730）。
- 血管通道的依据：组织类几何对 block 深度轴的尺度几乎不敏感（§4），血管能看见这个尺度。MRI 侧用局部中值暗斑，OCT 侧用我们自己在 24–35 µm 级做的 Frangi 暗管分割（组织内 top-1%）池化成密度，两者做 NCC。不需要任何标注。
- 所有窗口/核都是物理尺寸（mm / µm），金字塔层级按各自体素大小取整，FFT 搜索网格按 block 大小补零，所以同一套参数用于 0.12–0.15 mm 的 MRI 与 12–35 µm 的 OCT。

## 2. 数据（DANDI:000026，Costantini 2023）

| subject | OCT block（µm，z,y,x） | MRI（EPIC 校正 FLASH，体素） | 手工标注 |
|---|---|---|---|
| I46 | 627×1271×1230 @ 12（sidecar；实际深度 ≈14） | 20°，0.15 mm，1280×1040×576 | 全半球 GM/WM/层、BA44/45、MRI 血管、OCT 血管分割 |
| I38 | 30×30×35 | 20°，0.12 mm，1600×1400×640 | 同上（含 OCT 血管分割） |
| I55 | 12×12×14 | 20°，0.15 mm | 全半球、BA、MRI 血管 |
| I56 | 30×30×35 | 20°，0.12 mm | 同上 |
| I62 | 30×30×35 | 20°，0.12 mm | BA、MRI 血管（无全半球标签） |
| I57 / I61 | 30×30×35 | 20°，0.12 mm | 全半球、BA（血管标注为空） |
| I48 | 30×30×35 | flip-2（sidecar 缺失），0.12 mm | 全半球、BA（血管标注为空） |
| I58 | 30×30×35 | 10° 与 30°（无 20°） | 全半球、BA、MRI 血管 |

下载清单 `data/manifest_multisub.txt`（81 个文件，60 GB）；拉取脚本 `data/restart_dl.sh`（aria2，预签名 URL 6 h 过期，`data/dl_watchdog.sh` 定时重启）。没有下载的：SPIM/LSFM（数 TB）、原始多回波 MRI、其他 flip angle、定量图。

## 3. 结果（表：`work/runs/summary_all/summary.md`；QC 图逐一目检）

DANDI 8 个已完成 subject（整半球、无先验、无训练；`work/runs/I*_otsu`）：**5 个目检正确**——I46（血管 265/0.28→167/0.47，own 123/0.56，对照 ~352；Dice 0.80/0.83；深度拉伸 1.22）、I55（Dice 0.92/0.93、血管 141/0.52、深度拉伸仅 1.10——它的 sidecar 深度间距是对的，反向验证 I46 元数据缺失）、I38/I56/I62（30–50 mm 大 slab，搜索分数 0.29–0.34 但重启 10–12/12 收敛，QC 逐脑回对应；低 Dice 是标签覆盖不全的评价假象）。**1 个存疑**（I48，搜索几乎无分差 0.257/0.256）、**3 个错误**（I57/I61：浴液很重的 0.12 mm MRI 上组织 mask 失败/不稳，block 被平坦区捕获；I58：仅有 10° 翻转角的 MRI，GM/WM 对比度几乎为零——与脑干 OCT 琼脂糖同类的"MRI 侧前景/对比度"问题，v1.1 一并修）。极性与手性逐 subject 不同（I46 GM 亮+镜像，I55 WM 亮+不镜像……），成功 case 全部自动判对。

Xiangrui 的 I58 脑干对（20 µm OCT NIfTI + 0.08 mm cropped MRI；`work/runs/xiangrui_I58bs_novasc`）：OCT 无法按强度区分组织和琼脂糖（mask 含包埋介质），overlap 门槛改为按两侧组织体积自适应后，全局搜索+仿射**无先验找到正确姿态**（三个切片轮廓/穹顶/独立小块都对应，重启 10/12，无镜像），残差数 mm、深端更大；血管通道在此数据无效（OCT "血管"被切片接缝伪影主导，NCC≈0，已对此对关闭）；跨模态强度 NCC 也无效（≈0.03）。v1.1：纹理边界的标本 mask + 抗伪影细通道。

## 4. 验证与发现（I46 上的深入验证，详见 `scripts/validation_i46/` 与上一版 README 的 §4）

1. 用手工标签+手工血管拟合的 oracle 仿射证明：结构阶段之后还差 ~1.4 mm（角点），主要是沿 OCT 深度轴 1.14–1.18 的拉伸；沿该轴 NCC 与标签 Dice 都是平的，组织类几何看不见它；血管通道把它找回来（到 oracle 角点 0.29 mm，血管指标到达标注上限 0.47–0.48）；对照（翻转/打乱血管密度）无增益；交叉验证的 oracle held-out 0.42。
2. 深度轴拉伸的来源：I46 的 OCT sidecar 写 12 µm 各向同性，而数据集里其他 subject 的 sidecar 是 30×30×35 或 12×12×14 µm——深度方向比面内大 1.167 倍，正是我们测到的 1.18。即 I46 的元数据少写了深度间距，血管阶段把它校正回来了。[待确认：其他 subject 用正确 sidecar 后深度尺度 ≈ 1.0]
3. 极性对称：两类 WM/GM 图换类之后在另一个位置可以匹配到同样高的搜索分数（0.741 vs 0.726），甚至对手工标签的 Dice 也高（0.86/0.77）——因为皮层是一张 GM 包 WM 的"片"，换类相当于找一个反过来的位置。所以极性要么当协议参数给定，要么靠精配准 NCC 决定；血管通道 NCC 是独立的检验（错位置上 ≈ 0）。
4. 非刚性与自蒸馏：在 I46 上都试过（自由形变 Dice +0.015、血管 +0.01；蒸馏 OCT parser 搜索更强但继承初始尺度）。收益小、复杂度高，v1 不含。

## 5. 复现（Xiangrui 对：`bash octreg/scripts/run_xiangrui.sh`）

```bash
# 服务器 ~/autodl-tmp/oct-mri-registration，conda activate octmri
bash octreg/scripts/run_subject.sh I38 flip-2      # prep + 整半球 otsu 运行 + parser 对照（~40 min）
python octreg/scripts/prep_subject.py --work work/X --mri X.nii.gz --oct X.ome.tiff [--labels-wholehemi ... --labels-ba ... --mri-vessels ... --oct-vesseg ...]
python octreg/scripts/register.py --work work/X --out work/runs/X_otsu [--oct-wm-bright no|yes|auto] [--crop-centre x,y,z --crop-half-mm 30]
python octreg/scripts/viz_result.py --work work/X --run work/runs/X_otsu     # 三张深度切片：OCT 血管(绿)/MRI 暗斑/手工血管(黄)
python octreg/scripts/summarize.py                                          # 汇总表 work/runs/summary_all/summary.md
```

Xiangrui 的数据（一个 OCT + 一个 cropped MRI）：`prep_subject.py --mri crop.nii.gz --oct oct.tif --oct-spacing-um z,y,x`，然后 `register.py`；MRI 若是 WM 亮就加 `--mri-wm-bright`；OCT 极性不确定就 `--oct-wm-bright auto`。输出 `T_oct2mri.npy/.lta`（OCT 世界 → MRI 世界）、`oct_in_mri_region.nii.gz`、`qc_*.png`、`result.json`。

## 6. 文件

- `octreg/common.py` 仿射/重采样/池化/OCT 归一化；`features.py` 两类划分（含拉平的低内存版）、Frangi；`search.py` FFT 全局搜索；`refine.py` 仿射精配准（masked NCC）；`vascular.py` 血管通道；`evaluate.py` 评价与 QC 图；`parser.py`/`synth.py`/`train_parser.py` 可选的 parser 对照；`nonrigid.py` 仅 I46 验证用。
- `scripts/`：`prep_subject.py`、`register.py`、`run_subject.sh`、`parse_mri.py`、`viz_result.py`、`summarize.py`；`scripts/validation_i46/` 是 I46 深入验证脚本（oracle、CV、尺度剖面、非刚性、蒸馏等）；`scripts/legacy/` 是上一版的 I46 专用脚本。
