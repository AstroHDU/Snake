# SRI

从Gaia源级观测计算恒星复合体的五项结构评分、综合SRI、分组和等级，并独立提供KPD结构分区及主体core_SRI。

## 快速运行

需要Python 3.10或更新版本。在本目录打开终端：

```bash
python -m pip install -r requirements.txt
python run_sri.py --input examples/raw_basic/Snake_Member_Catalogue.csv --output example
```

也可将一个CSV放入`input/`后运行`python run_sri.py`。在Spyder中将工作目录设为本目录，或在`run_sri.py`顶部将INPUT、OUTPUT、CONFIG设置为完整路径。相对路径以运行时工作目录为准；程序不依赖作者计算机的目录。

当前本地运行入口默认启用背景星，并以`id_part`限制支持范围；普通无背景数据请将PARENT和BRIDGE_SCOPE_COL都设为None。如需要将节点标签为空的源作为空间桥接候选：

```bash
python run_sri.py --input members.csv --parent true --output results
```

可选`--bridge-scope-col id_part`限制背景只支持相同scope。背景身份仅由节点标签为空决定。背景不参与实体成员数、速度中心、速度评分和Scen宽度。

## 输入

一个CSV，一行一颗源。标识列默认为`Snake`、`id_node`、`source_id`，可用`--group-col`、`--node-col`、`--source-col`修改。每组至少两个基础节点；每个节点需有足够、非退化的成员云。

| Gaia列 | 单位/要求 |
|---|---|
| ra, dec | ICRS度 |
| parallax | mas，必须正且有限；使用1000/parallax得到pc |
| pmra, pmdec | mas/yr；pmra包含cos(dec) |
| radial_velocity | km/s；缺失允许 |
| parallax_error, pmra_error, pmdec_error | 默认误差模式必须正且有效 |
| radial_velocity_error | RV实测值存在时必须正且有效，km/s |
| parallax_pmra_corr, parallax_pmdec_corr | 可省略；省略采用零相关约定；若提供必须每行有效且在[-1,1] |

源ID按文本读取。同一系统中不允许重复source_id。输入表已有的X/Y/Z、Vra/Vdec、U/V/W和评分不会被使用。所有工作坐标从上述原始观测重新计算。

配置默认`use_uncertainty=true`（运行入口`USE_UNCERTAINTY=None`表示遵循配置）：必要误差或实体协方差无效时，整个系统评分失败并记录原因，继续处理其他系统；不静默替换为零误差。没有RV与有RV但误差无效不同。`--no-use-uncertainty`显式运行名义模式，此时不使用协方差缓和，也不使用需要误差的RV合并否决，因此实体划分也可能改变；不是默认发布结果。

## 评分口径

- Smem：成员云局部联系；共享稳健散布归一化，邻接响应`1/sqrt(1+(d/spatial_mem_scale)^2)`，默认尺度为2。
- Scen：物理中心最长树边相对其余树边Q75参照；云宽度为同一Smem散布矩阵的方向投影平方根。两项共享散布估计器，职责和响应不同。
- Vmem：独立的二维成员云联系，采用共同切面；RV充分边使用模型透视输运，否则两端统一使用RV-free表示。
- Vcen、Cross：使用独立实体三维中心。实体Vra、Vdec、RV分别取自身有效观测的中位数，再在实体代表方向转成UVW。没有为每颗成员补造RV后计算三维中心。
- 中心误差保留`I^-1 + s_int^2/N_valid`。每个分量使用自己的有效源数量，通过方向旋转形成实体三维协方差。
- 两项cen在有效RV实体数M_RV>=3时启动；Cross在M_RV>=4时启动。Scen激活后使用全部空间有效实体，Vcen/Cross仅用三维有效实体。RV充分指至少3个有效RV源。
- Cross使用误差调整后残差模长的算术平均及当前方向误差缓和。高误差下的高相容分不是独立的高可靠性证明。

A组M_RV<3，B组M_RV=3，C组M_RV>=4。Gold：SRI>=0.7；Silver：0.5<=SRI<0.7；Bronze：SRI<0.5。等级为本评分约定，不是成员概率或显著性。本评分不依赖目录级null或qref。可选背景支持内部仍使用100次旋转对照，这与最终评分分位数定标不同。

通道内部采用几何平均`S=sqrt(Smem*Scen)`、`V=sqrt(Vmem*Vcen)`；未激活cen时使用对应mem。外层`R0=2*S*V/(S+V)`；Cross激活时`SRI=R0*Cross`，否则`SRI=R0`。

所有可调科学默认值集中在`config/defaults.json`。`CONFIG=None`会读取该文件，而非绕过它。主要参数：`outlier_prominence=3`、`h_cross_kms=5`、`spatial_mem_scale=2`、`affine_ridge=0.1`、等级线0.5/0.7。修改配置后重新启动Python进程；Spyder可重启内核。输入/输出、背景开关和误差运行开关仍在run_sri.py顶部；显式命令行参数优先。

完整公式与参数边界见[方法说明](docs/method.md)。

## KPD与主体

KPD独立于Full SRI，不修改Full分数，不自动删除原始成员：

1. 所有空间有效实体参与空间树；最长边与其余最大连接及云宽度比较，严格比值>3时切一条边并递归。
2. 空间组内，RV不足实体只通过统一RV-free二维云与组内其他实体比较。二维切分只允许拆出完全由RV不足实体组成的一侧，不改分已有三维实体之间的关系。
3. 各部分内，RV充分实体按三维速度MST最长/次长>3递归切分。
4. 二维并不异常的RV不足实体，按其RV-free云与三维各块合并成员云的距离归入最近块。归属用固定的RV充分成员云参照，不因先后分配而更新。不生成假的RV或三维速度。
5. 输出唯一最终分区。精确距离并列则归属未确定；不强行决定主体。小于三实体的子块不再作边比值判断。

只在KPD=1时计算core_SRI：先选实体数最多的最终块；同数则选成员星数最多的块，背景星不计数。两项都并列时不计算core；主体少于两个实体时不计算。主体从保留源重新计算全部分项、协方差、组别与等级，保持既定实体划分，不再次合并。排除部分的成员不转成背景。

RV-free二维补充仍可能受大天区投影影响，只是结构提议，不等价于实测三维异常确认。KPD=0表示未触发规则，不证明绝对无异常；不可评估用空值表示。

## 输出

- `SRI.csv`：Full五项、空间/速度综合分、SRI、ABC、等级、KPD及core摘要。
- `KPD.csv`：标记、唯一分区、各通道切分标记、每次边比及主体状态和core各项。
- `Core_SRI.csv`：仅成功计算的KPD主体，每行含core各项、总分、ABC及等级。
- `Derived_Entity_Kinematics.csv`：Full/core实体代表UVW、有效RV数及误差信息。
- `Derived_Node_Catalogue.csv`：从源表派生的节点数据，可用`--no-derived`关闭。
- `failures.csv`：失败系统与原因；严格误差模式下core计算失败也使该系统失败。
- `input_mapping.json`、`run_config.json`：输入映射、参数与本次运行状态。

跳过KPD和core可用`--no-kpd`。成功退出码0；有系统失败退出码2。不要把core分数替换Full分数而不说明样本改变。

## 验证

```bash
python -m pytest tests -q
```

`MANIFEST.sha256`记录发布文件校验值。示例仅演示输入格式与运行方式。内部研究记录和历史实验不包含在本目录；input中的真实数据仅供本地运行，公开分发前需确认数据授权。
所有结果表统一使用`system_id`保存输入的原始系统标识。`Snake`和`rng_id`仅是内部处理编号，不输出到结果表；原始列名与内部映射见`input_mapping.json`。
