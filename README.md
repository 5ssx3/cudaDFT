# cudaDFT：从 LiH 算例开始计算

cudaDFT 是面向孤立分子的 C++/CUDA 逆 Kohn–Sham 计算项目：从分子结构和目标 AO 密度矩阵出发，在均匀实空间网格上使用有限差分、Davidson 本征求解器和 L-BFGS 优化器反演局域有效势 `veff`。

本文面向已经在任意位置完成编译安装的用户，介绍如何准备输入、启动计算和检查结果。运行示例使用已加入 `PATH` 的 `cudaDFT` 命令，不依赖源码目录或特定机器的安装路径。若程序未加入 `PATH`，将命令中的 `cudaDFT` 替换为你安装的可执行文件的完整路径。

本文按当前源码说明输入和运行行为；除明确标为可选的字段外，输入项均需提供。参数示例用于说明格式，不能保证任意网格和体系都在给定步数内收敛。

## 1. 快速开始

### 1.1 确认程序与 GPU 环境

在准备运行计算的终端检查：

```bash
command -v cudaDFT
nvidia-smi
```

需要已安装的 cudaDFT、可用的 NVIDIA GPU 驱动，以及与编译版本匹配的 CUDA 运行库。程序和运行目录可以位于不同位置。集群用户应在已分配 GPU 资源的作业中执行后续计算命令，模块加载和作业提交方式按所在集群配置。

复用已有 LiH 输入进行反演时，无需重新运行 PySCF。只有准备新体系的分子和密度文件时，才需要使用生成这些输入的工具。

### 1.2 获取 LiH 输入

可从 [LiH 算例](test/LiH_160x160x180_FD8/target) 获取配套的 `molecule.json` 和 `density_total.npy`。将第 2 节的当前格式保存为 `input.yaml`，组成以下目录：

```text
LiH_160x160x180_FD8/
├── input.yaml
└── target/
    ├── molecule.json
    └── density_total.npy
```

| 文件 | 内容 |
| --- | --- |
| `input.yaml` | 输入路径、计算盒、网格、求解器、收敛条件和输出设置。 |
| `molecule.json` | PySCF 分子序列化文件，提供原子、核电荷、分子电荷、基组和 AO 顺序。 |
| `density_total.npy` | 与分子文件配套的实数、AO 表示、自旋求和总密度矩阵，形状为 `(nao, nao)`。 |

这三个文件足以准备一次计算，不需要复制算例中的旧 `results/` 和 `run.log`。仓库中的 LiH 运行记录和结果属于历史版本，其 `input.yaml` 包含旧字段；使用当前程序时请采用第 2 节的格式，不直接照搬历史配置或以旧日志判定新程序的收敛条件。

该 LiH 体系有 4 个电子。固定占据模式下，最低两个空间轨道各占据两个电子；额外求解的空态占据数为零。本文设置 `extra_states: 1`，共求解三个态。

### 1.3 建立独立运行目录

在任意工作目录下放好上述 `LiH_160x160x180_FD8/target/` 文件夹，然后执行：

```bash
mkdir -p LiH_run/target
cp LiH_160x160x180_FD8/target/molecule.json LiH_run/target/
cp LiH_160x160x180_FD8/target/density_total.npy LiH_run/target/
```

将第 2 节的 YAML 保存为 `LiH_run/input.yaml` 后即可运行。`LiH_run` 是本次新计算的目录名，可以自行更换。重复计算时建议使用新目录，保留已有结果。

### 1.4 启动计算与查看日志

仍在上述工作目录执行：

```bash
cudaDFT LiH_run/input.yaml > LiH_run/run.log 2>&1
echo "Exit status: $?"
```

程序在前台运行，结束后打印退出状态。`run.log` 同时保存标准输出和错误输出。在另一终端进入同一工作目录，可查看进度：

```bash
tail -f LiH_run/run.log
```

也可以先进入 `LiH_run/` 再执行 `cudaDFT input.yaml`；省略参数时，程序读取当前工作目录的 `input.yaml`。

### 1.5 确认完成与输出位置

完整成功运行应满足：退出状态为 `0`，日志包含最终 `IKS | CONVERGED` 汇总，`Density L2 / tol` 标为 `converged`，且没有后续输出错误。单次 `L-BFGS | CONVERGED` 只表示梯度达标，不代表完整反演成功。启用示例中的全部输出开关时，运行目录为：

```text
LiH_run/
├── input.yaml
├── target/
│   ├── molecule.json
│   └── density_total.npy
├── run.log
└── results/
    ├── LOCPOT
    ├── DENSITY
    ├── DENSITY_DIFFERENCE
    ├── EIGENVALUES
    ├── ORBITAL_0001
    ├── ORBITAL_0002
    └── ORBITAL_0003
```

若未收敛或程序报错，先查看 `run.log` 的终止原因。程序在反演返回后会尝试输出，即使反演失败也可能存在结果文件；文件存在不能证明收敛。输出开关和文件单位见第 3.8 节；更详细的输出格式见第 5 节。

### 1.6 输入路径规则

YAML 中的 `system.molecule_file`、`system.density_file` 和 `output.directory` 均相对于 **YAML 文件所在目录**解析，也可使用你自己的绝对路径。例如 `LiH_run/input.yaml` 中的 `target/molecule.json` 指向 `LiH_run/target/molecule.json`，`./results` 指向 `LiH_run/results/`，与程序安装位置无关。

修改已支持的输入参数无需重新编译。LiH 参考算例使用 `160 × 160 × 180` 个内部节点和差分半宽 `4`，具有约 461 万个网格点；计算前应确认 GPU 显存能够容纳求解器工作空间。

## 2. LiH 算例的完整输入参数

以下是与当前读取器匹配的完整 LiH 输入，使用 `160 × 160 × 180` 网格。它不是历史运行记录的原始配置，也不是已验证必然收敛的参数组合；实际计算需要检查网格、计算盒、正则强度和求解阈值的收敛性。

```yaml
system:
  molecule_file: target/molecule.json
  density_file: target/density_total.npy
  spin: unpolarized

domain:
  padding: [12.0, 12.0, 12.0]
  orbital_boundary: dirichlet_zero

mesh:
  type: uniform
  finite_half_width: 4
  rgrids: 160 160 180

inversion:
  initial_potential: zero
  regularization_eta: 1.0e-8
  relativ_rho_L2tol: 1.0e-5
  extra_states: 1

  eigensolver:
    method: davidson
    residual_tolerance: 1.0e-9
    max_subspace_dim: 32
    max_iterations: 10000
    nextra_retain: 3
    direction_tolerance: 1.0e-8

  occupations:
    method: fixed
    temperature: 0

  optimizer:
    method: lbfgs
    history: 10
    max_iterations: 1000
    max_linesearch_step: 50
    gradient_tolerance: 5.0e-6
    c1: 1.0e-4
    c2: 0.9

output:
  directory: ./results
  write_veff: true
  write_density: true
  write_density_difference: true
  write_eigenvalues: true
  write_wave: true
```

输入采用严格校验：只允许一个 YAML 文档，拒绝未知字段、重复字段、缺失必填项和非法类型。`relativ_rho_L2tol` 的拼写与程序一致，不能改为 `relative_rho_L2tol`。旧的 `stencil_half_width`、`hartree_density`、`inversion.objective` 和 `inversion.convergence` 已不被接受。

## 3. 参数说明

### 3.1 `system`：目标体系

| 参数 | 示例 | 含义与约束 |
| --- | --- | --- |
| `molecule_file` | `target/molecule.json` | 分子及基组文件路径，不能为空。 |
| `density_file` | `target/density_total.npy` | AO 总密度矩阵文件路径，不能为空。 |
| `spin` | `unpolarized` | 当前仅支持此值，对应 `spin=0`、正偶数电子的闭壳层体系；每个占据空间轨道容纳两个电子。 |

分子电荷、电子数和基组从 `molecule.json` 获取，无需在 YAML 中重复设置。电子数为核电荷之和减去分子电荷。

### 3.2 `domain`：计算盒与边界

| 参数 | 示例 | 含义与约束 |
| --- | --- | --- |
| `padding` | `[12.0, 12.0, 12.0]` | 在原子坐标包围盒的 x、y、z 各方向两侧分别增加的长度，单位 bohr；必须为三个有限、非负数，最终每个方向盒长必须大于零。 |
| `orbital_boundary` | `dirichlet_zero` | 当前仅支持此值，轨道在物理盒表面取零。 |

当前主流程直接优化完整有效势，未单独构造 Hartree 势；`hartree_boundary` 和 `multipole_order` 不是当前支持的输入字段。

### 3.3 `mesh`：均匀节点网格

| 参数 | 示例 | 含义与约束 |
| --- | --- | --- |
| `type` | `uniform` | 当前仅支持全域均匀网格。 |
| `rgrids` | `160 160 180` | x、y、z 三个方向的内部节点数，必须是一个以空格分隔三个正整数的标量，不能写成 YAML 列表 `[160, 160, 180]`。总点数不得超过求解器的 `int` 范围。 |
| `finite_half_width` | `4` | 有限差分半宽 `w`，正整数且不超过任一方向节点数。内部中心差分的标称阶数为 `2w`：`1` 对应二阶，`4` 对应八阶。实际精度需通过收敛测试确认。 |

`rgrids` 不包含盒表面的零边界节点，也不包含 halo。对任一方向，设物理盒长为 `L`、内部节点数为 `n`，则：

```text
h = L / (n + 1)
x[i] = lower + (i + 1) * h,  i = 0, ..., n - 1
dV = hx * hy * hz
```

因此，网格间距由原子坐标、`padding` 和 `rgrids` 共同决定。增大计算盒但保持节点数不变会使网格变粗。场数组按 `(z * ny + y) * nx + x` 排列，x 方向连续。

### 3.4 `inversion`：反演与初始条件

| 参数 | 示例 | 含义与实际行为 |
| --- | --- | --- |
| `initial_potential` | `zero` | 只接受 `zero`；当前初始有效势全为零。 |
| `regularization_eta` | `1.0e-8` | 有限、非负的势平滑正则系数，已加入目标函数和梯度；设为 `0` 关闭正则项。 |
| `relativ_rho_L2tol` | `1.0e-5` | 有限且大于零的相对密度 L2 误差阈值；当前在优化器返回梯度收敛后检查。 |
| `extra_states` | `1` | 额外求解的空态数，非负整数；实际求解态数为 `Ne/2 + extra_states`。 |
| `occupations.method` | `fixed` | 只接受 `fixed`；最低 `Ne/2` 个空间轨道各占据两个电子，不使用温度或展宽参数。 |
| `occupations.temperature` | `0` | 必填的有限、非负数；固定占据模式下不影响占据数。 |

### 3.5 `inversion.eigensolver`：Davidson 本征求解

| 参数 | 示例 | 含义与实际行为 |
| --- | --- | --- |
| `method` | `davidson` | 当前仅支持 Davidson。 |
| `residual_tolerance` | `1.0e-9` | 本征残差二范数 `‖Hψ - εψ‖₂` 的绝对阈值，有限且大于零。 |
| `max_subspace_dim` | `32` | 最大子空间维数，正整数且不超过总网格点数；还须满足下方约束。增大时显存需求增加。 |
| `max_iterations` | `10000` | 每次 Davidson 调用允许的最大迭代次数，须大于 `1`；不同于反演外层迭代上限。 |
| `nextra_retain` | `3` | 重启时在全部目标态之外额外保留的 Ritz 向量数，非负整数；不改变实际求解态数。 |
| `direction_tolerance` | `1.0e-8` | 正交化后接受新搜索方向的相对范数阈值，必须满足 `0 < value < 1`。 |

令 `Ngrid = nx * ny * nz`，当前输入校验与求解器初始化共同要求：

```text
nstates = Ne / 2 + extra_states
nstates + nextra_retain < max_subspace_dim <= Ngrid
```

实际态数需要读取分子电子数后才能确定。对于电子数较多的体系，需要相应增加 `max_subspace_dim`。单次 Davidson 达到迭代上限时，反演回调最多继续调用 10 次；仍未收敛则返回错误。

### 3.6 `inversion.optimizer`：L-BFGS 优化

| 参数 | 示例 | 含义与实际行为 |
| --- | --- | --- |
| `method` | `lbfgs` | 当前仅支持 L-BFGS。 |
| `history` | `10` | 保存的历史向量组数，正整数；增大时显存需求增加。 |
| `max_iterations` | `1000` | 反演外层调用优化器的最大次数，正整数；一次调用至多接受一步。 |
| `max_linesearch_step` | `50` | 每次调用的线搜索试探上限，正整数；实际传入优化器。 |
| `gradient_tolerance` | `5.0e-6` | 优化梯度二范数阈值，有限且非负；实际梯度还含有正则项。 |
| `c1` | `1.0e-4` | Wolfe 充分下降条件参数，实际传入优化器。 |
| `c2` | `0.9` | 强 Wolfe 曲率条件参数，实际传入优化器。 |

输入中的 Wolfe 参数仍必须满足 `0 < c1 < c2 < 1`。达到单次线搜索上限时，会保留已接受的状态并输出警告；外层可继续重试，该调用仍计入 `max_iterations`。

### 3.7 目标函数、正则项与收敛判据

目标密度 `rho_target` 由外部输入构建，在反演期间固定。当前回调优化的函数和离散梯度为：

```text
f(v) = -sum(occupation[s] * eigenvalue[s])
       + dV * sum(v * rho_target) + R(v)
R(v) = -eta * dV * v^T * L_reg * v
g = dV * (rho_target - rho_KS) - 2 * eta * dV * L_reg * v
```

`eta` 对应 `regularization_eta`。`L_reg` 使用有限差分二阶导数系数，只累加网格内有效邻居的势差，不对网格外势值补零。因此常数势偏移不改变正则项；比较不同势或绝对能级时，需要约定相同零点。动能算子的轨道零边界处理与正则算子分别实现。

日志中的 `f` 是反演目标函数值，不是分子总能量。相对密度误差定义为：

```text
density_relative_l2 = sqrt(sum((rho_KS - rho_target)^2) / sum(rho_target^2))
```

当前反演停止流程为：

1. 每次 L-BFGS 调用至多接受一步；接受步满足充分下降与强 Wolfe 曲率条件，梯度模不要求单调下降。
2. 只有返回梯度收敛状态时，才检查并打印 `relative density diff L2 / tol`。密度误差未达标时继续优化，不再设置额外的密度重试次数上限。
3. 梯度达标且密度误差小于 `relativ_rho_L2tol` 时，返回完整反演收敛。若外层次数耗尽仍未通过密度检查，返回 `-401`，最终汇总为 `IKS | FAILED`；尚未执行密度检查时，汇总显示 `n/a`。

正则梯度可能与密度梯度部分抵消，所以总梯度小不能代替实际密度误差。`gradient_tolerance` 过小也可能推迟密度检查；不能假定减小 `regularization_eta` 或增加迭代次数一定能达标。二者应结合实际密度误差与势的平滑程度做收敛测试。

梯度含有 `dV`。仅在 `eta=0` 时，`||g||₂ = dV * ||rho_KS-rho_target||₂`；调整网格后应重新评估梯度阈值。反演未通过密度检查时，继续迭代所用的内部标记可能覆盖最终汇总的梯度值；真实接受点的梯度以相应 L-BFGS 日志为准。

### 3.8 `output`：输出目录与开关

| 参数 | 示例 | 输出内容 |
| --- | --- | --- |
| `directory` | `./results` | 输出目录，相对于 YAML 所在目录解析；写结果时自动创建。 |
| `write_veff` | `true` | 写入 `LOCPOT`，保存最终反演有效势，单位 eV。 |
| `write_density` | `true` | 写入 `DENSITY`，保存重建的 KS 电子密度，单位 bohr⁻³。 |
| `write_density_difference` | `true` | 写入 `DENSITY_DIFFERENCE`，数值为 `rho_KS - rho_target`，单位 bohr⁻³。 |
| `write_wave` | `true` | 写入 `ORBITAL_0001` 等带符号实空间轨道，单位 bohr⁻³ᐟ²，积分范数为 1。 |
| `write_eigenvalues` | `true` | 可选，省略时为 `false`；写入 `EIGENVALUES`，三列依次为态编号、Hartree 单位的本征值、实际占据数。 |

开关使用 YAML 布尔值 `true` 或 `false`。除 `write_eigenvalues` 外，上表开关均为必填项。当前字段名为 `write_wave`，不接受 `write_orbitals`、`write_vxc`、`write_checkpoint` 或 `write_resolved_input`。

反演返回后会尝试输出，再根据反演状态返回进程退出码；未通过密度判据时也可能写出文件。程序被中途终止时，不能依赖该输出流程保存当前状态。当前没有检查点或从保存势恢复计算的输入接口，只支持零势初猜。输出相同文件名会覆盖已有内容；关闭某个开关不会删除该目录中已有的对应文件。

## 4. 分子与目标密度文件

### 4.1 `molecule.json`

使用 PySCF 分子对象的 `mol.dumps()` 序列化结果，保留 `_atm`、`_bas`、`_env` 等实际数组。当前支持：

- 全电子、点核模型，不支持 ECP 或赝势。
- `spin=0`、正偶数电子的闭壳层体系。
- `cart=false` 的实球谐 s/p/d/f 高斯基组，角动量 `l <= 3`。

读取器使用 PySCF 内部数组中的 bohr 坐标，不会依据原始 `unit` 字段再次转换；不要手动改写内部坐标或 AO 顺序。

### 4.2 `density_total.npy`

密度矩阵必须满足以下格式：

| 项目 | 要求 |
| --- | --- |
| NPY 版本 | `1.0`。 |
| 数据类型 | 小端实数 float64，即 `<f8`。 |
| 存储顺序 | C 连续顺序，`fortran_order=false`。 |
| 形状 | `(nao, nao)`，与分子文件中的 AO 数目和顺序一致。 |
| 数值 | 有限实数，矩阵须对称；对称性容差为 `1e-12 * max(1, max(abs(P)))`。 |
| 物理含义 | 自旋求和的 AO 一阶密度矩阵，不能直接提供 MO 表示矩阵或网格密度数组。 |

若已有同次计算的 PySCF 分子对象 `mol` 和实数 AO 总密度矩阵 `dm_ao`，可按以下方式导出。对于输出 MO 密度矩阵的方法，需要先转换到 AO 表示。

```python
from pathlib import Path
import numpy as np

target = Path("target")
target.mkdir(exist_ok=True)
(target / "molecule.json").write_text(mol.dumps(), encoding="utf-8")
dm = np.ascontiguousarray(dm_ao, dtype="<f8")
with (target / "density_total.npy").open("wb") as stream:
    np.lib.format.write_array(stream, dm, version=(1, 0), allow_pickle=False)

# AO 基组一般不正交，应使用 Tr(P S) 检查源矩阵电子数。
overlap = mol.intor_symmetric("int1e_ovlp")
print("electrons:", np.einsum("ij,ji->", dm, overlap))
```

程序先在网格上计算 `rho(r) = phi(r)^T P phi(r)`，不额外乘自旋因子；随后将目标网格密度按以下比例缩放，使离散积分等于分子电子数：

```text
rho_target = rho * Ne / (dV * sum(rho))
```

该步骤不修改源 NPY 文件。它也无法替代对源矩阵电子数、AO 顺序以及计算盒和网格精度的检查。

## 5. 输出格式与结果解读

`LOCPOT`、`DENSITY`、`DENSITY_DIFFERENCE` 和 `ORBITAL_####` 使用 VASP LOCPOT 的文本布局，场值按 x 方向最快的顺序排列，保留实际内部节点样本。文件头的晶格和原子坐标单位为 Å；场值单位见输出参数表。

为保存内部节点而不补边界或插值，输出文件原点取 `lower + gap`，文件晶胞各方向长度取 `rgrids * gap`，原子坐标相对于该原点平移；标题记录原点和原始物理盒边界。因此，输出文件晶胞长度与求解时的完整物理盒长不同。

密度文件保存直接的 bohr⁻³ 数值，不按 VASP CHGCAR 的体积缩放约定写入。`LOCPOT` 保存优化变量 `veff` 的原始零点，仅将 Hartree 转换为 eV，没有减费米能，也没有单独分离交换关联势。

`write_wave: true` 时，输出全部 `Ne/2 + extra_states` 个带符号轨道，而非仅占据态；密度只使用固定占据的最低 `Ne/2` 个态。`EIGENVALUES` 的表头为 `# state eigenvalue_Ha occupation`。至少求解一个空态后才能从输出读取最低空态与最高占据态之差；势零点和空态的盒尺寸依赖仍需注意。

运行日志包含每次外层迭代、Davidson/L-BFGS 状态和最终 IKS 汇总。`Density L2 / tol` 给出已检查的密度误差、阈值和状态，未检查时为 `n/a`。`Inversion time` 覆盖初始目标函数评价及外层迭代，不包含输入读取、网格/CUDA 初始化和结果输出时间。

## 6. 从 LiH 扩展到自己的体系

1. 为新体系建立独立运行目录，复制 LiH 的 `input.yaml`，将 `target/` 中的分子和密度矩阵替换为新体系的配套文件；确认 `Tr(P S)` 与分子电子数一致。新体系也须满足第 4 节的输入支持范围。
2. 根据分子大小调整 `padding` 和 `rgrids`。可用较小网格完成输入、求解与输出流程检查，再分别增加计算盒和节点数检查收敛；固定盒长时增加 `rgrids` 才会直接细化网格。
3. 按实际占据态数设置 `max_subspace_dim`，为重启保留态和新搜索方向留出空间；同时关注显存占用。
4. 将 Davidson 残差阈值收紧到足以稳定外层优化的程度，再结合密度误差调整 `gradient_tolerance`；更小的阈值通常需要更多迭代。
5. 区分本征求解迭代上限、反演外层上限和单次线搜索上限，并检查日志中的警告和最终终止原因。

本文按当前程序行为说明参数。使用其他版本的可执行文件时，应同时核对该版本的输入说明，尤其是参数是否已接入求解流程。

## 7. 首次运行的常见问题

| 现象 | 检查方向 |
| --- | --- |
| 找不到 `cudaDFT` 命令 | 将自己的安装目录加入 `PATH`，或使用已安装可执行文件的完整路径。 |
| 提示找不到共享库 | 检查所用安装的 CUDA 及其他运行库是否已加载，库路径是否与编译环境匹配。 |
| CUDA 初始化失败或没有可用 GPU | 确认正在获准使用 GPU 的环境中运行，检查驱动、设备可见性及程序编译的 GPU 架构。 |
| 无法打开分子或密度文件 | 核对文件是否齐全，以及路径是否相对于 `input.yaml` 所在目录填写。 |
| `required parameter is missing` 或 `unknown parameter` | 对照第 2 节的完整 YAML 和安装版本支持的字段，检查缩进、拼写和必填项。 |
| 没有预期结果文件 | 查看日志终止原因、程序退出状态、输出开关及 YAML 指定的输出目录。 |
