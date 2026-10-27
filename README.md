# cudaDFT：从 LiH 算例开始计算

cudaDFT 是面向孤立分子的 C++/CUDA 逆 Kohn–Sham 计算项目：从分子结构和目标 AO 密度矩阵出发，在均匀实空间网格上使用有限差分、Davidson 本征求解器和 L-BFGS 优化器反演局域有效势 `veff`。

本文面向已经在任意位置完成编译安装的用户，介绍如何准备输入、启动计算和检查结果。运行示例使用已加入 `PATH` 的 `cudaDFT` 命令，不依赖源码目录或特定机器的安装路径。若程序未加入 `PATH`，将命令中的 `cudaDFT` 替换为你安装的可执行文件的完整路径。

参数示例采用 LiH 算例的输入设置；除明确标为可选的字段外，输入项均需提供。部分字段目前仅被读取和校验，尚未接入数值流程，具体见参数表。

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

可从 [LiH 算例](https://github.com/5ssx3/cudaDFT/tree/master/test/LiH_160x160x180_FD8) 获取以下三个文件，并保留它们的相对目录结构：

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

这三个文件足以准备一次计算，不需要复制算例中的旧 `results/` 和 `run.log`。该 LiH 体系有 4 个电子，当前使用两个空间轨道、每个轨道占据两个电子。

### 1.3 建立独立运行目录

在任意工作目录下放好上述 `LiH_160x160x180_FD8/` 文件夹，然后执行：

```bash
mkdir -p LiH_run/target
cp LiH_160x160x180_FD8/input.yaml LiH_run/input.yaml
cp LiH_160x160x180_FD8/target/molecule.json LiH_run/target/
cp LiH_160x160x180_FD8/target/density_total.npy LiH_run/target/
```

`LiH_run` 是本次新计算的目录名，可以自行更换。重复计算时建议使用新目录，保留已有结果。

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

完整成功运行应满足：退出状态为 `0`，日志包含最终 `IKS | CONVERGED` 汇总，且没有后续的最终求解或输出错误。启用示例中的全部输出开关时，运行目录为：

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
    └── ORBITAL_0002
```

若未收敛或程序报错，先查看 `run.log` 的终止原因。输出开关和文件单位见第 3.8 节；更详细的输出格式见第 5 节。

### 1.6 输入路径规则

YAML 中的 `system.molecule_file`、`system.density_file` 和 `output.directory` 均相对于 **YAML 文件所在目录**解析，也可使用你自己的绝对路径。例如 `LiH_run/input.yaml` 中的 `target/molecule.json` 指向 `LiH_run/target/molecule.json`，`./results` 指向 `LiH_run/results/`，与程序安装位置无关。

修改已支持的输入参数无需重新编译。LiH 参考算例使用 `160 × 160 × 180` 个内部节点和差分半宽 `4`，具有约 461 万个网格点；计算前应确认 GPU 显存能够容纳求解器工作空间。

## 2. LiH 算例的完整输入参数

以下内容对应 LiH 算例的 `input.yaml`。首次复现可先保留这些设置；用于其他体系时，需要检查网格、计算盒和求解阈值的收敛性。

```yaml
system:
  molecule_file: target/molecule.json
  density_file: target/density_total.npy
  spin: unpolarized

domain:
  padding: [12.0, 12.0, 12.0]
  orbital_boundary: dirichlet_zero
  # hartree_boundary: multipole
  # multipole_order: 4

mesh:
  type: uniform
  stencil_half_width: 4
  rgrids: 160 160 180

inversion:
  hartree_density: target
  initial_potential: zero

  eigensolver:
    method: davidson
    residual_tolerance: 1.0e-9
    extra_states: 4
    max_subspace_dim: 32
    max_iterations: 10000
    nextra_retain: 3
    direction_tolerance: 1.0e-8

  occupations:
    method: fixed

  optimizer:
    method: lbfgs
    history: 10
    max_iterations: 1000
    max_linesearch_step: 50
    c1: 1.0e-4
    c2: 0.9

  objective:
    density_metric: relative_l2
    regularization:
      type: gradient_l2
      strength: 1.0e-6

  convergence:
    density_relative_l2: 1.0e-5
    gradient_tolerance: 1.0e-6

output:
  directory: ./results
  write_veff: true
  write_density: true
  write_density_difference: true
  write_eigenvalues: true
  write_wave: true
```

输入采用严格校验：只允许一个 YAML 文档，拒绝未知字段、重复字段、缺失必填项和非法类型。不能随意添加泛函名称、温度、检查点或其他未支持的选项。

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
| `hartree_boundary` | `multipole` | 可选，只接受 `multipole`；当前仅读取和校验，未接入主计算流程。 |
| `multipole_order` | `4` | 可选，非负整数；须与 `hartree_boundary` 一起提供，当前未接入主计算流程。 |

注释掉的字段不会生效；不提供 Hartree 边界字段时，不会自动启用多极边界。当前主流程直接优化完整有效势，未单独构造由这些参数控制的 Hartree 势。

### 3.3 `mesh`：均匀节点网格

| 参数 | 示例 | 含义与约束 |
| --- | --- | --- |
| `type` | `uniform` | 当前仅支持全域均匀网格。 |
| `rgrids` | `160 160 180` | x、y、z 三个方向的内部节点数，必须是一个以空格分隔三个正整数的标量，不能写成 YAML 列表 `[160, 160, 180]`。总点数不得超过求解器的 `int` 范围。 |
| `stencil_half_width` | `4` | 有限差分半宽 `w`，正整数且不超过任一方向节点数。内部中心差分的标称阶数为 `2w`：`1` 对应二阶，`4` 对应八阶。实际精度需通过收敛测试确认。 |

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
| `hartree_density` | `target` | 只接受 `target`；当前仅读取和校验，没有单独的 Hartree 更新步骤。 |
| `initial_potential` | `zero` | 只接受 `zero`；当前初始有效势全为零。 |
| `occupations.method` | `fixed` | 只接受 `fixed`；最低 `Ne/2` 个空间轨道各占据两个电子，不使用温度或展宽参数。 |

### 3.5 `inversion.eigensolver`：Davidson 本征求解

| 参数 | 示例 | 含义与实际行为 |
| --- | --- | --- |
| `method` | `davidson` | 当前仅支持 Davidson。 |
| `residual_tolerance` | `1.0e-9` | 本征残差二范数 `‖Hψ - εψ‖₂` 的绝对阈值，有限且大于零。 |
| `extra_states` | `4` | 非负整数；当前只参与输入校验，尚未用于增加求解态数。实际只求解 `Ne/2` 个占据态。 |
| `max_subspace_dim` | `32` | 最大子空间维数，正整数且不超过总网格点数；还须满足下方约束。增大时显存需求增加。 |
| `max_iterations` | `10000` | 每次 Davidson 调用允许的最大迭代次数，正整数；不同于反演外层迭代上限。 |
| `nextra_retain` | `3` | 重启时在占据态之外额外保留的 Ritz 向量数，非负整数；不改变实际求解态数。 |
| `direction_tolerance` | `1.0e-8` | 正交化后接受新搜索方向的相对范数阈值，必须满足 `0 < value < 1`。 |

令 `Ngrid = nx * ny * nz`，当前输入校验与求解器初始化共同要求：

```text
extra_states + nextra_retain + 1 < max_subspace_dim <= Ngrid
Ne / 2 + nextra_retain < max_subspace_dim
```

第二条需要读取分子电子数后才能检查。对于电子数较多的体系，需要相应增加 `max_subspace_dim`。

### 3.6 `inversion.optimizer`：L-BFGS 优化

| 参数 | 示例 | 含义与实际行为 |
| --- | --- | --- |
| `method` | `lbfgs` | 当前仅支持 L-BFGS。 |
| `history` | `10` | 保存的历史向量组数，正整数；增大时显存需求增加。 |
| `max_iterations` | `1000` | 反演外层调用优化器的最大次数，正整数；一次调用至多接受一步。 |
| `max_linesearch_step` | `50` | 正整数；当前 YAML 值仅被读取和校验，优化器初始化仍固定使用 `50`。 |
| `c1` | `1.0e-4` | Wolfe 充分下降条件参数；当前 YAML 值未传入优化器，实际使用内部初值 `1.0e-4`。 |
| `c2` | `0.9` | Wolfe 曲率条件参数；当前 YAML 值未传入优化器，实际使用内部初值 `0.9`。 |

输入中的 Wolfe 参数仍必须满足 `0 < c1 < c2 < 1`。达到单次线搜索上限时，会保留已接受的状态并输出警告；外层可继续重试，该调用仍计入 `max_iterations`。

### 3.7 `inversion.objective` 与 `inversion.convergence`

| 参数路径（相对于 `inversion`） | 示例 | 含义与实际行为 |
| --- | --- | --- |
| `objective.density_metric` | `relative_l2` | 只接受此值；当前仅读取和校验，优化回调没有将密度相对 L2 误差作为目标函数。 |
| `objective.regularization.type` | `gradient_l2` | 只接受此值；当前正则化项尚未加入目标函数或梯度。 |
| `objective.regularization.strength` | `1.0e-6` | 有限、非负数；当前不影响计算。 |
| `convergence.density_relative_l2` | `1.0e-5` | 有限且大于零；当前尚未用于停止判据。 |
| `convergence.gradient_tolerance` | `1.0e-6` | 优化梯度二范数阈值，有限且非负；**这是当前实际使用的反演收敛阈值**。 |

当前回调优化的函数和离散梯度为：

```text
f(v) = -2 * sum(occupied eigenvalues) + dV * sum(v * rho_target)
g[p] = (rho_target[p] - rho_KS[p]) * dV
```

日志中的 `f` 是反演目标函数值，不是分子总能量。`||g||₂` 达到阈值不代表程序已经独立检查了 YAML 中的密度相对误差阈值；需要时应另行计算：

```text
density_relative_l2 = sqrt(sum((rho_KS - rho_target)^2) / sum(rho_target^2))
```

梯度含有 `dV`，调整网格后应重新评估阈值与密度误差的对应关系。

### 3.8 `output`：输出目录与开关

| 参数 | 示例 | 输出内容 |
| --- | --- | --- |
| `directory` | `./results` | 输出目录，相对于 YAML 所在目录解析；写结果时自动创建。 |
| `write_veff` | `true` | 写入 `LOCPOT`，保存最终反演有效势，单位 eV。 |
| `write_density` | `true` | 写入 `DENSITY`，保存重建的 KS 电子密度，单位 bohr⁻³。 |
| `write_density_difference` | `true` | 写入 `DENSITY_DIFFERENCE`，数值为 `rho_KS - rho_target`，单位 bohr⁻³。 |
| `write_wave` | `true` | 写入 `ORBITAL_0001` 等带符号实空间轨道，单位 bohr⁻³ᐟ²，积分范数为 1。 |
| `write_eigenvalues` | `true` | 可选，省略时为 `false`；写入 `EIGENVALUES`，列为从 1 开始的态编号及 Hartree 单位的本征值。 |

开关使用 YAML 布尔值 `true` 或 `false`。除 `write_eigenvalues` 外，上表开关均为必填项。当前字段名为 `write_wave`，不接受 `write_orbitals`、`write_vxc`、`write_checkpoint` 或 `write_resolved_input`。

只有反演达到梯度收敛条件后，程序才执行最终结果输出；达到外层上限或发生计算错误时返回失败。输出相同文件名会覆盖已有内容；关闭某个开关不会删除该目录中已有的对应文件。

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

密度文件保存直接的 bohr⁻³ 数值，不按 VASP CHGCAR 的体积缩放约定写入。`LOCPOT` 保存优化变量 `veff` 的原始零点，仅将 Hartree 转换为 eV，没有减费米能，也没有单独分离交换关联势。当前只求解占据态，输出能级不能据此确定 HOMO–LUMO 能隙。

运行日志包含每次外层迭代、Davidson/L-BFGS 状态和最终 IKS 汇总。`Inversion time` 覆盖初始目标函数评价及外层迭代，不包含输入读取、网格/CUDA 初始化和最终重算、输出时间。

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
