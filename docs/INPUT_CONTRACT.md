# cudaDFT 逆 KS 输入参数与输入文件约定

更新日期：2026-10-24。本文汇总用户在输入设计会话中最后确认的内容，用于向以下两个已有会话交接设计上下文。最新输入采用 PySCF 的 `molecule.json` 与 AO 总密度矩阵 `density_total.npy`，网格为不按 block 分块的全域均匀网格；差分参数输入并保存半宽，C++ 成员命名为 `finite_difference_halfwidth`：

- [自适应网格有限差分](codex://threads/01a14381-907a-7312-9d16-890ac860dd53?hostId=remote-ssh-codex-managed%3A6f821bbf-d61d-4adf-8fcd-a3f9975514f1)
- [smpi.cpp的CPU 与 GPU 路径](codex://threads/01a1439f-aa70-7c62-900c-edaba0cb5a44?hostId=remote-ssh-codex-managed%3A6f821bbf-d61d-4adf-8fcd-a3f9975514f1)

共享文件路径：`/home/pengchengjie/Softwares/cudaDFT/docs/INPUT_CONTRACT.md`。

本文是输入设计交接文档。当前已实现 YAML 参数读取、所提供格式的 PySCF JSON/NPY 读取、盒内节点网格和球谐 s/p/d/f AO 密度评价；完整反演驱动尚未实现。参数数值是当前配置示例，不是经过精度收敛验证的生产参数。下面区分已确认约定与仍需落实的文件编码细节，不得将后者擅自当作用户批准的新输入参数。

求解器参数按 `src/Davidson.cpp`、`src/Davidson.hpp`、`src/LBFGS.cpp` 和 `src/LBFGS.hpp` 的实际接口设置。当前选择 Davidson 和 L-BFGS，删除旧切比雪夫谱界、滤波阶数及回溯线搜索假设。参数到 C++ 成员的映射见第 3.4.1、3.4.2 节；读取器与求解器的外层反演连接仍未实现。

## 1. 已确认的计算范围

- C++ 项目，面向孤立分子的逆 Kohn–Sham 问题，无 k 点。
- 反演变量是局域交换关联势 `vxc`；不输入交换关联泛函名称或 XC 参数。
- 计算参数和内部物理量固定使用原子单位制：长度 bohr、能量 Hartree、密度 bohr⁻³。PySCF 序列化 JSON 的单位处理见第 4 节，不能按其原始输入单位重复转换内部坐标。
- 使用全域均匀网格，物理场存储在盒内节点。`rgrids` 不包含盒表面的物理零边界和域外 halo；间距为 `L/(n+1)`，首点为下边界加一个间距，与现有 `read_odd_axis()` 对齐。不设置 block 尺寸，也没有粗细层级或原子半径细化。
- 一次反演过程中网格保持固定。每个内部网格点只参与一次未知量、密度积分和轨道内积计数，边界零点和 halo 不进入物理数组。
- 差分参数采用 `mesh.stencil_half_width`，输入正整数半宽 w，读入 `InputParameters::Mesh::finite_difference_halfwidth` 并原样保存在网格结构体中。精度阶数 2w 仅按需推导，不保存 `finite_difference_order`；不保留直接输入阶数、`discretization` 层或 `kinetic_operator` 字段。算子仍需满足厄米性、边界条件与实际收敛精度要求。
- 势的常数自由度不在计算内部通过平移处理，不设置 `gauge`。最终有效势输出使用 `LBFGSState.d_x` 原值，仅转换为 eV，不减费米能。
- PySCF 的 `molecule.json` 提供结构和基组，`density_total.npy` 提供已转换到 AO 表示的自旋求和一阶密度矩阵。程序在实际均匀网格采样点上评价目标电子密度，不需要重新运行 PySCF SCF/FCI，也不需要再提供 MO 系数或对矩阵乘 2。
- 当前配置采用固定占据 `occupations.method: fixed`，不设置 `kbt`。

## 2. 三个输入文件的职责

| 文件 | 唯一职责 |
|---|---|
| `input.yaml` | 系统文件路径、计算盒、网格、反演算法和输出选项 |
| `molecule.json` | PySCF 序列化分子信息，包含结构、实际基组、AO 顺序、电荷和自旋等 |
| `density_total.npy` | 自旋求和、AO 表示的一阶密度矩阵 |

`density_total.npy` 是矩阵，不是网格上的预采样密度。必须与同次计算导出的 `molecule.json` 配套，以匹配 AO 定义和顺序。无需另行输入 XYZ 或转换成自定义 HDF5 格式。

避免在多个输入文件重复手工维护结构、基组或网格信息。均匀网格由 JSON 中的结构及 YAML 中的计算盒、网格设置构建，不写入密度矩阵文件。当前工作文件已使用 `rgrids` 指定网格尺寸，本次原样保留；使用已有 PySCF JSON 时保留其原始序列化内容，不要求删除其中的内部重建信息。

示例运行目录为：

```text
calculation/
├── input.yaml
└── target/
    ├── molecule.json
    └── density_total.npy
```

`target/` 是相对于运行用 `input.yaml` 的示例路径，不是必须固定的目录名。当前 `/home/pengchengjie/Softwares/cudaDFT/src/test/LiH_FCI_aug-cc-pVTZ` 仅用于展示和核验文件格式，不应硬编码成实际运行路径；本次配置更新不复制或移动这些源文件。

### 2.1 编译与运行时读取

`input.yaml` 是可执行程序在运行时读取的外部配置，不参与 C++ 编译。修改已支持的参数或输入文件无需重新编译；修改源码或加入尚未实现的算法才需要重新编译。

建议启动接口为：

```bash
./cudaDFT /path/to/input.yaml
```

这是建议的接口形式，不表示当前已经存在该可执行程序或命令行解析器。程序先解析并校验 YAML，再读取其中指定的结构与密度源文件，构建网格、生成目标密度并执行反演。相对输入路径及 `output.directory` 统一相对于 `input.yaml` 所在目录解释，避免启动目录影响文件定位。

## 3. 当前确认的 input.yaml

以下内容对应 `/home/pengchengjie/Softwares/cudaDFT/src/input.yaml`。保持五个顶层分组，不自行补回早期讨论中删除的字段。

```yaml
# 1. 系统体系
system:
  molecule_file: target/molecule.json
  density_file: target/density_total.npy
  spin: unpolarized

# 2. 计算盒设置
domain:
  padding: [12.0, 12.0, 12.0]
  orbital_boundary: dirichlet_zero
  # hartree_boundary: multipole
  # multipole_order: 4

# 3. 网格设置
mesh:
  type: uniform
  stencil_half_width: 1
  rgrids: 80 80 80

# 4. 反解设置
inversion:
  hartree_density: target
  initial_potential: zero

  eigensolver:
    method: davidson
    residual_tolerance: 1.0e-9
    extra_states: 4
    max_subspace_dim: 32
    max_iterations: 100
    nextra_retain: 3
    direction_tolerance: 1.0e-8

  occupations:
    method: fixed

  optimizer:
    method: lbfgs
    history: 10
    max_iterations: 100
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
    gradient_tolerance: 1.0e-7

# 5. 输出设置
output:
  directory: ./results
  write_veff: true
  write_density: true
  write_density_difference: true
  write_wave: true
```

最新版本不设置单独的结构文件字段、block 尺寸、粗细网格参数、`task`、`schema_version`、`charge`、`variable`、`runtime`、`gauge`、`electron_number_tolerance`、对齐开关或 `mesh.update`。当前固定占据也不设置 `kbt`。讨论中出现过的旧字段不能擅自恢复到当前输入规范。

### 3.1 系统体系

| 字段 | 解释 |
|---|---|
| `molecule_file` | PySCF 分子 JSON 路径，提供结构和基组，内容见第 4 节 |
| `density_file` | AO 总密度矩阵 NPY 路径，内容见第 5 节 |
| `spin` | `unpolarized`：自旋非极化，使用自旋求和的电子数密度 |

当前工作文件未设置 `electron_model`，不在本次求解器参数更新中补回。示例 JSON 来源仍是全电子计算，物理模型的识别和核奇点离散处理需要由读取与势构建模块落实。

建议文件相对路径统一相对于 `input.yaml` 所在目录解析，而非依赖启动时的工作目录。

### 3.2 计算盒

| 字段 | 解释 |
|---|---|
| `padding` | 原子坐标包围盒在 x、y、z 各方向每一侧增加的宽度 |
| `orbital_boundary` | `dirichlet_zero`：盒表面轨道为零，需检查盒尺寸收敛 |

工作文件中的 `hartree_boundary`、`multipole_order` 已被注释，保留注释；读取器对未提供的 Hartree 边界保留空值，不将注释中的多极设置当作生效值。

轨道边界与 Hartree 边界不同。电子 Hartree 势满足 ∇²vH = −4πn，远场具有 Ne/r 尾部。

当前工作文件使用 `rgrids`，没有独立的 `spacing` 参数。每个方向先取原子坐标的最小/最大值，在两侧分别加 `padding` 得到盒边界；盒长必须为正。随后取 `h = (upper-lower)/(n+1)`，内部坐标为 `lower+(i+1)*h`，其中 `i=0,...,n-1`。差分半宽不改变物理盒、内部点数或间距，不引入 block 对齐。

日志及派生配置应记录实际盒原点、盒长、三个方向的有效单元数和间距。边界或并行运算所需的 ghost/halo 不属于物理域，不参与电子数积分计数。

### 3.3 全域均匀网格

| 字段 | 解释 |
|---|---|
| `type` | `uniform`：全域统一间距的均匀网格 |
| `stencil_half_width` | 中心差分模板沿每个坐标方向的单侧半宽 w，读入 C++ 成员 `finite_difference_halfwidth`，不乘 2 |
| `rgrids` | 当前工作文件为 `80 80 80`，在 YAML 中是一个字符串；读取器需解析为 nx、ny、nz 三个正整数，传给 Davidson 初始化接口 |

只保存半宽；如需展示精度阶数或模板跨度，可临时推导：

```cpp
// 参数读取器已将 YAML 的 stencil_half_width 存为原始半宽。
const int fd_half_width = parameters.mesh.finite_difference_halfwidth;
const int fd_order = 2 * fd_half_width;
const int stencil_points = 2 * fd_half_width + 1;
```

| 输入半宽 w | 内部中心差分阶数 2w | 一维模板跨度 2w+1 |
|---:|---:|---:|
| 1 | 2 | 3 |
| 2 | 4 | 5 |
| 3 | 6 | 7 |

这些阶数关系适用于对应精度的标准中心差分系数，物理边界处理仍需单独验证。向现有 `creat_finitePlan()` 传入原始半宽 w 和网格的三个间距，不将 2w 作为半宽传入。YAML 解析已在 `read_input.cpp` 实现。

本次不改变 `rgrids` 的现有写法和数值，也不将其改回间距输入。均匀网格每个有效单元的体积相同，为 dV = gap.x × gap.y × gap.z；三个方向间距相同时才写成 h³。所有密度积分和物理内积仍需包含体积因子。网格总层数、细化比、分级半径、相邻层平衡和块大小均不输入；实际全电子计算仍需检查空间收敛。

### 3.4 反解设置

| 字段 | 解释 |
|---|---|
| `hartree_density: target` | 从目标密度计算 Hartree 势并在反演期间固定 |
| `initial_potential: zero` | 初始 vxc 为零，不是总有效势为零 |
| `eigensolver.method` | `davidson`，调用现有 `DavidsonSolver`，不使用谱界或滤波阶数 |
| `eigensolver.residual_tolerance` | 映射到 `DavidsonPlan.residual_tol`，检查欧氏归一化 Ritz 向量的欧氏残差二范数 |
| `eigensolver.extra_states` | 外层在占据态数基础上增加的目标空态数，用来推导传给初始化函数的 `num_states` |
| `eigensolver.max_subspace_dim` | Davidson 搜索子空间的最大列数 |
| `eigensolver.max_iterations` | 每次调用 `DavidsonSolver` 的内部迭代上限 |
| `eigensolver.nextra_retain` | 子空间满后重启时，在所有目标态之外额外保留的 Ritz 向量数 |
| `eigensolver.direction_tolerance` | 映射到 `DavidsonPlan.tNorm_tol`，筛选投影后保留下来的修正方向，默认 1e-8 |
| `occupations.method: fixed` | 固定轨道占据数；当前闭壳层 LiH 使用两条双占据空间轨道，其余态占据为零 |
| `optimizer.method` | 势优化采用 L-BFGS，需要响应或伴随计算给出正确梯度 |
| `optimizer.history` | L-BFGS 保存的历史向量对数量 |
| `optimizer.max_iterations` | 外层最多调用 `LBFGSOptimizer` 的次数，包含未接受新步的调用；不是库内成员，也不是回调次数 |
| `optimizer.max_linesearch_step` | 每次 L-BFGS 调用允许的线搜索试探次数，现有初始化默认 50 |
| `optimizer.c1` | 强 Wolfe 线搜索中的 Armijo 充分下降常数，结构体默认 1e-4 |
| `optimizer.c2` | 强 Wolfe 线搜索中的曲率条件常数，结构体默认 0.9 |
| `objective.density_metric` | 采用体积加权相对 L2 密度误差 |
| `objective.regularization.type` | 惩罚 vxc 的梯度平方积分，抑制振荡，不约束势的常数项 |
| `objective.regularization.strength` | 正则化权重，示例值需要收敛与稳定性检查 |
| `convergence.density_relative_l2` | 外层相对密度误差阈值 |
| `convergence.gradient_tolerance` | 唯一梯度阈值，映射到 `LBFGSState.g_tol`，当前直接检查回调返回梯度数组的欧氏二范数 |

#### 3.4.1 Davidson 的输入映射和约束

`method` 的正确拼写为 `davidson`。当前初始化接口为：

```cpp
DavidsonSolver_init(context, plan, num_states, nx, ny, nz,
                    nextra_retain, max_subspace_dim,
                    max_iterations, residual_tol);
```

YAML 中 `residual_tolerance` 传给 `residual_tol`；`direction_tolerance` 不在初始化参数表中，成功初始化后赋给 `plan.tNorm_tol`，在调用求解函数前生效。

固定闭壳层占据下，外层驱动先确定 `Nocc = Ne / 2`，再计算 `num_states = Nocc + extra_states`。当前 LiH 为 4 电子、2 条占据空间轨道，`extra_states: 4` 对应实际求解 6 个目标本征态。`extra_states` 是外层规划参数，不是 Davidson 内部自动识别的字段。

`max_subspace_dim: 32` 是本配置选择的示例容量，不是源码自带默认值；它为 6 个目标态留出扩展空间，数值需要结合收敛和显存检查。`max_iterations: 100` 对应结构体的默认上限。`nextra_retain: 3` 和 `direction_tolerance: 1e-8` 沿用结构体默认值。原配置的残差阈值 1e-9 保留；源码结构体默认残差阈值为 1e-8，两者不可混淆。

输入读取后应满足源码检查：

```text
num_states > 0
num_states < max_subspace_dim <= nx * ny * nz
0 <= nextra_retain < max_subspace_dim - num_states
max_iterations > 0
residual_tolerance 有限且 > 0
direction_tolerance 有限且 0 < direction_tolerance < 1
```

子空间满时保留 `num_states + nextra_retain` 个 Ritz 向量；`nextra_retain` 不增加最终输出本征态数，也不是 `extra_states` 的别名。修正方向仅在 `norm_after > direction_tolerance * norm_before` 时加入子空间，该参数不是本征残差阈值。

当前残差定义为 `ri = H ui - εi ui`，用 `cublasDnrm2` 检查 `||ri||₂ <= residual_tol`。所有目标态都满足阈值才返回成功；达到迭代上限或没有新方向时分别返回 `kDavidsonMaxIterations`、`kDavidsonStagnation`，不能当作已收敛。

Davidson 的 QR、正交化和投影使用欧氏内积。若物理轨道采用体积归一化，调用层应保持与 `u = sqrt(dV) * ψ` 一致的表示，并在密度重构时做对应转换，而不是声称求解器内部已自动乘体积权重。预条件器的分母保护阈值在现有实现中固定为 1e-10，本次不新增可配置选项。

#### 3.4.2 L-BFGS 的输入映射和外层控制

| YAML 字段 | C++ 对应 | 赋值时机 |
|---|---|---|
| `optimizer.history` | `LBFGSOptimizer_init` 的 `history_capacity` 参数 | 初始化时 |
| `optimizer.c1`、`optimizer.c2` | `LBFGSState.c1`、`c2` | 初始化前设置，初始化会校验 |
| `convergence.gradient_tolerance` | `LBFGSState.g_tol` | 初始化前设置，避免重复定义梯度阈值 |
| `optimizer.max_linesearch_step` | `LBFGSState.max_linesearch_step` | 成功初始化后设置，初始化内部会先重置为 50 |
| `optimizer.max_iterations` | 外层驱动的调用次数上限 | 外层循环控制，不能写入不存在的状态成员 |

应保证 `history > 0`、`max_iterations > 0`、`max_linesearch_step > 0`、`0 < c1 < c2 < 1`，且梯度阈值有限并满足 `gradient_tolerance >= 0`。正的外层上限和线搜索上限由读取/驱动层校验，不能假定库接口会代为检查所有配置。

`history: 10` 和外层调用上限 100 保留现有配置值。`max_linesearch_step: 50`、`c1: 1e-4`、`c2: 0.9` 沿用实现默认。梯度阈值仍为现有输入的 1e-7；结构体默认 `g_tol=1e-6`，因此读取器必须显式覆盖。

线搜索使用 `WolfeLineSearch`：每次从步长 1 开始，先检查 Armijo 条件，再检查强 Wolfe 曲率条件；若试探点梯度已满足 `g_tol`，可在通过 Armijo 后直接接受并返回收敛。未建立上界时步长翻倍，建立区间后取中点。它不是单纯缩短步长的回溯法，不设置可选择的 `line_search` 方法或输入初始步长。

`LBFGSOptimizer` 每次调用最多接受一步。返回 `kLBFGSConverged` 表示梯度阈值达到；返回 0 不代表整体收敛，也不一定接受了新步。线搜索次数耗尽时函数会输出警告、保留原接受点和历史、设置 `step_size=0` 并返回 0；外层应记录该状态并计入调用上限，避免无限重试。`state.iterations` 仅统计接受步数，`state.evaluations` 统计回调次数，二者都不同于外层调用次数。

`convergence.density_relative_l2` 是外层物理收敛条件，不由通用 L-BFGS 库判断。若梯度已收敛而密度误差仍不满足，应报告这种状态，不能擅自把通用优化收敛当作反演成功。目标函数、正则化和伴随/响应梯度由 `calculate_F` 回调计算，两份求解器源码没有直接实现这些物理目标。

前向方程为：

```text
H[vxc] = −½∇² + vext + vH[ntarget] + vxc
Hψi = εiψi
fi = 2（占据态）或 0（空态），当前闭壳层固定占据
Σi fi = Ne
nKS(r) = Σi fi |ψi(r)|²
```

这里的 2 是非极化空间轨道的自旋简并因子，不得在密度重构时再乘一次 2。固定占据不执行 FD 展宽或通过 FD 方程更新占据。最终有效势输出保留优化器中的势零点，不减费米能，不增加费米能输入参数。

此前建议的目标函数定义为：

```text
ηn² = Σp Vp (nKS,p−ntarget,p)² / Σp Vp ntarget,p²
J = ½ηn² + (α/2) ∫ |∇vxc|² dr
```

当前 L-BFGS 直接对回调提供的数组计算 `sqrt(Σp |g_p|²)`，不自动进行体积加权或 RMS 归一化。因此 `gradient_tolerance` 必须按该欧氏二范数解释。若先得到满足 δJ = Σp Vp gfield,p δvp 的场梯度，且优化变量就是逐点势值 vp，则回调应返回欧氏导数 `g_p = Vp * gfield,p`。不能把场梯度不经转换直接交给优化器，再将其阈值解释为体积加权 RMS。

当前 YAML 不重复设置电荷或电子数。全电子体系可由分子 JSON 的核电荷和 `charge` 确定 Ne，并检查目标矩阵的 Tr(PS) 与之相符，其中 S 为对应基组的重叠矩阵。不能忽略 JSON 电荷而默认体系中性，也不能用普通 Tr(P) 代替 Tr(PS) 或静默缩放目标密度。当前 LiH 的电荷为 0，目标电子数为 4。

### 3.5 输出

| 字段 | 解释 |
|---|---|
| `directory` | 结果目录 |
| `write_veff` | `LOCPOT`：最终 `LBFGSState.d_x` 原值，Hartree 转 eV，不减费米能 |
| `write_density` | `DENSITY`：最终轨道及固定双占据重构的密度，bohr⁻³ |
| `write_density_difference` | `DENSITY_DIFFERENCE`：nKS−ntarget，bohr⁻³ |
| `write_wave` | `ORBITAL_0001` 等：全部求解态的带符号实波函数，bohr⁻³ᐟ²，每个轨道归一化为 1 |
| `write_eigenvalues` | 可选，默认 false；`EIGENVALUES`：全部求解态的编号（从 1 开始）、本征能级（Hartree）及对应的实际占据数；列名为 `state eigenvalue_Ha occupation` |

上述场文件均采用 VASP LOCPOT 文本布局（晶格和原子位置为 Å，x 方向最快，每行最多五个场值），不补边界、不插值。密度仍是 bohr⁻³，不能作为按晶胞体积缩放的 VASP CHGCAR 解读。为精确保留内部节点，文件原点为 `lower+gap`，文件晶胞边长为 `rgrids*gap`，原子坐标减去文件原点；标题记录场单位、原点和实际物理盒上下边界（bohr）。这只是输出采样区域，计算仍采用原来的孤立体系物理边界。

`write_results(context, parameters, grid, lbfgs, davidson, occupation, error)` 由外层在最终收敛后调用；调用方须确保 Davidson 输出对应最终接受的 `lbfgs.d_x`，`occupation.d_occupation` 与这些能级一一对应，且 `d_psi` 是未经原地平方覆盖的欧氏归一化 Ritz 轨道。写出器以 `2*sum(psi²)/dV` 重构闭壳层密度，以 `psi/sqrt(dV)` 写物理轨道，不修改设备数组。同名文件会覆盖；关闭的输出既不创建也不删除已有文件。失败返回 1 并填写错误信息，可能已产生部分文件。

不提供检查点或展开配置输出，读取器不再接受 `write_checkpoint`、`write_resolved_input` 或旧的 `write_vxc` 字段。

## 4. molecule.json

文件采用原计算导出的 PySCF 分子序列化 JSON。无需另外输入结构、基组名称或 AO 顺序；读取器必须保留原始 AO 的定义和排列，不能只凭 `aug-cc-pVTZ` 这个名称重新建立一套可能重排的基组。

当前示例中的关键内容为：

| 字段 | 用途 |
|---|---|
| `_atom` | PySCF 内部原子顺序与坐标，坐标已经是 bohr |
| `_atm` | 原子核电荷及其在 `_env` 中的索引等内部信息 |
| `_bas` | 按原计算顺序排列的 shell 信息，包括中心索引、角动量、原函数数、收缩数和数据偏移 |
| `_env` | PySCF 内部坐标、Gaussian 指数和已处理的基组系数等数值数据 |
| `cart` | 球谐或笛卡尔表示；当前为 `false`，即球谐 AO |
| `charge`、`spin` | 分子电荷及 PySCF 的 Nα−Nβ；当前均为 0 |
| `atom`、`basis`、`unit` | 原始构建输入的描述，不能替代已经构建的内部数组 |

### 4.1 单位与顺序

示例 JSON 的 `unit` 是 `Angstrom`，原始 `atom` 字符串中 LiH 键长为 1.594900；但 `_atom` 和 `_env` 保存的内部键长已经是 3.013924196068817 bohr。使用内部数组时不得因顶层 `unit` 再转换一次。

YAML 中的 `padding` 始终按 bohr 解释，`rgrids` 是无量纲网格数量。JSON 的原始输入单位不会改变本项目的运行参数单位。原子、shell、收缩和角分量顺序必须与密度矩阵行列一致，不能无声重排。

### 4.2 当前 C++ 评价器的范围

`structure.cpp` 直接解码 `_atm`、`_bas`、`_env`，保留 shell、收缩和球谐角分量的原顺序。当前支持无 ECP、spin=0、球谐 s/p/d/f，支持一个 shell 的多个收缩；其余类型明确报错。内部 `_env` 的径向系数不再重复归一化。

部分 JSON 描述字段是字符串，不是直接的 JSON 数值数组。当前 C++ 读取器通过已有 yaml-cpp 解析 JSON 中的内部数值数组，不执行 `atom`、`basis` 等描述字符串，也不按基组名称重建 AO。新版本 PySCF 或未覆盖的文件布局仍需单独验证。

## 5. density_total.npy

### 5.1 已核实的格式与来源

来源为 [n2v 会话](codex://threads/01a0af68-b206-7d62-a86f-4ae5570556e1)。该会话中的生成代码明确为：

```python
dm_mo = solver.make_rdm1(ci, mol.nao_nr(), mol.nelec)
dm = mf.mo_coeff @ dm_mo @ mf.mo_coeff.T
np.save("density_total.npy", dm)
```

因此它是从 PySCF FCI 波函数导出并转换到 AO 表示的一阶总密度矩阵，已包含两个自旋的贡献。转换只改变表示方式，不是密度拟合，也没有经过 n2v 反演。

当前示例文件的实际格式为：

| 项目 | 已检查结果 |
|---|---|
| 容器 | NumPy `.npy`，格式版本 1.0 |
| 数值类型 | `<f8`：小端双精度实数 |
| 数组维度 | `(69, 69)` |
| 存储顺序 | `fortran_order=False`，C 顺序 |
| AO 数量 | JSON 中 shell 展开的球谐 AO 数为 69，与矩阵匹配 |
| 元素有限性 | 全部有限 |
| 最大不对称误差 | 约 2.78×10⁻¹⁶ |

读取后不再进行 MO→AO 变换，也不再乘 2；当前这对文件无需额外的 MO 系数文件。行列顺序必须与配套 JSON 完全一致。

2026-10-24 核对本地文件与来源会话中的 SHA256 完全一致：

```text
density_total.npy  d769704e5a23ef6f1054ec094610cbacaac1e112be6229dc12f109c4f911ca5a
molecule.json      de7943d4f5434b679d7fa363341d1d206a6022303d165eadafa2fc3e4a359636
```

这些哈希用于记录示例文件身份，不是以后每次计算必须匹配的输入参数。

### 5.2 电子数与精度

AO 基组不正交，电子数应按 Ne = Tr(PS) 检查，而不是 Tr(P)。示例矩阵的普通对角和约为 7.7101，不能据此认定 LiH 有约 7.71 个电子；当前全电子中性 LiH 应有 4 个电子。

均匀网格还要检查 Σp dV ntarget,p 与 4 的偏差，以评估采样及有限盒误差。不得通过静默重标定目标密度掩盖这些误差。

FCI 目标是选定有限基组与数值收敛精度下的密度，不等于连续空间的严格精确密度。更换分子或来源文件时仍需检查矩阵维度、对称性、电子数和 AO 顺序；当前格式范围是实数、自旋求和的 AO 一阶密度矩阵。

## 6. 从源表示生成均匀网格目标密度

```text
读取 input.yaml
  → 读取 molecule.json 的结构和基组
  → 按 padding 和 rgrids 构建计算盒与均匀网格
  → 确定全域盒内节点
  → 读取 density_total.npy 的 AO 总密度矩阵
  → 逐点评价 AO（复用一个点的临时缓冲区）
  → 计算 ntarget,p = Σμν Pμν φμ(rp) φν(rp)
  → 检查积分和采样精度
  → 计算固定的目标 Hartree 势
  → 开始势反演
```

当前实数基组使用上式；复杂数情况需单独明确共轭约定。采样得到的是内部节点点值，不是单元平均密度。

网格和目标源不变时，只构建一次目标密度并复用。更改盒子、间距或原子位置后，在新点重新评价即可，不必重新进行源电子结构计算，但原子位置改变时原有密度矩阵是否仍对应期望目标必须由使用者明确。

避免保存全体网格点乘全部 AO 的大矩阵：按点分批评价并收缩，保留最终一维目标密度即可。这里的点批次只控制临时内存，不是物理网格的 block 划分，也无需在源文件中保存网格点坐标。

## 7. 对两个接收会话的实现衔接

### 自适应网格有限差分会话

- 按最新均匀网格约定更新此前自适应网格方案，不再执行粗细网格生成与交界重构。
- 上层将 YAML 的 `stencil_half_width` 原样读入 `finite_difference_halfwidth`，配合当前 `rgrids` 构造索引和算子，不要求块尺寸。
- 所有物理积分和目标函数使用相同的单元体积 dV，不含 halo；进入 Davidson 的轨道采用第 3.4.1 节说明的欧氏表示。
- 电子数积分误差、内部和物理边界差分误差、核附近误差、有限盒误差均需独立评估，不能让反演势的振荡掩盖离散误差。

### smpi.cpp 的 CPU/GPU 会话

- 孤立体系，无 k 点，也不输入 block 分块参数。若实现 MPI，均匀网格的进程间数据分布由并行层决定，不额外建立 AMR 或用户指定的 block 网格。
- 相同 JSON 与 NPY 可供各 rank 对本地拥有的单元评价目标密度；输入数据读入/广播策略由并行层决定，不改变文件物理含义。
- CPU/GPU 路径共用密度含义、体积权重和全局电子数；归约只统计拥有的数据，避免复制数据被重复累加。
- 网格单元的数据所有权与 GPU 批次独立；分批执行不会改变全域均匀网格定义。
- 正交化、投影和占据需保持全局一致；GPU 批次不能独立求解后遗漏批次间子空间耦合。
- 当前输入没有运行时设备/批次配置，不自行增加 `runtime` 顶层组。

## 8. 落实前需固定的事项

以下事项仍需实现或验证：

- 当前球谐 s/p/d/f 范围之外的源文件兼容性与 AO 类型支持。
- 全电子核奇点及对应离散势的处理。
- 将 JSON 电子数与 Tr(PS) 的独立核验接入运行流程。当前构建器记录网格密度积分，但不按未指定的容差拒绝或缩放密度。
- L-BFGS 梯度的响应/伴随实现，采用当前固定占据设置。
- 按现有欧氏梯度和轨道表示实现回调、密度收敛检查及求解器参数到初始化接口的连接；不再实现当前配置不使用的自动谱估计或切比雪夫阶数选择。
- 网格、目标密度与外层 GPU 反演驱动的连接；当前主机数组索引已固定为 `(z*ny+y)*nx+x`。

最小验证应覆盖：C++ AO 与 PySCF 在相同采样点的数值一致性、NPY 维度/类型/存储顺序、P 的对称性、Tr(PS)、均匀网格密度积分、差分算子的解析函数验证、厄米性、本征残差和已知势生成目标后的反演恢复。网格积分误差不能通过静默缩放目标密度掩盖。

## 9. 接收会话使用说明

请读取本文作为当前输入约定的交接资料，覆盖旧上下文中与之冲突的自适应粗细网格、block 尺寸对齐、单独 XYZ/自定义 HDF5 源输入、普通 DFT 的 XC 泛函输入、计算内势平移和预采样目标网格输入等假设。保留两个会话各自已有的设计工作，不因本交接文档自动扩大到编码或数值计算任务。

本文写入共享工作目录并不等于消息已发送到两个会话；实际推送状态以交付会话的工具结果和最终说明为准。

## 10. 当前构建接口

`read_parameters()` 读取 YAML；`creat_StructGrid()` 读取分子文件并建立 `StructGrid`；`read_density_input()` 读取密度矩阵并填充 `grid.target_density`。均使用 0/1 返回状态和错误字符串，失败时保留已有输出。调用示例：

```cpp
InputParameters parameters;
StructGrid grid;
std::string error;
if (read_parameters(input_file, parameters, error) != 0) return 1;
if (creat_StructGrid(parameters, grid, error) != 0) return 1;
if (read_density_input(parameters.system.density_file, grid, error) != 0) return 1;
// grid.target_density 为主机 double 数组，可交给现有 upload() 上传。
```

读取模块依赖 C++17 和 yaml-cpp，无 Python/CUDA 运行依赖。NPY 当前支持版本 1.0、`<f8`、C 顺序、与 AO 数匹配的二维实矩阵，检查数据有限性及对称性。密度评价复用单点 AO 缓冲区，不分配全网格 AO 矩阵。

当前 LiH 示例用 `80 80 80`、三方向 padding=12 得到 512000 个内部点。C++ 与 PySCF 在这些点的最大密度绝对差约为 `3.7e-14`；源矩阵 `Tr(PS)` 约为 4，网格积分约为 `3.9873170964`，保留该离散/有限盒误差，不将密度强制缩放到 4。网格与盒尺寸仍需做精度收敛检查。
