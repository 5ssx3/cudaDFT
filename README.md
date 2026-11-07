# cudaDFT：编译与开始计算

cudaDFT 是面向孤立分子的 C++/CUDA 逆 Kohn–Sham 计算程序。它从分子结构和外部 AO 目标密度矩阵构造实空间目标密度，使用均匀网格有限差分、Davidson 本征求解器和 L-BFGS 优化器反演局域有效势。

## 1. 编译

### 环境要求

- Linux、支持 C++17 的 C++ 编译器。
- NVIDIA GPU、与 CUDA Toolkit 匹配的驱动，以及包含 `nvcc` 的 CUDA Toolkit。
- CMake 3.20 或更新版本。
- yaml-cpp，需提供 CMake 配置和 `yaml-cpp::yaml-cpp` 目标；已验证使用 yaml-cpp 0.8.0。
- Git。仓库保留轻量输入与参考日志，无需下载大型网格输出。

已验证的编译环境为 CUDA 13.2、RTX 5090（计算架构 `120`）和 yaml-cpp 0.8.0。其他 GPU 请按其计算能力选择架构，并使用支持该架构的 CUDA Toolkit。

### 获取源码

```bash
git clone https://github.com/5ssx3/cudaDFT.git
cd cudaDFT
```

当前版本的算例仅包含轻量文件；历史版本中的大型输出使用 Git LFS 保存。

### 准备 yaml-cpp

如果系统已提供上述 yaml-cpp CMake 目标，可直接进入下一节。也可以在项目目录内编译依赖，并安装到自己的用户目录：

```bash
mkdir -p temp
git clone --branch 0.8.0 --depth 1 https://github.com/jbeder/yaml-cpp.git temp/yaml-cpp
cmake -S temp/yaml-cpp -B temp/yaml-cpp-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local" \
  -DYAML_CPP_BUILD_TESTS=OFF \
  -DYAML_CPP_BUILD_TOOLS=OFF
cmake --build temp/yaml-cpp-build --parallel 4
cmake --install temp/yaml-cpp-build
```

### 编译 cudaDFT

先确认编译器和驱动可用：

```bash
nvcc --version
nvidia-smi
```

以下命令在项目根目录执行。将 `cuda_arch=120` 改为目标 GPU 的架构编号，例如 A100 为 `80`、RTX 4090 为 `89`、RTX 5090 为 `120`：

```bash
cuda_arch=120
env -u CPATH -u CPLUS_INCLUDE_PATH cmake -S . -B temp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_HOST_COMPILER=g++ \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
  -DCMAKE_PREFIX_PATH="$HOME/.local"
env -u CPATH -u CPLUS_INCLUDE_PATH cmake --build temp/build --parallel 4
```

可执行文件生成在 `bin/cudaDFT`。`CMAKE_PREFIX_PATH` 应指向 yaml-cpp 的安装前缀；如采用其他位置，请相应修改。清除 `CPATH` 和 `CPLUS_INCLUDE_PATH` 可避免其他编译环境的头文件干扰 CUDA 编译。当前项目没有 `cmake --install` 安装规则，编译后可直接使用该可执行文件。

运行时需要可用 GPU，以及匹配的 CUDA 和 yaml-cpp 运行库。集群上应在分配到 GPU 的作业环境中运行；模块和作业提交方式依集群配置。确认程序可运行后，可删除 `temp/build` 等编译中间目录，保留 `bin/cudaDFT` 和所需运行库。

## 2. 从 LiH 或 Li2 开始计算

### 算例文件

仓库提供 [LiH](test/LiH) 和 [Li2](test/Li2) 两个算例，每个目录包含：

```text
test/LiH/                 # Li2 的目录结构相同
├── input.yaml
├── target/
│   ├── molecule.json
│   └── density_total.npy
├── run.log
└── results/
    └── EIGENVALUES
```

新计算只需 `input.yaml` 和 `target/`：`molecule.json` 提供分子和基组信息；`density_total.npy` 是与之配套的、自旋求和的实数 AO 密度矩阵，形状为 `(nao, nao)`，并非直接存储的网格密度。复用这些输入不需要安装或重新运行 PySCF。当前程序支持闭壳层、非自旋极化、全电子分子。

`run.log` 和 `results/EIGENVALUES` 是已完成计算的参考输出。每个数百 MiB 的势、密度和轨道网格文件不上传，可按输入重新计算生成；原始计算目录中的完整输出保留。启动脚本、调度记录、临时文件和测试文件不包含在算例中。

### 建立运行目录并启动

在项目根目录执行以下命令，将 `case_name=LiH` 改为 `case_name=Li2` 即可运行另一算例：

```bash
case_name=LiH
cudadft_bin="$(realpath bin/cudaDFT)"
mkdir -p "runs/$case_name/target"
cp "test/$case_name/input.yaml" "runs/$case_name/"
cp "test/$case_name/target/molecule.json" "runs/$case_name/target/"
cp "test/$case_name/target/density_total.npy" "runs/$case_name/target/"
cd "runs/$case_name"
"$cudadft_bin" input.yaml > run.log 2>&1
echo "Exit status: $?"
```

请为重复计算使用新的运行目录，以保留已有结果。程序在前台运行；集群上可将运行命令放入自己的 GPU 作业脚本。在另一个终端进入本次运行目录，执行 `tail -f run.log` 查看进度。

程序接受 `cudaDFT input.yaml`；省略参数时读取当前目录的 `input.yaml`。输入中的目标文件和输出目录均相对于 **YAML 文件所在目录**解析，因此可以把输入目录放到任意位置。进入运行目录后启动，也便于将日志保存在对应算例中。

### 主要参数

直接复制算例中的完整 `input.yaml`，然后按需求修改；不要从下面的参数表拼接不完整输入。读取器拒绝未知字段、重复字段和缺失的必填项。

| 参数 | 含义与算例设置 |
| --- | --- |
| `system.molecule_file` / `density_file` | 配套的分子 JSON 和 AO 密度矩阵路径。 |
| `domain.padding` | 分子坐标包围盒各方向两侧的额外长度，单位 bohr；两例均为 `[12, 12, 12]`。 |
| `domain.orbital_boundary` | 当前为 `dirichlet_zero`，轨道在计算盒边界取零。 |
| `mesh.rgrids` | 三个方向的内部网格节点数；两例均为 `240 240 270`，共 15,552,000 点。 |
| `mesh.finite_half_width` | 差分半宽；`4` 对应内部八阶差分格式。 |
| `inversion.initial_potential` | 当前为 `zero`，从零势开始，无重启文件输入。 |
| `inversion.regularization_eta` | 平滑正则化强度；LiH 为 `1e-8`，Li2 为 `1e-9`。 |
| `inversion.relativ_rho_L2tol` | 相对密度误差阈值，两例均为 `5e-5`；字段拼写须保持 `relativ`。 |
| `inversion.extra_states` | 除占据态外额外求解的态数；两例均为 `1`。 |
| `inversion.eigensolver.residual_tolerance` | Davidson 残差阈值，两例均为 `1e-9`。 |
| `inversion.occupations.method` / `temperature` | 两例为 `fixed` / `0`；固定占据时每个占据空间轨道有两个电子，空态为零。 |
| `inversion.optimizer.gradient_tolerance` | 离散目标函数梯度的二范数阈值，两例均为 `5e-6`。 |
| `inversion.optimizer.max_iterations` | 外层优化调用上限，两例均为 `1000`。 |
| `output.directory` / `write_*` | 输出目录及各类结果的写出开关；两例均写入 `./results`，启用全部输出。 |

两例均采用百万级以上网格和双精度 GPU 计算，需要较大的显存和较长时间。网格、计算盒、正则化和求解阈值应联合检查；参考参数不保证其他体系也收敛。修改输入参数后无需重新编译。

### 检查收敛与结果

完整成功运行应满足：进程退出状态为 `0`，最终日志包含 `IKS | CONVERGED`，`Density L2 / tol` 标为 `converged`，且没有后续输出错误。单次 `L-BFGS | CONVERGED` 只代表梯度达标；程序随后才检查相对密度误差 `||rho - rho_target||₂ / ||rho_target||₂`。正则化也参与梯度，因此梯度达标并不自动意味着密度达标。

以下数据来自随仓库保存的参考日志，耗时是在 RTX 5090 上的该次反演时间，不含编译，不能视为其他硬件的性能保证：

| 算例 | 电子数 / 求解态数 | 外层调用次数 | 最终梯度范数 | 相对密度 L2 误差 | 反演时间 |
| --- | --- | --- | --- | --- | --- |
| LiH | 4 / 3 | 205 | `2.320724e-6` | `4.246491e-5` | 约 18.58 小时 |
| Li2 | 6 / 4 | 144 | `3.635772e-6` | `4.474698e-5` | 约 14.46 小时 |

这些输出来自已有运行记录；不同编译环境和硬件可能导致数值及迭代次数差异。

| 输出文件 | 内容与单位 |
| --- | --- |
| `LOCPOT` | 反演的完整有效势，单位 eV。 |
| `DENSITY` | 计算电子密度，单位 bohr⁻³。 |
| `DENSITY_DIFFERENCE` | 计算密度减目标密度，单位 bohr⁻³。 |
| `EIGENVALUES` | 三列：态编号、本征值（Hartree）、实际占据数。 |
| `ORBITAL_0001` 等 | 带符号的实空间轨道，包括额外空态，单位 bohr⁻³ᐟ²。 |

网格输出采用类 VASP 文本布局，物理量及单位以各文件首行说明为准；几何长度以 Å 写出。按算例输入运行后，LiH 会生成三个轨道文件，Li2 会生成四个；这些大型文件不随仓库分发。即使反演失败，程序也可能尝试写出结果，不能仅凭文件存在判定计算成功。
