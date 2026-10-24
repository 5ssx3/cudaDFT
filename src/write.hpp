#pragma once

#include <string>

struct Context;
struct InputParameters;
struct StructGrid;
struct LBFGSState;
struct DavidsonPlan;

// 调用方确认外层已收敛，且 Davidson 输出对应最终 lbfgs.d_x。
// d_psi 必须是未经平方覆盖、欧氏归一化的 Ritz 轨道：[state][z][y][x]。
// 最低 Ne/2 个空间轨道各占据 2 个电子，与反演一致；密度为 2*sum(psi^2)/dV。
// 按开关写 LOCPOT、DENSITY、DENSITY_DIFFERENCE、ORBITAL_0001 等文件。
// write_wave 控制带符号的实空间波函数输出，每个轨道的积分范数为 1。
// write_eigenvalues 写 EIGENVALUES：从 1 开始的轨道编号及本征能级，单位 Hartree。
// 场文件为 VASP LOCPOT 文本布局；势为 eV（不减费米能），密度为 bohr^-3，
// 实轨道为 bohr^-3/2。密度文件不是按晶胞体积缩放的 VASP CHGCAR。
// 文件原点为 lower+gap，晶胞为 rgrids*gap，原子坐标随原点平移；
// 标题记录原点与物理盒边界（bohr）。不补边界、不插值、不修改输入。
// 相同文件名会被覆盖；关闭的输出不创建、不删除已有文件。
// 返回 0 并清空 error 表示成功；失败返回 1 并填写 error，可能已写出部分文件。
int write_results(const Context& context, const InputParameters& parameters,
                  const StructGrid& grid, const LBFGSState& lbfgs,
                  const DavidsonPlan& davidson, std::string& error);
