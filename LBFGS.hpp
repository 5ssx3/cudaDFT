#pragma once
#include <cstddef>
struct Context;
// 通用错误沿用 cuda_math.hpp 中的 Status；L-BFGS 专用状态从 200 开始。
enum LBFGSStatus : int {
    kLBFGSSuccess = 0,           // 完成一次更新，尚未达到梯度阈值
    kLBFGSConverged = 200,
    kLBFGSCublasError,
    kLBFGSCusolverError,
    kLBFGSCurandError,
    kLBFGSNonFinite,
    kLBFGSNotDescent,
    kLBFGSInvalidCurvature,
    kLBFGSLineSearchFailed,
    kLBFGSEvaluationError,
    kLBFGSNotInitialized,
};

const char* lbfgs_status_message(int status) noexcept;

using calculate_F = int (*)(
    const Context& context,
    const double* d_x,  // 输入：设备上的试探变量
    double& f,          // 输出：主机上的函数值
    double* d_g,        // 输出：设备上的梯度
    void* user_data     // 外部计算所需的数据
);
struct LBFGSState {
    // 问题规模与历史管理
    size_t n{};                 // 优化变量个数
    int history_capacity{};     // 最大历史组数 m
    int max_linesearch_step{};  // 最大线搜索步数
    int history_count{};        // 当前有效历史组数
    int history_next{};         // 下一组历史的写入位置，循环覆盖

    // GPU 历史数据，布局为 [history_capacity][n]
    double* d_s{};
    double* d_y{};

    // 两层循环使用的标量数组，长度为 history_capacity
    double* d_rho{};             // rho_i = 1 / <s_i, y_i>
    double* d_alpha{};           // 第一层循环计算，第二层循环复用

    // 当前已接受状态，长度均为 n
    double* d_x{};
    double* d_g{};

    // 线搜索试探状态，长度均为 n
    double* d_x_trial{};
    double* d_g_trial{};

    // 搜索方向，也可复用为两层循环的工作向量
    double* d_p{};

    // 主机端标量
    double f{};                  // 当前目标函数值
    double step_size{1.0};       // 线搜索步长
    double gamma{1.0};           // 初始逆 Hessian 缩放：H^(0) = gamma I
    double c1 = 1e-4;
    double c2 = 0.9;
    double g_tol = 1e-6;

    double g_norm{};             // 当前接受点的梯度二范数
    size_t iterations{};         // 累计接受步数
    size_t evaluations{};        // 累计回调次数，含初始化和失败的回调
    int line_search_trials{};    // 最近一次线搜索的试探次数
    // 可用状态；单步结果由返回值表示。计算故障后需要销毁并重新初始化。
    int status = kLBFGSNotInitialized;
};

int destory_LBFGSOptimizer(LBFGSState& lbfgs);
int LBFGSOptimizer_init(Context& context,LBFGSState& lbfgs,int n, 
    int history_capacity, calculate_F evaluate, void* user_data, double* d_x_guess);
// 每次至多接受一步；返回 0 表示完成一步，kLBFGSConverged 表示达到梯度阈值。
// 其他返回值为错误；回调错误和线搜索次数耗尽时，接受点及历史保持不变，可重试。
// context.cublas 必须使用 context.stream，入口标量指针模式必须为 HOST。
// 错误及收敛时自动输出日志，普通成功步骤不输出。
int LBFGSOptimizer(Context& context, LBFGSState& lbfgs, calculate_F evaluate, void* user_data);
void write_log(const LBFGSState& lbfgs, int result, const char* stage) noexcept;
