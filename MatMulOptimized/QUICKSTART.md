# MatMul优化算子 - 快速入门

## 概述

这是为华为昇腾AI算法挑战赛优化的MatMul(矩阵乘法)算子实现，基于CATLASS高性能模板库。

## 快速开始

### 1. 环境准备

确保已安装CANN 8.3.RC1或更高版本：

```bash
# 设置CANN环境（选择与你安装路径匹配的命令）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 或
source $HOME/Ascend/ascend-toolkit/set_env.sh
```

验证环境：
```bash
npu-smi info  # 查看NPU设备信息
```

### 2. 编译项目

```bash
cd competition/MatMulOptimized

# 赋予构建脚本执行权限
chmod +x build.sh

# 编译（针对Ascend910B）
./build.sh -v Ascend910B
```

编译选项：
- `-v <版本>`: 指定SOC版本 (Ascend910B, Ascend310P等)
- `-c`: 清理构建目录
- `-d`: Debug模式构建
- `-h`: 显示帮助信息

### 3. 运行测试

```bash
cd output/bin

# 运行性能测试
./benchmark_matmul 0
```

预期输出：
```
==================================================
  MatMul Optimized Performance Benchmark
==================================================
Test Size                       | Latency                 | Throughput        | Accuracy
----------------------------------------------------------------------------------------------------
Tiny (16x16x16)                 | Avg:    0.025 ms       | Throughput:    0.080 GB/s | PASS
...
==================================================
Benchmark completed!
==================================================
```

## 项目结构

```
MatMulOptimized/
├── op_kernel/              # NPU核函数实现
│   ├── matmul_optimized.cpp
│   └── matmul_optimized_tiling.h
├── op_host/                # Host侧Tiling
│   └── matmul_optimized.cpp
├── test/                   # 测试和基准
│   └── benchmark_matmul.cpp
├── CMakeLists.txt          # CMake构建配置
├── build.sh                # 构建脚本
├── README.md               # 详细文档
├── TECHNICAL_REPORT.md     # 技术报告
└── QUICKSTART.md           # 本文件
```

## 核心优化技术

| 技术 | 说明 | 性能提升 |
|------|------|----------|
| **CATLASS模板库** | 高性能GEMM模板库 | ★★★★★ |
| **自适应分块** | 根据矩阵大小动态调整tile | 20-30% |
| **Padding优化** | 自动检测并优化内存对齐 | 5-10% |
| **多核并行** | 充分利用24个Cube核心 | 2-4x |
| **Ping-Pong流水线** | 数据传输与计算重叠 | 15-25% |

## 预期性能

### 延迟 (ms)

| 规模 | Baseline | 优化后 | 提升 |
|------|----------|--------|------|
| 256³ | 0.8 | 0.4 | **50%** |
| 1024³ | 25.0 | 9.0 | **64%** |
| 4096³ | 1400.0 | 380.0 | **73%** |

### 吞吐量 (GB/s)

| 规模 | Baseline | 优化后 | 提升 |
|------|----------|--------|------|
| 256³ | 0.16 | 0.32 | **100%** |
| 4096³ | 0.52 | 1.92 | **269%** |

## 使用场景

本优化算子适用于：
- 大规模矩阵乘法 (M,N,K ≥ 256)
- 批量矩阵运算
- 神经网络中的全连接层
- 注意力机制中的矩阵运算

## 限制条件

- 输入数据类型: float16
- 输出数据类型: float16
- 矩阵布局: RowMajor × ColumnMajor
- 硬件要求: Ascend910B或兼容型号

## 故障排除

### 编译错误

**问题**: `ascendc_kernel_cmake not found`
**解决**: 检查CANN环境是否正确设置
```bash
echo $ASCEND_HOME_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**问题**: 模板编译错误
**解决**: 确保GCC版本 ≥ 7.5
```bash
gcc --version
```

### 运行时错误

**问题**: `Device not found`
**解决**: 检查NPU设备状态
```bash
npu-smi info
```

**问题**: 精度验证失败
**解决**:
1. 检查输入数据是否正确初始化
2. 确认浮点精度设置
3. 查看详细错误信息

## 参考文档

- [完整README](README.md) - 详细技术文档
- [技术报告](TECHNICAL_REPORT.md) - 深度优化分析
- [CATLASS文档](../../catlass/docs/) - 模板库使用指南

## 提交流览

**比赛**: 昇腾AI算法挑战赛-中阶赛
**任务**: MatMul算子优化
**完成日期**: 2025-02-23

**优化要点**:
- ✅ CATLASS高性能模板库
- ✅ 自适应多策略分块
- ✅ 智能Padding优化
- ✅ 布局特定Tile形状
- ✅ 多核并行计算
- ✅ Ping-Pong流水线

---

**快速命令参考**

```bash
# 环境设置
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 编译
cd competition/MatMulOptimized
chmod +x build.sh
./build.sh -v Ascend910B

# 测试
cd output/bin
./benchmark_matmul 0

# 清理重建
./build.sh -v Ascend910B -c
```
