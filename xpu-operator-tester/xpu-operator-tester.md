# xpu-operator-tester

验证 XDNN（XPU Deep Neural Network）底层 C++ 算子的功能边界与特性支持性。

## 用途

当需要确认某个 XDNN 底层算子是否支持特定输入形状、边界条件或功能变体（如部分旋转位置编码、变长序列、不同数据类型等）时，直接编写独立的 C++ 测试程序调用 XDNN API，绕过 Paddle 框架层的任何封装与预处理，获取算子最原子的行为特征。

## 触发条件

- 用户询问某个 XPU 底层算子（`xpu::xxx`）是否支持某功能
- 用户需要验证 XDNN 算子在特定形状/参数下的行为
- 用户怀疑框架层代码与底层算子之间的兼容性问题
- 用户需要为 XPU 算子编写独立的 C++ 正确性验证

## 工作流程

### 1. 定位算子声明

在 Paddle 构建目录下查找 XDNN 头文件中的算子声明：

```bash
find build/third_party/xpu/src/extern_xpu/ -name "*.h" | xargs grep -l "rotary_embedding_xxx"
```

重点阅读：
- 参数列表（特别是 `shape` 参数的含义）
- 模板类型约束
- 默认值与可选参数

### 2. 查找 GPU 参考实现（如有）

如果 Paddle 有对应的 GPU 实现，阅读其 kernel 代码以理解算子的预期数学行为：

```
paddle/phi/kernels/fusion/gpu/fused_xxx_utils.h
paddle/phi/kernels/fusion/gpu/fused_xxx_kernel.cu
```

这有助于推断：
- 旋转/计算的精确公式
- 索引方式（layout、stride）
- 部分功能的支持方式（如 partial RoPE 是"前 d2 维旋转"还是"后 d2 维旋转"）

### 3. 编写 C++ 单测程序

单测程序结构：

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "xpu/xdnn.h"

// 1. 辅助函数：内存分配与拷贝
int allocate_and_copy(void** dev_ptr, const void* host_ptr, size_t size);
int copy_to_host(void* host_ptr, const void* dev_ptr, size_t size);

// 2. CPU Reference 实现
// 根据 GPU kernel 或数学公式编写，这是判断"正确输出"的基准
void ref_op_xxx(const float* input, const float* sin, const float* cos,
                float* out, int batch, int seq_len, ...);

// 3. XPU 调用封装
TestResult run_xpu_op_xxx(...);

// 4. 测试场景
void test_full_shape();      // 正常形状，用于校准 ref
void test_partial_shape();   // 目标边界/功能测试
void test_grad();            // 如有需要，测试反向

int main() {
    xpu_set_device(0);
    test_full_shape();       // 先确保 ref 和 XPU 输出一致
    test_partial_shape();    // 再测试目标功能
    return 0;
}
```

**关键原则**：
- 先用**正常形状**校准 CPU ref，确保 ref 公式和算子完全一致
- 再测试**目标边界条件**，对比 XPU 输出与 ref
- 如果 XPU 返回非 0 错误码，记录错误码含义
- 如果 XPU 返回 0 但数值不匹配，说明算子"静默错误"

### 4. 构造输入数据

使用**可追踪的 pattern**（如 `i * 0.1f + 1.0f` 或 `[1, 2, 3, ...]`），避免随机数，方便手动验证每个位置的数值。

### 5. 编译与运行

```bash
XPU_INCLUDE=.../xpu/include
XDNN_INCLUDE=.../xdnn/include
XPU_LIB=.../install/xpu/lib

g++ -std=c++11 \
  -I"${XPU_INCLUDE}" -I"${XDNN_INCLUDE}" \
  test_xxx.cc \
  -L"${XPU_LIB}" -lxpuapi -lcudart -lxpucuda -lxpurt \
  -Wl,-rpath,"${XPU_LIB}" \
  -o test_xxx

./test_xxx
```

### 6. 形成测试报告

将结果整理为 Markdown 报告：

- 测试目的
- 环境信息
- 测试方法（含 ref 公式）
- 结果表格（返回值 + 数值对比）
- 结论（支持/不支持/静默错误）
- 对框架层的启示

## 输出规范

- 测试代码放入 `/root/paddlejob/tmp/test/`（或用户指定目录）
- 报告文件命名为 `test_xxx_report.md`
- 报告中必须包含 CPU ref 公式，说明"正确输出是什么"
- 结论必须明确：算子是否支持目标功能，以及支持的维度（前向/反向/形状/边界）

## 注意事项

- XDNN 算子对非法输入可能不报错（返回 SUCCESS 但输出错误），因此**必须同时检查返回值和数值结果**
- `freqs_shape` 和 `t_shape` 的长度一致性是常见陷阱，需分别测试 `==`、`>`、`<` 三种情况
- 当 GPU 和 XPU 的实现公式有细微差异时（如 sin/cos 索引方式），以 XPU 实际输出为准校准 ref，再测试边界
