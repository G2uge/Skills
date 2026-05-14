# xpu-fusion-op-alignment

将 CUDA 融合算子功能对齐到 XPU，覆盖框架层实现、输入校验、测试文件的全流程。

## 用途

当 Paddle 的 XPU 融合算子相比 CUDA 实现缺少功能（如部分旋转、MQA/GQA、自定义 strides 等）时，按照此流程完成从底层验证到框架层实现、再到测试覆盖的完整对齐工作。

## 触发条件

- 用户需要为 XPU 补充 CUDA 已有的某个融合算子功能
- 用户发现 XPU 融合算子在特定场景下报错或结果不正确
- 用户需要将 CUDA 的 `fused_xxx` 算子迁移/对齐到 XPU
- 用户需要对比 CUDA/XPU 算子实现并评估可补充的功能

## 前置依赖

- 已完成 `xpu-operator-tester` 流程（验证 XDNN 底层算子是否支持目标功能）
- 已有 CUDA 实现的 kernel 文件（`paddle/phi/kernels/fusion/gpu/`）
- 已有 XPU 实现的 kernel 文件（`paddle/phi/kernels/fusion/xpu/`）

## 工作流程

### 1. 分析功能差异

阅读 CUDA 和 XPU 的 `fused_xxx_utils.h` 及 `fused_xxx_kernel.cu/.cc`：

```bash
# CUDA 实现
cat paddle/phi/kernels/fusion/gpu/fused_xxx_utils.h
cat paddle/phi/kernels/fusion/gpu/fused_xxx_kernel.cu

# XPU 实现
cat paddle/phi/kernels/fusion/xpu/fused_xxx_utils.h
cat paddle/phi/kernels/fusion/xpu/fused_xxx_kernel.cc
```

逐项对比：

| 对比维度 | CUDA | XPU | 是否可补充 |
|---------|------|-----|-----------|
| 核心功能（如部分旋转） | 支持 | 缺失/部分 | 框架层可实现 |
| `time_major` | 支持 | 可能不支持 | 取决于 XDNN API |
| 自定义 strides | 支持 | 可能不支持 | 取决于 XDNN API |
| K/V batch_size 校验 | 有 | 无 | 框架层可补充 |
| MQA/GQA 校验 | 有 | 无 | 框架层可补充 |
| sin/cos shape 校验 | 有 | 不全 | 框架层可补充 |
| 零尺寸/零 heads | 有 | 无 | 框架层可补充 |

**分类**：
- **可补充**：框架层纯代码修改，不依赖 XDNN 新功能
- **不可补充**：受限于 XDNN API（如 strides 参数缺失、layout 硬编码）

### 2. 修改框架层实现（.h 文件）

#### 2.1 新增功能参数透传

如果 CUDA 通过 `d2` / `freqs_head_dim` 支持部分功能，XPU 需要在 `.h` 的各层函数中透传该参数：

- 顶层 `XXXImpl`：从输入 tensor shape 计算 `freqs_head_dim`
- sin/cos 生成层：接收 `freqs_head_dim`，只生成对应长度
- 底层算子调用层：将 `freqs_head_dim` 传入 XDNN 算子的 `freqs_shape`

#### 2.2 底层算子不支持时的 workaround

若 XDNN 算子不支持部分功能（如 `rotary_embedding_everytwo` 不支持 `freqs_shape[-1] < t_shape[-1]`），在框架层实现 workaround：

```cpp
// 示例：将 sin/cos pad 到完整 head_dim
// 前 freqs_head_dim 维：正常计算值
// 后 head_dim - freqs_head_dim 维：sin=0, cos=1
// 使得旋转公式在额外维度上退化为 copy
```

常用 XDNN API：
- `xpu::constant<T>(ctx, dst, len, value)` — 填充常量
- `xpu::strided_slice_view_update(...)` — 子 tensor 写入

#### 2.3 对齐输入校验

将 CUDA kernel 中的 `PADDLE_ENFORCE_XXX` 迁移到 XPU 的 `.cc` kernel 文件中：

- K/V 的 `batch_size >= q.batch_size`
- MQA/GQA 的 `num_heads` 一致性
- sin/cos 4D 时 `batch_size == 1 && num_heads == 1`

### 3. 修改 kernel 入口文件（.cc 文件）

XPU 的 `fused_xxx_kernel.cc` 通常比 CUDA 的 `.cu` 简单，需要补充：

1. `numel <= 0` 提前返回
2. K/V batch_size 下界检查
3. MQA/GQA num_heads 检查
4. V 存在时 K 必须存在检查

```cpp
template <typename T, typename Context>
void FusedRopeKernel(...) {
  int64_t numel = q.numel();
  if (numel <= 0) return;

  auto batch_size = q.dims()[0];
  auto num_heads = q.dims()[2];

  // K validation
  if (k && k->numel() > 0) {
    PADDLE_ENFORCE_LE(batch_size, k->dims()[0], ...);
  }

  // V validation (with MQA/GQA checks)
  if (v && v->numel() > 0) {
    PADDLE_ENFORCE_EQ(k_num_heads == v_num_heads, ...);
    PADDLE_ENFORCE_EQ(
        num_heads == v_num_heads || num_heads % v_num_heads == 0, ...);
    PADDLE_ENFORCE_LE(batch_size, v->dims()[0], ...);
  }

  // launch
  ...
}
```

### 4. 修改测试文件

#### 4.1 参考函数修改

XPU 的参考函数（Python）必须支持新增功能（如部分旋转）：

```python
def mult_qkv_rotate_every_two(value, cos_tensor, sin_tensor):
    rot_dim = cos_tensor.shape[-1]
    value_rot, value_pass = value[..., :rot_dim], value[..., rot_dim:]
    # 旋转 value_rot
    query = ...
    return paddle.concat([query, value_pass], axis=-1)
```

#### 4.2 测试基础设施

`get_inputs` 新增 `rotary_percent` 参数：

```python
def get_inputs(self, seed, with_sin_cos, rotary_percent=1.0, dtype="float32"):
    pe_head_dim = int(tensor_q.shape[3] * rotary_percent)
    tensor_sin, tensor_cos = get_sin_cos_tensor(..., pe_head_dim, dtype)
```

注意：`test_static` 中调用 `get_inputs` 需使用关键字参数避免位置错位：

```python
# 错误：self.get_inputs(self.seed, True, self.dtype)
# 正确：
self.get_inputs(self.seed, True, dtype=self.dtype)
```

#### 4.3 补充测试场景

参考 CUDA 测试文件，为 XPU 补充以下场景：

| 场景 | CUDA 有 | XPU 需补充 |
|------|---------|-----------|
| 部分旋转（rotary_percent=0.5） | ✅ | ✅ 核心功能 |
| MQA 模式 | ✅ | ✅ |
| GQA 模式 | ✅ | 可能已有 |
| 零尺寸 tensor | ✅ | ✅ |
| 零 num_heads | ✅ | ✅ |
| 只有 Q/K/V | ✅ | 可选 |
| time_major | ✅ | ❌ XPU 不支持 |
| 错误参数 | ✅ | 可选 |

### 5. 编译

#### 5.1 增量编译 phi 库

```bash
cd /root/paddlejob/tmp/repos/Paddle/build
make phi -j$(nproc)
```

这会重新编译所有修改过的 `.cc` 和 `.h` 文件到 `libphi_core.so`。

#### 5.2 重新链接 libpaddle.so

如果只替换 `libphi_core.so`，可能出现符号不兼容（`undefined symbol`）。需要同时重新链接 `libpaddle.so`：

```bash
cd /root/paddlejob/tmp/repos/Paddle/build/python
make copy_libpaddle -j$(nproc)
```

#### 5.3 验证库时间戳

```bash
ls -la build/python/paddle/libs/libphi_core.so
ls -la build/python/paddle/base/libpaddle.so
```

两者时间戳都应为最新。

### 6. 运行测试

#### 6.1 使用 build 目录的 Python 包

不要直接替换系统安装的 `.so`，而是通过 `PYTHONPATH` 指向 build 目录：

```bash
cd /root/paddlejob/tmp/repos/Paddle/test/xpu
PYTHONPATH=/root/paddlejob/tmp/repos/Paddle/build/python:/root/paddlejob/tmp/repos/Paddle/test/legacy_test \
  python test_fused_rotary_position_embedding_op_xpu.py
```

#### 6.2 分阶段验证

先跑单个核心测试确认环境正常：

```bash
python test_fused_xxx_op_xpu.py XPUTestFusedXXX.test_fused_xxx
```

再跑新增功能测试：

```bash
python test_fused_xxx_op_xpu.py XPUTestFusedXXX.test_fused_xxx_rotary_percent
```

最后跑全部测试：

```bash
python test_fused_xxx_op_xpu.py
```

#### 6.3 预期结果

全部测试应显示 `OK`。如有失败，检查：
- 参考函数是否正确实现了部分旋转/copy 逻辑
- `get_inputs` 的 `rotary_percent` 是否被正确传递
- 静态图测试中 feed 的 sin/cos shape 是否与输入一致

## 输出规范

- 框架层修改文件：`paddle/phi/kernels/fusion/xpu/fused_xxx_utils.h`、`fused_xxx_kernel.cc`
- 测试文件：`test/xpu/test_fused_xxx_op_xpu.py`
- 运行方式记录：使用 `PYTHONPATH` 而非替换系统 `.so`
- 功能差异对比表：明确标注哪些可补充、哪些受限于 XDNN

## 注意事项

1. **不要只替换 `libphi_core.so`** — `libpaddle.so` 也必须重新链接，否则会出现 `undefined symbol` 错误
2. **`get_inputs` 签名变更后检查所有调用点** — 静态图测试中的 `self.get_inputs(self.seed, True, self.dtype)` 会错位到 `rotary_percent` 参数
3. **参考函数必须支持部分旋转** — 否则即使算子实现正确，测试也会失败
4. **XDNN 算子 pad workaround 的数学正确性** — 例如 `sin=0, cos=1` 使得 `out[d] = q[d]*1 ± q[paired]*0 = q[d]`，需验证边界情况
