# P4G (P4 for Pegasus): MLP Code Generation

P4G 将基于 **Pegasus 语法**定义的多层感知机（MLP）模型，自动编译为可执行的 P4 程序，用于实现高速、可部署的神经网络推理。

## 完整示例

mlp32：

```
/* ───────────────────────────────────────────────
 *  Pegasus model description for mlp32
 * ─────────────────────────────────────────────── */

/* Input 向量：32 维，每个字段 32 bit */
struct InputVec_t {
    bit<32> input_dim[32]; 
}

/* Output 向量：4 维，每个字段 32 bit */
struct OutputVec_t {
    bit<32> out[4];
}

/* Ingress Metadata */
struct ig_metadata_t {
    InputVec_t  input_vec;
    OutputVec_t output_vec;
}
ig_metadata_t meta;

/* Pegasus 主语句 */
meta.output_vec = SumReduce(
    Map(
        Partition(meta.input_vec, dim = 2, stride = 2),
        clustering_depth = 4
    )
);
```

## 输入格式：MLP 模型定义

### 输入向量支持两种结构表示：

#### 写法 1（标准展开）

```p4
struct InputVec_t {
    bit<32> input_dim0;
    bit<32> input_dim1;
    ...
    bit<32> input_dim31;
}
```

#### 写法 2（数组表示）

```p4
struct InputVec_t {
    bit<32> input_dim[32];
}
```

> **说明**：两种写法都等价，表示一个 32 维、每维 32-bit 的输入向量。

## 输出格式

输出向量同理。

#### 写法 1（标准展开）

```p4
struct OutputVec_t {
    bit<32> out0;
    bit<32> out1;
    bit<32> out2;
    bit<32> out3;
}
```

#### 写法 2（数组表示）

```p4
struct OutputVec_t {
    bit<32> out[4];
}
```

## 模型结构约定

典型 MLP 模型结构如下：

```cpp
meta.output_vec = SumReduce(
    Map(
        Partition(meta.input_vec, dim = 2, stride = 2),
        clustering_depth = 4
    )
);
```

表示：

- 输入向量按 dim=2, stride=2 分组（不重叠）
- 每组映射为 4 个隐藏神经元输出
- 最终通过 SumReduce 聚合为 4 个输出字段

如果是 CNN 还需要三个参数，`CNN_dimension`, `CNN_kernel`和`CNN_stride`：

```
meta.output_vec = SumReduce(
    Map(
        Partition(meta.input_vec, dim = 2, stride = 2),
        clustering_depth = 4,
        CNN_dimension= 3,
        CNN_kernel = cnn_kernel,
        CNN_stride = cnn_stride
    )
);
```

## 使用方法

```bash
python codegen.py <model_p4g_file> <output_p4_file>
```

### 参数说明

| 参数名           | 说明                                   |
| ---------------- | -------------------------------------- |
| `model_p4g_file` | 使用 Pegasus Syntax 编写的模型结构定义 |
| `output_p4_file` | 生成的 P4 源文件路径                   |

### 示例命令

以下命令用于生成不同规模的模型对应的 P4 程序：

```bash
python codegen.py mlp16.p4g basic16.p4
python codegen.py mlp32.p4g basic32.p4

python codegen.py cnn1.p4g new-cnn1.p4
python codegen.py cnn2.p4g new-cnn2.p4
```





