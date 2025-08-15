# P4G (P4 for Pegasus): MLP Code Generation

P4G automatically compiles a multilayer perceptron (MLP) model defined using **Pegasus syntax** into an executable P4 program for high-speed, deployable neural network inference.

## Complete Example

mlp32:

```p4
/* ───────────────────────────────────────────────
 *  Pegasus model description for mlp32
 * ─────────────────────────────────────────────── */

/* Input vector: 32 dimensions, each field is 32 bits */
struct InputVec_t {
    bit<32> input_dim[32]; 
}

/* Output vector: 4 dimensions, each field is 32 bits */
struct OutputVec_t {
    bit<32> out[4];
}

/* Ingress Metadata */
struct ig_metadata_t {
    InputVec_t  input_vec;
    OutputVec_t output_vec;
}
ig_metadata_t meta;

/* Pegasus main statement */
meta.output_vec = SumReduce(
    Map(
        Partition(meta.input_vec, dim = 2, stride = 2),
        clustering_depth = 4
    )
);
```

## Input Format: MLP Model Definition

### The input vector supports two structure definitions:

#### Style 1 (fully expanded)

```p4
struct InputVec_t {
    bit<32> input_dim0;
    bit<32> input_dim1;
    ...
    bit<32> input_dim31;
}
```

#### Style 2 (array notation)

```p4
struct InputVec_t {
    bit<32> input_dim[32];
}
```

> **Note**: Both styles are equivalent; they represent a 32-dimensional input vector with each dimension being 32 bits.

## Output Format

The output vector follows the same two styles.

#### Style 1 (fully expanded)

```p4
struct OutputVec_t {
    bit<32> out0;
    bit<32> out1;
    bit<32> out2;
    bit<32> out3;
}
```

#### Style 2 (array notation)

```p4
struct OutputVec_t {
    bit<32> out[4];
}
```

## Model Structure Convention

A typical MLP model structure:

```cpp
meta.output_vec = SumReduce(
    Map(
        Partition(meta.input_vec, dim = 2, stride = 2),
        clustering_depth = 4
    )
);
```

This means:

* The input vector is grouped with `dim = 2` and `stride = 2` (non-overlapping)
* Each group is mapped to 4 hidden neuron outputs
* Finally, `SumReduce` aggregates them into 4 output fields

If it’s a CNN, three additional parameters are required: `CNN_dimension`, `CNN_kernel`, and `CNN_stride`:

```p4
meta.output_vec = SumReduce(
    Map(
        Partition(meta.input_vec, dim = 2, stride = 2),
        clustering_depth = 4,
        CNN_dimension = 3,
        CNN_kernel = cnn_kernel,
        CNN_stride = cnn_stride
    )
);
```

## Usage

```bash
python codegen.py <model_p4g_file> <output_p4_file>
```

### Parameter Description

| Parameter        | Description                                     |
| ---------------- | ----------------------------------------------- |
| `model_p4g_file` | Model definition file written in Pegasus Syntax |
| `output_p4_file` | Path to the generated P4 source file            |

### Example Commands

The following commands generate P4 programs for models of different scales:

```bash
python codegen.py mlp16.p4g basic16.p4
python codegen.py mlp32.p4g basic32.p4

python codegen.py cnn1.p4g new-cnn1.p4
python codegen.py cnn2.p4g new-cnn2.p4
```
