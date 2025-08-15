import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class InputIR:
    bits: int
    size: int

@dataclass
class OutputIR:
    bits: int
    size: int

@dataclass
class PartitionIR:
    dim: int
    stride: int
    input_size: int

@dataclass
class MapIR:
    clustering_depth: int
    groups: int

@dataclass
class CNNParams:
    dimension: int
    kernel: int
    stride: int

@dataclass
class ModelIR:
    partition: PartitionIR
    map: MapIR
    input: InputIR
    output: OutputIR
    reduce_op: Optional[str] = None
    cnn: Optional[CNNParams] = None
    @property
    def model_type(self) -> str:
        return "cnn" if self.cnn is not None else "mlp"


PEGASUS_RE = re.compile(
    r"Partition\([^)]*?"
    r"dim\s*=\s*(\d+).*?"
    r"stride\s*=\s*(\d+)[^)]*\).*?"
    r"clustering_depth\s*=\s*(\d+)",
    re.S
)

CNN_PARAM_RE = re.compile(
    r"CNN_dimension\s*=\s*(\d+).*?"
    r"CNN_kernel\s*=\s*(\d+).*?"
    r"CNN_stride\s*=\s*(\d+)",
    re.S
)

INPUT_RE = re.compile(r"bit<\s*(\d+)\s*>\s+input_dim(?:\[(\d+)\]|\d+)", re.M)
OUTPUT_RE = re.compile(r"bit<\s*(\d+)\s*>\s+out(?:\[(\d+)\]|\d+)", re.M)

def parse(text: str) -> ModelIR:
    m = PEGASUS_RE.search(text)
    if not m:
        raise ValueError("Pegasus 语法匹配失败")
    dim, stride, depth = map(int, m.groups())

    # --- 解析输入 ---
    input_matches = INPUT_RE.findall(text)
    if not input_matches:
        raise ValueError("找不到 input 定义")

    input_bits = int(input_matches[0][0])
    input_size = int(input_matches[0][1]) if input_matches[0][1] else len(input_matches)

    # --- 解析输出 ---
    output_matches = OUTPUT_RE.findall(text)
    if not output_matches:
        raise ValueError("找不到 output 定义")

    output_bits = int(output_matches[0][0])
    output_size = int(output_matches[0][1]) if output_matches[0][1] else len(output_matches)

    # --- 组织结构 ---
    part = PartitionIR(dim=dim, stride=stride, input_size=input_size)
    mp = MapIR(clustering_depth=depth, groups=input_size // dim)

    # --- 解析 CNN 参数（可选）---
    cnn = None
    cnn_match = CNN_PARAM_RE.search(text)
    if cnn_match:
        cnn = CNNParams(
            dimension=int(cnn_match.group(1)),
            kernel=int(cnn_match.group(2)),
            stride=int(cnn_match.group(3))
        )

    return ModelIR(
        partition=part,
        map=mp,
        input=InputIR(bits=input_bits, size=input_size),
        output=OutputIR(bits=output_bits, size=output_size),
        reduce_op=None if cnn else "sum",
        cnn=cnn
    )
