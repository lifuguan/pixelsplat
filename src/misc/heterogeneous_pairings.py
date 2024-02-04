import torch
from einops import repeat
from jaxtyping import Int
from torch import Tensor

Index = Int[Tensor, "n n-1"]


def generate_heterogeneous_index(
    n: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Index, Index]:
    """Generate indices for all pairs except self-pairs."""
    arange = torch.arange(n, device=device)

    # Generate an index that represents the item itself.
    index_self = repeat(arange, "h -> h w", w=n - 1)

    # Generate an index that represents the other items.
    index_other = repeat(arange, "w -> h w", h=n).clone()
    index_other += torch.ones((n, n), device=device, dtype=torch.int64).triu()
    index_other = index_other[:, :-1]

    return index_self, index_other


def generate_heterogeneous_index_transpose(
    n: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Index, Index]:
    """生成一个可用于“转置”异构索引的索引。第二次应用该索引会反转“转置”。"""
    # 创建一个从0到n-1的整数序列
    arange = torch.arange(n, device=device)
    # 创建一个n x n的全1矩阵，数据类型为int64
    ones = torch.ones((n, n), device=device, dtype=torch.int64)

    # 对arange进行复制，并进行转置和切片操作
    index_self = repeat(arange, "w -> h w", h=n).clone()
    # 在index_self的右上角添加1，左下角保持不变
    index_self = index_self + ones.triu()

    # 对arange进行复制，并进行转置和切片操作
    index_other = repeat(arange, "h -> h w", w=n)
    # 在index_other的左上角减去1，右下角保持不变
    index_other = index_other - (1 - ones.triu())

    return index_self[:, :-1], index_other[:, :-1]
