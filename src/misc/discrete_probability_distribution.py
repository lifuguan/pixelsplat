import torch
from einops import reduce
from jaxtyping import Float, Int64
from torch import Tensor
from einops import rearrange

def sample_discrete_distribution(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Int64[Tensor, "*batch sample"],  # index
    Float[Tensor, "*batch sample"],  # probability density
]:
    *batch, bucket = pdf.shape
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    cdf = normalized_pdf.cumsum(dim=-1)
    samples = torch.rand((*batch, num_samples), device=pdf.device)
    index = torch.searchsorted(cdf, samples, right=True).clip(max=bucket - 1)
    return index, normalized_pdf.gather(dim=-1, index=index)
    
    # num_samples=32
    # samples = torch.rand((*batch, num_samples), device=pdf.device)#随即采了3个概率
    # #试着将sample改成重要性采样,还是采三个点,但是我尽量都采在概率大的地方
    # #这样采样的显著性问题在于对于depth分布不明显的地方,可能估计的depth会i比较差.
    # #在train的时候这样设置,当然可以有利于学到depth一个分布
    # #但是这样测试就会具有一定的随机性 这里pxielsplat相当于用的是均匀采样
    # #所以如何保证测试的时候稳定且好呢用他的topk?
    # #以均匀分布采样p(x)=1 
    # index =torch.range(0,31).cuda()/32
    # index_1=index.broadcast_to(pdf.shape)
    # # for i in gaussians_per_pixel:
    # #         e_depth=1/gaussians_per_pixel*pdf_i[:,:,:,:,i]/(1/32)

    # index = torch.searchsorted(index_1, samples, right=True).clip(max=bucket - 1)#这3个概率对应的depth的位置
    # # index =index/32 #这个是采样点x  px=1
    # pi_x=normalized_pdf.gather(dim=-1,index=index)  #pi(x) 采样点在元分布上的概率
    # f_x=index/32   #fx=x 即depth
    # # e_depth=0   #depth的期望
    # # for i in num_samples:
    # e_depth=pi_x[...,:]*1*f_x[...,:]
    # e_depth=e_depth.mean(dim=-1).unsqueeze(-1)
    # index = torch.searchsorted(cdf, e_depth, right=True).clip(max=bucket - 1)#这3个概率对应的depth的位置 以e_depth期望找到最靠近的index
    # return index, normalized_pdf.gather(dim=-1, index=index),e_depth


def gather_discrete_topk(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Int64[Tensor, "*batch sample"],  # index
    Float[Tensor, "*batch sample"],  # probability density
]:
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    index = pdf.topk(k=num_samples, dim=-1).indices
    return index, normalized_pdf.gather(dim=-1, index=index)
