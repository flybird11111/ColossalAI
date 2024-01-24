"""
Shardformer Benchmark
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
from einops import rearrange

import colossalai
from colossalai.kernel.cuda_native import AttnMaskType, ColoAttention


def data_gen(batch_size, seq_length_s, seq_length_l, num_head, head_dim):
    query = torch.rand(batch_size, num_head, seq_length_l, head_dim, dtype=torch.float16, device="cuda")
    key = torch.rand(batch_size, num_head, seq_length_s, head_dim, dtype=torch.float16, device="cuda")
    value = torch.rand(batch_size, num_head, seq_length_s, head_dim, dtype=torch.float16, device="cuda")
    attn_mask = torch.rand(batch_size, 1, seq_length_l, seq_length_s, dtype=torch.float16, device="cuda")
    return dict(query=query, key=key, value=value, attn_mask=attn_mask)


def data_gen_for_causal(batch_size, seq_length_s, seq_length_l, num_head, head_dim):
    query = torch.rand(batch_size, num_head, seq_length_l, head_dim, dtype=torch.float16, device="cuda")
    key = torch.rand(batch_size, num_head, seq_length_s, head_dim, dtype=torch.float16, device="cuda")
    value = torch.rand(batch_size, num_head, seq_length_s, head_dim, dtype=torch.float16, device="cuda")
    return dict(query=query, key=key, value=value)


def pytorch_flash_run(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    # with torch.backends.cuda.enable_flash_sdp(True):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
    attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
    return attn_output


def coloattention_flash(attention, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn_mask_type = None
    flash_attention_mask = None
    if attn_mask is not None:
        flash_attention_mask = ~(attn_mask[:, :, -1].squeeze(1).to(torch.bool)).contiguous()
        attn_mask_type = AttnMaskType.padding
        if is_causal:
            attn_mask_type = AttnMaskType.paddedcausal
    elif is_causal:
        attn_mask_type = AttnMaskType.causal

    attn_output = attention(query, key, value, attn_mask=flash_attention_mask, attn_mask_type=attn_mask_type)
    return attn_output


BATCH, N_HEADS, N_CTX, D_HEAD = 32, 16, 4096, 256

# vary seq length for fixed head and batch=4
configs = [
    triton.testing.Benchmark(
        x_names=["seq_length"],
        x_vals=[2**i for i in range(10, 13)],
        line_arg="provider",
        line_vals=["pytorch_flash", "coloattention_flash"],
        line_names=["pytorch_flash", "coloattention_flash"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name=f"flash-benchmark-run-bs{BATCH}-n_head{N_HEADS}-h_dim{D_HEAD}",
        args={"batch_size": BATCH, "num_head": N_HEADS, "head_dim": D_HEAD, "dtype": torch.float16},
    )
]


@triton.testing.perf_report(configs)
def bench_shardformer(batch_size, num_head, head_dim, seq_length, provider, dtype=torch.float32, device="cuda"):
    warmup = 10
    rep = 100
    # prepare data
    data = data_gen_for_causal(batch_size, seq_length, seq_length, num_head, head_dim)
    if provider == "pytorch_flash":
        fn = lambda: pytorch_flash_run(**data, is_causal=False)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    if provider == "coloattention_flash":
        attention = ColoAttention(embed_dim=head_dim, num_heads=num_head)
        fn = lambda: coloattention_flash(attention, **data, is_causal=False)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms


# start benchmark, command:
# torchrun --standalone --nproc_per_node=2 performance_benchmark.py
if __name__ == "__main__":
    colossalai.launch_from_torch({})
    bench_shardformer.run(save_path=".", print_data=dist.get_rank() == 0)
