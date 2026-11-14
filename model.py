"""从随机初始化训练的因果 Transformer 语言模型。"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class GPTConfig:
    vocab_size: int = 10000
    context_length: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    ffn_hidden: int = 3072

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_length", "n_layer", "n_head", "n_embd", "ffn_hidden"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数，实际为 {value!r}")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, channels = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)

        # 从 [B, T, C] 拆出多个注意力头，变为 [B, H, T, D]。
        def split_heads(a: mx.array) -> mx.array:
            return a.reshape(batch, length, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5, mask="causal"
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, channels)
        return self.proj(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = nn.RMSNorm(config.n_embd)
        self.fc_in = nn.Linear(config.n_embd, config.ffn_hidden)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(config.ffn_hidden, config.n_embd)

    def __call__(self, x: mx.array) -> mx.array:
        # 每个子层先归一化，再通过残差连接保留输入。
        x = x + self.attn(self.attn_norm(x))
        x = x + self.fc_out(self.activation(self.fc_in(self.ffn_norm(x))))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.final_norm = nn.RMSNorm(config.n_embd)
        # 输出头独立于输入词嵌入，不共享权重。
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, tokens: mx.array, last_token_only: bool = False) -> mx.array:
        if not isinstance(tokens, mx.array):
            raise TypeError("tokens 必须是 MLX 数组")
        if tokens.ndim != 2:
            raise ValueError(f"tokens 必须是 [batch, sequence] 二维数组，实际形状为 {tokens.shape}")
        if not mx.issubdtype(tokens.dtype, mx.integer):
            raise TypeError(f"tokens 必须使用整数类型，实际为 {tokens.dtype}")
        batch, length = tokens.shape
        if batch == 0 or length == 0:
            raise ValueError("tokens 的 batch 和 sequence 维度必须非空")
        if length > self.config.context_length:
            raise ValueError(
                f"序列长度 {length} 超过 context_length {self.config.context_length}"
            )

        positions = mx.arange(length)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        if last_token_only:
            x = x[:, -1:, :]
        return self.lm_head(x)

    def count_parameters(self) -> int:
        """返回模型中所有参数的标量数量。"""
        return count_parameters(self)


def count_parameters(model: nn.Module) -> int:
    """计算 MLX 模块的全部参数量，包括已冻结参数。"""
    return sum(parameter.size for _, parameter in tree_flatten(model.parameters()))
