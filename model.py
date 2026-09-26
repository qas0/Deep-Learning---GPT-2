import numpy as np

from nn import Module, Embedding, TransformerBlock, LayerNorm


class GPT(Module):
    
        
    def __init__(self, vocab_size, context_length, embedding_dim, num_heads, num_layers, rng=None):
        if num_layers <= 0:
            raise ValueError("layer count must be positive")
        rng = np.random.default_rng() if rng is None else rng
        self.token_embedding = Embedding(vocab_size, embedding_dim, rng=rng)
        self.position_embedding = Embedding(context_length, embedding_dim, rng=rng)
        self.blocks = [
            TransformerBlock(embedding_dim, num_heads=num_heads, rng=rng)
            for _ in range(num_layers)
        ]
        self.norm = LayerNorm(embedding_dim)
        self.vocab_size = vocab_size
        self.context_length = context_length

    def forward(self, token_ids):
        token_ids = np.asarray(token_ids)
        if token_ids.ndim not in (1, 2) or token_ids.size == 0:
            raise ValueError("expected nonempty token IDs shaped (tokens,) or (batch, tokens)")
        tokens = token_ids.shape[-1]
        if tokens > self.context_length:
            raise ValueError("sequence exceeds the context length")

        # position vectors are shared across the sequences in a batch
        x = self.token_embedding(token_ids) + self.position_embedding(np.arange(tokens))
        for block in self.blocks:
            x = block(x)
        
        return self.norm(x) @ self.token_embedding.weight.T
