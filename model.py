import numpy as np

from nn import Module, Embedding, TransformerBlock, LayerNorm
from tokeniser import CharacterTokeniser


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
        tokens = token_ids.shape[-1]
        if tokens > self.context_length:
            raise ValueError("sequence exceeds the context length")

        # position vectors are shared across the sequences in a batch
        x = self.token_embedding(token_ids) + self.position_embedding(np.arange(tokens))
        for block in self.blocks:
            x = block(x)
        
        return self.norm(x) @ self.token_embedding.weight.T


def save_model(model, tokeniser, path):
    """saves weights, model settings + the character vocabulary"""
    weights = {f"parameter_{index}": p.data for index, p in enumerate(model.parameters())}
    np.savez(
        path,
        characters=np.array(tokeniser.characters),
        context_length=model.context_length,
        embedding_dim=model.token_embedding.embedding_dim,
        num_heads=model.blocks[0].attention.num_heads,
        num_layers=len(model.blocks),
        **weights,
    )


def load_model(path):
    """rebuilds the model + tokeniser from a saved npz file"""
    with np.load(path, allow_pickle=False) as saved:
        tokeniser = CharacterTokeniser("".join(saved["characters"].tolist()))
        model = GPT(
            tokeniser.vocab_size,
            context_length=saved["context_length"].item(),
            embedding_dim=saved["embedding_dim"].item(),
            num_heads=saved["num_heads"].item(),
            num_layers=saved["num_layers"].item(),
            rng=np.random.default_rng(0),
        )
        for index, parameter in enumerate(model.parameters()):
            values = saved[f"parameter_{index}"]
            # wrong weight shape could be broadcasted
            if values.shape != parameter.shape:
                raise ValueError(f"saved parameter {index} has the wrong shape")
            parameter.data[...] = values
    return model, tokeniser
