from dataclasses import dataclass
from torch.nn import Module, Linear, GELU, Dropout, LayerNorm

@dataclass
class PaircheckHeadConfig:
    x_dim: int = 2048
    hidden_dim_1: int = 256
    drop_1: float = 0.2
    hidden_dim_2: int = 64
    drop_2: float = 0.1

class PaircheckHead(Module):
    def __init__(self, config: PaircheckHeadConfig):
        super().__init__()

        self.gelu = GELU()

        self._linear_1 = Linear(config.x_dim, config.hidden_dim_1)
        self._h1_norm = LayerNorm(config.hidden_dim_1)
        self._h1_dropout = Dropout(config.drop_1)

        self._linear_2 = Linear(config.hidden_dim_1, config.hidden_dim_2)
        self._h2_norm = LayerNorm(config.hidden_dim_2)
        self._h2_dropout = Dropout(config.drop_2)

        self._linear_3 = Linear(config.hidden_dim_2, 1)

    def forward(self, x):
        l1 = self._linear_1(x)
        n1 = self._h1_norm(l1)
        a1 = self.gelu(n1)
        h1 = self._h1_dropout(a1)

        l2 = self._linear_2(h1)
        n2 = self._h2_norm(l2)
        a2 = self.gelu(n2)
        h2 = self._h2_dropout(a2)

        y = self._linear_3(h2)

        return y.squeeze(-1)
