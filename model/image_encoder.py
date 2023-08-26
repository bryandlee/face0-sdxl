from typing import Dict, List, Tuple

import torch
from torch import nn

from common.util import Registerable


class BaseBackbone(torch.nn.Module, Registerable):
    pass


class BaseProjection(torch.nn.Module, Registerable):
    pass


class Projection(BaseProjection):
    def __init__(
        self,
        n_layers: int = 2,
        in_dim: int | List[int] = 1792,
        hidden_dim: int = 768,
        out_dim: int = 768,
        layer_norm: bool = False,
    ):
        super().__init__()

        if isinstance(in_dim, int):
            in_dim = [in_dim]

        self.heads = nn.ModuleList()

        for in_dim_i in in_dim:
            head = []
            for layer_i in range(n_layers):
                _in_dim = hidden_dim if layer_i > 0 else in_dim_i
                _out_dim = out_dim if layer_i == n_layers - 1 else hidden_dim
                head.append(nn.Linear(_in_dim, _out_dim))
                if layer_norm:
                    head.append(nn.LayerNorm(_out_dim))
                if layer_i < n_layers - 1:
                    head.append(nn.ReLU())
            self.heads.append(nn.Sequential(*head))

    def forward(self, x: List[torch.Tensor]):
        out = [head(x_i).unsqueeze(1) for head, x_i in zip(self.heads, x)]
        out = torch.cat(out, dim=1)
        return out


def build_backbone(_target_: str = "InceptionResnetV1", **kwargs):
    return BaseBackbone.registry[_target_](**kwargs)


def build_projection(_target_: str = "Projection", **kwargs):
    return BaseProjection.registry[_target_](**kwargs)


def build_encoders(backbone: Dict, projection: Dict) -> Tuple[BaseBackbone, BaseProjection]:
    backbone = build_backbone(**backbone)
    projection = build_projection(**projection)
    return backbone, projection
