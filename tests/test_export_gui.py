import numpy as np
import torch

from chess_nn.export_gui import weight_payload
from chess_nn.model import EvalNet, count_params


def test_weight_payload_round_trips_every_tensor():
    torch.manual_seed(0)
    model = EvalNet(hidden=8, head=(4, 3))
    manifest, blob = weight_payload(model)
    flat = np.frombuffer(blob, dtype="<f4")
    tensors = {t["name"]: flat[t["offset"] : t["offset"] + t["length"]].reshape(t["shape"]) for t in manifest}

    assert np.allclose(tensors["ft_by_feature"], model.ft.weight.detach().numpy().T)
    assert np.allclose(tensors["ft_bias"], model.ft.bias.detach().numpy())
    assert np.allclose(tensors["head1_w"], model.head[1].weight.detach().numpy())
    assert np.allclose(tensors["out_b"], model.out.bias.detach().numpy())
    assert len(flat) == sum(t["length"] for t in manifest) == count_params(model)
