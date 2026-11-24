import torch
from edge_ai.model import EdgeVTON
def test_shape_range():
    with torch.inference_mode():
        y=EdgeVTON().eval()(torch.zeros(1,9,64,64))
    assert y.shape==(1,3,64,64)
    assert torch.isfinite(y).all()
    assert y.min()>=0 and y.max()<=1
