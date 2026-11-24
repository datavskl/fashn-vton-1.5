"""Export trained weights to fixed-shape ONNX and check CPU parity."""
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort
from edge_ai.model import EdgeVTON
def main():
    p=argparse.ArgumentParser(); p.add_argument("--weights",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    m=EdgeVTON().eval(); m.load_state_dict(torch.load(a.weights,map_location="cpu",weights_only=True))
    x=torch.rand(1,9,384,256); x[:,6:]=0; x[:,6]=1
    with torch.inference_mode(): expected=m(x).numpy()
    torch.onnx.export(m,x,a.output,input_names=["input"],output_names=["result"],opset_version=17,dynamo=False)
    onnx.checker.check_model(a.output)
    actual=ort.InferenceSession(a.output,providers=["CPUExecutionProvider"]).run(["result"],{"input":x.numpy()})[0]
    np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
    print("CPU export parity passed; mobile execution not verified")
if __name__=="__main__": main()
