import onnxruntime as ort
from pprint import pprint

def log_session_details(onnx_session: ort.InferenceSession):
    SLEN=40
    inputs = [
        {
            "name": inp.name,
            "type": inp.type,
            "shape": inp.shape
        }
        for inp in onnx_session.get_inputs()
    ]

    outputs = [
        {
            "name": out.name,
            "type": out.type,
            "shape": out.shape
        }
        for out in onnx_session.get_outputs()
    ]

    print("-"*SLEN)
    print("Model Inputs")
    print("-"*SLEN)
    pprint(inputs)
    print("-"*SLEN)

    print("\n")
    print("-"*SLEN)
    print("Model Outputs")
    print("-"*SLEN)
    pprint(outputs)
    print("-"*SLEN)