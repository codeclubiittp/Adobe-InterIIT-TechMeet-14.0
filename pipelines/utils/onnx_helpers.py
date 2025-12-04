import onnxruntime as ort

def log_session_details(onnx_session: ort.InferenceSession, other: dict = None):
    print(f"--- onnx session details ---")
    if other:
        for k, v in other.items():
            print(f"{k}: {v}")
            
    inputs = onnx_session.get_inputs()
    print(f"inputs ({len(inputs)}):")
    for i, input_node in enumerate(inputs):
        print(f"  {i}: name={input_node.name}, shape={input_node.shape}, type={input_node.type}")
        
    outputs = onnx_session.get_outputs()
    print(f"outputs ({len(outputs)}):")
    for i, output_node in enumerate(outputs):
        print(f"  {i}: name={output_node.name}, shape={output_node.shape}, type={output_node.type}")
    print("------------------------------")