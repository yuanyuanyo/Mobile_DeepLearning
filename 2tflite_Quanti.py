import ai_edge_torch
import torch
import numpy as np
import tensorflow as tf


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 1
model = torch.load('final_pruned_model.pth', map_location=torch.device('cpu'))
model.eval()  


original_params = count_model_parameters(model)
print(f"Original PyTorch Model Parameters: {original_params}")

# Step 2
device = torch.device("cpu")
model.to(device)


sample_inputs = (torch.randn(1, 3, 224, 224).to(device),)  

# Step 3
torch_output = model(*sample_inputs)

# Step 4
edge_model = ai_edge_torch.convert(model, sample_inputs)

# Step 5
edge_output = edge_model(*sample_inputs)


if np.allclose(torch_output.detach().numpy(), edge_output, atol=1e-5, rtol=1e-5):
    print("Inference result with PyTorch and LiteRT was within tolerance.")
else:
    print("Warning: PyTorch → LiteRT conversion may have issues.")

# Step 6
print("Starting TensorFlow Lite Quantization...")


tfl_converter_flags = {'optimizations': [tf.lite.Optimize.DEFAULT]}


tfl_drq_model = ai_edge_torch.convert(
    model, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags
)


quantized_params = count_model_parameters(model)
print(f"Quantized Model Parameters (approximate): {quantized_params}")

# Step 7
tfl_drq_model.export('final_pruned_model_tflite_quant.tflite')
print("TFLite Quantized Model exported as 'final_pruned_model_tflite_quant.tflite'.")