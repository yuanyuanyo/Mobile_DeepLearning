import ai_edge_torch
import torch
import numpy as np

# Step 1
model = torch.load('full_model.pth', map_location=torch.device('cpu'))
model.eval() 

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


if np.allclose(
    torch_output.detach().numpy(),
    edge_output,
    atol=1e-5,
    rtol=1e-5,
):
    print("Inference result with PyTorch and LiteRT was within tolerance.")
else:
    print("Something went wrong with PyTorch --> LiteRT conversion.")

# Step 6
edge_model.export('final_model.tflite')
print("Model exported as 'final_pruned_model.tflite'.")