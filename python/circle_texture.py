import torch
import torch.nn as nn
import slangpy as spy
import numpy as np
from PIL import Image
import pathlib

# Create a SlangPy device
slang_device = spy.create_torch_device(type=spy.DeviceType.cuda, include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load the module
slang_module = spy.Module.load_from_file(slang_device, "../slang/circle_texture.slang")


# Init model parameters
torch_device = torch.device("cuda")

torch_model = nn.Sequential(
    nn.Linear(2, 32),
    nn.LeakyReLU(0.1),
    nn.Linear(32, 32),
    nn.LeakyReLU(0.1),
    nn.Linear(32, 1),
    nn.Sigmoid()
).to(torch_device)

optimizer = torch.optim.Adam(
    torch_model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()

# Create s SlangPy equivalent of the PyTorch model parameters
layer1 = torch_model[0]
layer2 = torch_model[2]
layer3 = torch_model[4]

weights1 = layer1.weight
bias1 = layer1.bias

weights2 = layer2.weight
bias2 = layer2.bias

weights3 = layer3.weight
bias3 = layer3.bias

# Training Loop
batch_size = 4096*4
num_epochs = 2000

print("Training Slang Neural Texture...")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Sample random UV coordinates in [0, 1]
    uvs = torch.rand((batch_size, 2), device=torch_device, dtype=torch.float32, requires_grad=True)
    y = ((uvs[:, 0] - 0.5) ** 2 + (uvs[:, 1] - 0.5) ** 2 < 0.25).to(torch.float32)

    # Compute forwardpass using Slang
    y_hat = slang_module.forward(
        spy.grid(shape=(batch_size,)),
        uvs=uvs,
        weights1=weights1,
        bias1=bias1,
        weights2=weights2,
        bias2=bias2,
        weights3=weights3,
        bias3=bias3
    )
    
    loss = criterion(y_hat, y)
    loss.backward()

    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Step {epoch:04d} | Loss: {loss.mean().item():.5f}")



# Inference
print("Rendering trained procedural texture...")
res = 256
grid_x, grid_y = torch.meshgrid(
    torch.linspace(0, 1, res, device=torch_device),
    torch.linspace(0, 1, res, device=torch_device),
    indexing="ij"
)
eval_uvs = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
output_image = torch_model(eval_uvs).reshape(-1, 1)

img_np = output_image.reshape(res, res).clamp(0.0, 1.0).detach().cpu().numpy()
Image.fromarray((img_np * 255).astype(np.uint8)).save("out/procedural_circle.png")