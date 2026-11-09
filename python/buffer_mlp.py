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
slang_module = spy.Module.load_from_file(slang_device, "../slang/buffer_mlp_fixed.slang")

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
class SlangPyModel(nn.Module):
    def __init__(self, model:nn.Module, slang_module:spy.Module, device):
        super().__init__()
        self.model = model
        self.slang_module = slang_module
        self.device = device

        self.weight_list = []
        self.layer_list = []

        for layer in model:
            self.layer_list.append(layer.in_features)
            break
        
        for layer in model:
            if isinstance(layer, nn.Linear):
                self.weight_list.append(layer.weight.view(-1))
                self.weight_list.append(layer.bias.view(-1))
                self.layer_list.append(layer.out_features)
            elif isinstance(layer, nn.LeakyReLU):
                self.layer_list.append(-1)
            elif isinstance(layer, nn.Tanh):
                self.layer_list.append(-2)
            elif isinstance(layer, nn.Sigmoid):
                self.layer_list.append(-3)

    def forward(self, x):
        return self.slang_module.forward(
            spy.grid(shape=(x.shape[0],)),
            uvs=x,
            weights=torch.cat(self.weight_list, dim=0),
            layers=torch.tensor(self.layer_list, dtype=torch.int32, device=self.device)
        )
    
slangpy_model = SlangPyModel(torch_model, slang_module=slang_module, device=torch_device)


# Training Loop
batch_size = 16
num_epochs = 10

torch_model.train()
slangpy_model.train()

print("Training Slang Neural Texture...")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Sample random UV coordinates in [0, 1]
    uvs = torch.rand((batch_size, 2), device=torch_device, dtype=torch.float32, requires_grad=True)
    y = ((uvs[:, 0] - 0.5) ** 2 + (uvs[:, 1] - 0.5) ** 2 < 0.25).to(torch.float32)

    # Compute forwardpass using Slang
    y_hat = slangpy_model(uvs)
    loss = criterion(y_hat, y)
    loss.backward()

    optimizer.step()
    
    if epoch % 100 == 0:
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