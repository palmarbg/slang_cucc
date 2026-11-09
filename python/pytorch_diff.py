import slangpy as spy
import torch
import pathlib
import numpy as np

# Create a SlangPy device
device = spy.create_torch_device(type=spy.DeviceType.cuda, include_paths=[
    pathlib.Path(__file__).parent.absolute(),
])

# Load the module
module = spy.Module.load_from_file(device, "../slang/example.slang")

x = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device='cuda', requires_grad=True)

result = module.polynomial(a=2, b=8, c=-1, x=x)
print(result)

result.backward(torch.ones_like(result))
print(x.grad)