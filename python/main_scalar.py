import slangpy as spy
import numpy as np
import pathlib

# Create a SlangPy device; it will look in the local folder for any Slang includes
device = spy.create_device(include_paths=[
        pathlib.Path(__file__).parent.absolute(),
])

# Load the module
module = spy.Module.load_from_file(device, "../slang/example.slang")

# Call the function and print the result
result = module.add(1.0, 2.0)
print(result)

# SlangPy also supports named parameters
result = module.add(a=1.0, b=2.0)
print(result)

# Create a couple of buffers with 128x128 random floats
a = np.random.rand(128, 128).astype(np.float32)
b = np.random.rand(128, 128).astype(np.float32)

# Call our function and ask for a texture back
result = module.add(a, b, _result='texture')

# Print the first 5x5 values
print(result.to_numpy()[:5, :5])

# Display the result using tev
spy.tev.show(result, name='add random')