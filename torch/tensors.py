import torch
import numpy as np

# Creating tensor from data
data = [[1,2], [3,4]]
x_tensor = torch.tensor(data=data)

# From numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor
x_ones = torch.ones_like(x_tensor) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_tensor, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# shape
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros = torch.zeros(shape)

print(f'{rand_tensor}\n', f'{ones_tensor}\n', f'{zeros}\n')

torch.cuda.is_available() # not available for non-nvidia

# some numpy like operations
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

# concat
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# In-place operations 
# Operations that have a _ suffix are in-place. 
# For example: x.copy_(y), x.t_(), will change x.

print(tensor, "\n")
tensor.add(5)
print(f'No _ {tensor}')
tensor.add_(5)
print(f'With _ {tensor}')