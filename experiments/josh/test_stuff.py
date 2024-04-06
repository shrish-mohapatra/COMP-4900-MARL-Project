import torch
import random
import numpy as np

# shuffled_block_indices = torch.randperm(3).repeat(10, 1) * 6
# random.shuffle(shuffled_block_indices)
# shuffled_block_indices = (torch.arange(2) * 6)[torch.randperm(2)]
# print(shuffled_block_indices)
# shuffled_indices = (shuffled_block_indices.unsqueeze(1).repeat(1, 6) + torch.arange(6)).view(1, -1)
# print(np.random.permutation(3))
# print(shuffled_indices)

ten = torch.arange(3).repeat(0)
comm = [
    torch.rand(3),
    torch.rand(3),
    ten
]
obs = torch.cat([*comm], dim=-1)
print(obs)