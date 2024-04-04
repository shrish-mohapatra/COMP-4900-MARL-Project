import random
import torch
from torchrl.envs.transforms import Transform
from tensordict.tensordict import TensorDictBase


class RandomizeMessageOrder(Transform):
    def __init__(self, randomization_rate: float):
        """
        Args
        - blockage_rate: chance of reversing the order of messages
        """
        super().__init__()
        self.randomization_rate = randomization_rate

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
            # print("\nbefore")
            # print(next_tensordict[("listener", "observation")])
            
            # next_tensordict[("listener", "observation")][:, 0] = next_tensordict[("listener", "observation")][:, 0].flip(1)

            for i in range(len(next_tensordict[("listener", "observation")])):
                if torch.rand(1) < self.randomization_rate:
                    next_tensordict[("listener", "observation")][i][0] = next_tensordict[("listener", "observation")][i][0].flip(0)
            
            # print("\nafter")
            # print(next_tensordict[("listener", "observation")])
            
            return next_tensordict