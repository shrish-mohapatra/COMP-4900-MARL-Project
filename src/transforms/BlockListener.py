import random
import torch
from torchrl.envs.transforms import Transform
from tensordict.tensordict import TensorDictBase


class BlockListener(Transform):
    def __init__(self, blockage_rate: float):
        """
        Args
        - blockage_rate: chance of message being blocked between 0 and 1
        """
        super().__init__()
        self.blockage_rate = blockage_rate

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
            # next_tensordict[("listener", "observation")][..., random.randrange(self.blockage_rate), :] = 0
            
            batch_size, _, msg_size = next_tensordict[("listener", "observation")].shape
            msg_size //= 2
            
            # Generate binary decisions based on probabilities for each speaker
            decisions_speaker1 = torch.rand(batch_size) > self.blockage_rate
            decisions_speaker2 = torch.rand(batch_size) > self.blockage_rate

            # print(f"\ndecisions_speaker1\n{decisions_speaker1}")
            # print(f"\ndecisions_speaker2\n{decisions_speaker2}")

            # Expand decisions to full message size and set values to 1 or 0
            msgs_speaker1 = decisions_speaker1.unsqueeze(1).unsqueeze(2).expand(-1, 1, msg_size).float()
            msgs_speaker2 = decisions_speaker2.unsqueeze(1).unsqueeze(2).expand(-1, 1, msg_size).float()

            # print(f"\nmsgs_speaker1\n{msgs_speaker1}")
            # print(f"\nmsgs_speaker1\n{msgs_speaker2}")

            # Concatenate messages from both speakers
            listener_observations = torch.cat((msgs_speaker1, msgs_speaker2), dim=2)
            next_tensordict[("listener", "observation")] *= listener_observations

            # print("\ntensordict")
            # print(next_tensordict[("listener", "observation")])
            return next_tensordict