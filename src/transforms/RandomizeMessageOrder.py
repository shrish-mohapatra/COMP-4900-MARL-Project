import torch
from torchrl.envs.transforms import Transform
from tensordict.tensordict import TensorDictBase


class RandomizeMessageOrder(Transform):
    def __init__(self, device):
        """
        Args
        - blockage_rate: chance of reversing the order of messages
        """
        super().__init__()
        self.device = device

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
            observation = tensordict[("listener", "observation")]
            agent_keys = list(tensordict.keys())
            non_agent_keys = {'collector', 'done', 'terminated'}
            agent_keys = [key for key in agent_keys if key not in non_agent_keys]
            num_agents = len(agent_keys) - 1 # subtract listener
            batch_size, _, seq_len = observation.shape
            dim_c = seq_len//num_agents

            device = observation.device

            shuffled_block_indices = (torch.arange(num_agents, device=device) * dim_c)[torch.randperm(num_agents, device=device)]
            shuffled_indices = (shuffled_block_indices.unsqueeze(1).repeat(1, dim_c) + torch.arange(dim_c, device=device)).view(1, -1).repeat(batch_size, 1)

            observation_flat = observation.view(batch_size, -1)
            shuffled_observation = torch.gather(observation_flat, 1, shuffled_indices)

            # Reshape back to the original shape
            tensordict[("listener", "observation")] = shuffled_observation.view_as(observation)

            return next_tensordict