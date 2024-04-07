import torch.nn as nn

class ObsDecoder(nn.Module):
    def __init__(self, encoded_size, original_obs_size):
        super(ObsDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoded_size, 128),
            nn.ReLU(),
            nn.Linear(128, original_obs_size),
            nn.Sigmoid()
        )

    def forward(self, encoded_obs):
        return self.decoder(encoded_obs)
