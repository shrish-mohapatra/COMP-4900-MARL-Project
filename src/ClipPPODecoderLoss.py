import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.objectives import ClipPPOLoss
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import dispatch, ProbabilisticTensorDictSequential, TensorDictModule

from src.ObsDecoder import ObsDecoder

import logging


class ClipPPODecoderLoss(ClipPPOLoss):
    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential = None,
        critic_network: TensorDictModule = None,
        *,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = True,
        gamma: float = None,
        separate_losses: bool = False,
        **kwargs,
    ):
        super(ClipPPOLoss, self).__init__(
            actor_network,
            critic_network,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            **kwargs,
        )
        actor_mlp = kwargs["actor"].module[0].module[0]
        self.agent_group = actor_mlp.agent_group
        self.device = actor_mlp.device

        # Create decoder network for speakers only
        self.decoder_nn = None
        if self.agent_group != "listener":
            self.decoder_nn = ObsDecoder(
                encoded_size=8,
                original_obs_size=actor_mlp.input_features,
            ).to(device=self.device)
            self.decoder_loss = nn.MSELoss()
            self.decoder_optimizer = optim.Adam(
                self.decoder_nn.parameters(),
                lr=1e-3,
            )

        self.debug_calls = 0

        self.register_buffer("clip_epsilon", torch.tensor(clip_epsilon))

        logger_name = f"decoder-logger-{self.agent_group}"
        file_handler = logging.FileHandler(f"{logger_name}.log")
        file_handler.setLevel(logging.DEBUG)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"DEBUG creating ClipPPOLoss module")

    def _get_decoder_loss(self, tensordict: TensorDictBase):
        original_msg = tensordict[self.agent_group]['observation']
        encoded_msg = tensordict[self.agent_group]['action']

        original_msg = original_msg.flatten(start_dim=1)
        encoded_msg = encoded_msg.flatten(start_dim=1)

        original_min, original_max = original_msg.min(), original_msg.max()
        encoded_min, encoded_max = encoded_msg.min(), encoded_msg.max()

        original_msg = (original_msg - original_min) / \
            (original_max - original_min + 1e-5)
        encoded_msg = (encoded_msg - encoded_min) / \
            (encoded_max - encoded_min + 1e-5)

        self.decoder_optimizer.zero_grad()
        decode = self.decoder_nn(encoded_msg)

        # print('decode.shape:', decode.shape)
        # print('original_msg.shape:', original_msg.shape)

        loss = self.decoder_loss(decode, original_msg)
        self.logger.debug(f'decoder loss={loss}')
        loss.backward()
        self.decoder_optimizer.step()

        return loss.item()

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # self.debug_calls += 1
        # if self.debug_calls == 1:
        #     if "listener" in tensordict.keys():
        #         observation = tensordict[("listener", "observation")]
        #         print(f"DEBUG listener obs={observation}")
        #     elif "civilian" in tensordict.keys():
        #         civilian_action = tensordict[("civilian", "action")]
        #         print(f"DEBUG civilian actions={civilian_action}")
        #     else:
        #         hq_action = tensordict[("policeHQ", "action")]
        #         print(f"DEBUG hq actions={hq_action}")

        tensordict = tensordict.clone(False)
        # print(f'DEBUG tensordict={tensordict}')
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-6)
            advantage = (advantage - loc) / scale

        log_weight, dist = self._log_weight(tensordict)
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same source. Here we sample according
            # to different, unrelated trajectories, which is not standard. Still it can give a idea of the dispersion
            # of the weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        gain1 = log_weight.exp() * advantage

        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        gain2 = log_weight_clip.exp() * advantage

        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        td_out = TensorDict({"loss_objective": -gain.mean()}, [])

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict)

            if self.decoder_nn:
                loss_decoder = self._get_decoder_loss(tensordict)
                loss_critic += loss_decoder

            td_out.set("loss_critic", loss_critic.mean())

        td_out.set("ESS", ess.mean() / batch)
        # print(f"DEBUG clippodecoder loss_critic={loss_critic}")
        return td_out
