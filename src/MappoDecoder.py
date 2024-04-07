from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Tuple, Type

from benchmarl.algorithms.mappo import Mappo
from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from src.ClipPPODecoderLoss import ClipPPODecoderLoss


class MappoDecoder(Mappo):
    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        # Loss
        loss_module = ClipPPODecoderLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss_module, False


@dataclass
class MappoDecoderConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Mappo`."""

    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return MappoDecoder

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True
