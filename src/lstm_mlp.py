from typing import Dict, Optional, Sequence, Tuple, Type, Union
import warnings
from torch import nn

# Assuming these imports are correct based on your project structure
from benchmarl.models.mlp import MLP
import torch
from torchrl.modules.models import MultiAgentMLP
from torchrl.data.utils import DEVICE_TYPING


# plz smd
class LSTMNet(nn.Module):
    """Josh was here An embedder for an LSTM preceded by an MLP.

    The forward method returns the hidden states of the current state (input hidden states) and the output, as
    the environment returns the 'observation' and 'next_observation'.

    Because the LSTM kernel only returns the last hidden state, hidden states
    are padded with zeros such that they have the right size to be stored in a
    TensorDict of size [batch x time_steps].

    If a 2D tensor is provided as input, it is assumed that it is a batch of data
    with only one time step. This means that we explicitely assume that users will
    unsqueeze inputs of a single batch with multiple time steps.

    Examples:
        >>> batch = 7
        >>> time_steps = 6
        >>> in_features = 4
        >>> out_features = 10
        >>> hidden_size = 5
        >>> net = LSTMNet(
        ...     out_features,
        ...     {"input_size": hidden_size, "hidden_size": hidden_size},
        ...     {"out_features": hidden_size},
        ... )
        >>> # test single step vs multi-step
        >>> x = torch.randn(batch, time_steps, in_features)  # >3 dims = multi-step
        >>> y, hidden0_in, hidden1_in, hidden0_out, hidden1_out = net(x)
        >>> x = torch.randn(batch, in_features)  # 2 dims = single step
        >>> y, hidden0_in, hidden1_in, hidden0_out, hidden1_out = net(x)

    """

    def __init__(
        self,
        out_features: int,
        lstm_kwargs: Dict,
        mlp_kwargs: Dict,
        device: Optional[DEVICE_TYPING] = None,
    ) -> None:
        print("DEBUG: using custom LSTMNet weeeoeoeoeoe")
        warnings.warn(
            "LSTMNet is being deprecated in favour of torchrl.modules.LSTMModule, and will be removed in v0.4.0.",
            category=DeprecationWarning,
        )
        super().__init__()
        lstm_kwargs.update({"batch_first": True})
        self.mlp = MLP(device=device, **mlp_kwargs)
        self.lstm = nn.LSTM(device=device, **lstm_kwargs)
        self.linear = nn.Linear(out_features, out_features, device=device)

    def _lstm(
        self,
        input: torch.Tensor,
        hidden0_in: Optional[torch.Tensor] = None,
        hidden1_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        squeeze0 = False
        squeeze1 = False
        if input.ndimension() == 1:
            squeeze0 = True
            input = input.unsqueeze(0).contiguous()

        if input.ndimension() == 2:
            squeeze1 = True
            input = input.unsqueeze(1).contiguous()
        batch, steps = input.shape[:2]

        if hidden1_in is None and hidden0_in is None:
            shape = (batch, steps) if not squeeze1 else (batch,)
            hidden0_in, hidden1_in = [
                torch.zeros(
                    *shape,
                    self.lstm.num_layers,
                    self.lstm.hidden_size,
                    device=input.device,
                    dtype=input.dtype,
                )
                for _ in range(2)
            ]
        elif hidden1_in is None or hidden0_in is None:
            raise RuntimeError(
                f"got type(hidden0)={type(hidden0_in)} and type(hidden1)={type(hidden1_in)}"
            )
        elif squeeze0:
            hidden0_in = hidden0_in.unsqueeze(0)
            hidden1_in = hidden1_in.unsqueeze(0)

        # we only need the first hidden state
        if not squeeze1:
            _hidden0_in = hidden0_in[:, 0]
            _hidden1_in = hidden1_in[:, 0]
        else:
            _hidden0_in = hidden0_in
            _hidden1_in = hidden1_in
        hidden = (
            _hidden0_in.transpose(-3, -2).contiguous(),
            _hidden1_in.transpose(-3, -2).contiguous(),
        )

        # print(f'\nDEBUG in _lstm input={input}\nhidden={hidden}')
        y0, hidden = self.lstm(input, hidden)
        # dim 0 in hidden is num_layers, but that will conflict with tensordict
        hidden = tuple(_h.transpose(0, 1) for _h in hidden)
        y = self.linear(y0)

        out = [y, hidden0_in, hidden1_in, *hidden]
        if squeeze1:
            # squeezes time
            out[0] = out[0].squeeze(1)
        if not squeeze1:
            # we pad the hidden states with zero to make tensordict happy
            for i in range(3, 5):
                out[i] = torch.stack(
                    [torch.zeros_like(out[i]) for _ in range(input.shape[1] - 1)]
                    + [out[i]],
                    1,
                )
        if squeeze0:
            out = [_out.squeeze(0) for _out in out]
        # return tuple(out)
        return out

    def forward(
        self,
        input: torch.Tensor,
        hidden0_in: Optional[torch.Tensor] = None,
        hidden1_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # print(f"DEBUG input pre-mlp={input}")
        input = self.mlp(input)
        # print(f"DEBUG input pre-lsmt={input}")
        output = self._lstm(input, hidden0_in, hidden1_in)
        # print(f"DEBUG output post-lsmt={output}")
        return output[0]


class MultiAgentLSTM(MultiAgentMLP):
    def __init__(
        self,
        n_agent_inputs: int,
        n_agent_outputs: int,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: Optional[DEVICE_TYPING] = None,
        depth: Optional[int] = None,
        num_cells: Optional[Union[Sequence, int]] = None,
        activation_class: Optional[Type[nn.Module]] = nn.Tanh,
        **kwargs,
    ):
        super().__init__(
            n_agent_inputs,
            n_agent_outputs,
            n_agents,
            centralised,
            share_params,
            device,
            depth,
            num_cells,
            activation_class,
            **kwargs
        )

        in_features = n_agent_inputs if not centralised else n_agent_inputs * n_agents

        self.agent_networks = nn.ModuleList(
            [
                LSTMNet(
                    out_features=n_agent_outputs,
                    lstm_kwargs={
                        "input_size": n_agent_outputs,  # Corrected to match input features
                        "hidden_size": n_agent_outputs,
                    },
                    mlp_kwargs={
                        "in_features": in_features,  # Ensuring consistency
                        "depth": depth,
                        "num_cells": num_cells,
                        "activation_class": activation_class,
                        "out_features": n_agent_outputs,
                    },
                    device=device,
                )
                for _ in range(n_agents if not share_params else 1)
            ]
        )

        print("DEBUG: Successfully created agent networks with LSTMNet")