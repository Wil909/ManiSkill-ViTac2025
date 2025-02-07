from typing import List, Tuple, Type

import gymnasium as gym
import torch
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.td3.policies import Actor
from torch import nn

from solutions.networks import PointNetFeatureExtractor
from typing import Optional, Dict, Tuple, Union, List, Type

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetActor(Actor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        features_extractor: nn.Module,
        pointnet_in_dim: int,
        pointnet_out_dim: int,
        normalize_images: bool = True,
        batchnorm=False,
        layernorm=True,
        use_relative_motion=True,
        use_state=True,
        zero_init_output=False,
        state_mlp_size=[64, 64], 
        state_mlp_activation_fn=nn.ReLU,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            **kwargs,
        )
        self.use_relative_motion = use_relative_motion
        self.use_state = use_state
        action_dim = get_action_dim(self.action_space)

        self.point_net_feature_extractor = PointNetFeatureExtractor(
            dim=pointnet_in_dim, out_dim=pointnet_out_dim, batchnorm=batchnorm
        )

        mlp_in_channels = 2 * pointnet_out_dim
                
        if self.use_state:
            net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]
            state_dim = 7
            self.state_mlp = nn.Sequential(*create_mlp(state_dim, output_dim, net_arch, state_mlp_activation_fn))
            mlp_in_channels += output_dim
            
        if self.use_relative_motion:
            mlp_in_channels += 4

        self.mlp_policy = nn.Sequential(
            nn.Linear(mlp_in_channels, 256),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        if zero_init_output:
            last_linear = None
            for m in self.mlp_policy.children():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                nn.init.zeros_(last_linear.bias)
                last_linear.weight.data.copy_(0.01 * last_linear.weight.data)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(False):
            marker_pos = self.extract_features(obs, self.features_extractor)

        if marker_pos.ndim == 3:
            marker_pos = torch.unsqueeze(marker_pos, dim=0)

        batch_num = marker_pos.shape[0]

        l_marker_pos = marker_pos[:, 0, ...]
        r_marker_pos = marker_pos[:, 1, ...]

        marker_pos_input = torch.cat([l_marker_pos, r_marker_pos], dim=0)

        point_flow_fea = self.point_net_feature_extractor(marker_pos_input)

        l_point_flow_fea = point_flow_fea[:batch_num, ...]
        r_point_flow_fea = point_flow_fea[batch_num:, ...]

        point_flow_fea = torch.cat([l_point_flow_fea, r_point_flow_fea], dim=-1)
        feature = [
            point_flow_fea,
        ]
        
        if self.use_state:
            gt_offset = obs["gt_offset"]  # 4
            relative_motion = obs["relative_motion"]  # 4
            state = torch.cat([gt_offset, relative_motion], dim=-1)
            state_feat = self.state_mlp(state)
            if len(state_feat.shape) == 1:
                state_feat = state_feat.unsqueeze(0) 
            feature.append(state_feat)

        if self.use_relative_motion:
            relative_motion = obs["relative_motion"]
            if relative_motion.ndim == 1:
                relative_motion = torch.unsqueeze(relative_motion, dim=0)
            feature.append(relative_motion)

        feature = torch.cat(feature, dim=-1)
        pred = self.mlp_policy(feature)
        return pred


class LongOpenLockPointNetActor(Actor):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        pointnet_in_dim: int,
        pointnet_out_dim: int,
        normalize_images: bool = True,
        batchnorm=False,
        layernorm=False,
        use_relative_motion=True,
        use_state=True,
        zero_init_output=False,
        state_mlp_size=[64, 64], 
        state_mlp_activation_fn=nn.ReLU,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            **kwargs,
        )
        self.use_relative_motion = use_relative_motion
        self.use_state = use_state
        action_dim = get_action_dim(self.action_space)
        self.point_net_feature_extractor = PointNetFeatureExtractor(
            dim=pointnet_in_dim, out_dim=pointnet_out_dim, batchnorm=batchnorm
        )

        mlp_in_channels = 2 * pointnet_out_dim
        
        if self.use_state:
            net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]
            state_dim = 3
            self.state_mlp = nn.Sequential(*create_mlp(state_dim, output_dim, net_arch, state_mlp_activation_fn))
            mlp_in_channels += output_dim
            
        if self.use_relative_motion:
            mlp_in_channels += 3

        self.mlp_policy = nn.Sequential(
            nn.Linear(mlp_in_channels, 256),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        if zero_init_output:
            last_linear = None
            for m in self.mlp_policy.children():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                nn.init.zeros_(last_linear.bias)
                last_linear.weight.data.copy_(0.01 * last_linear.weight.data)

    def forward(self, obs: dict) -> torch.Tensor:
        # (batch_num, 2 (left_and_right), 128 (marker_num), 4 (u0, v0; u1, v1))

        marker_pos = obs["marker_flow"]
        if marker_pos.ndim == 4:
            marker_pos = torch.unsqueeze(marker_pos, dim=0)

        l_marker_pos = torch.cat(
            [marker_pos[:, 0, 0, ...], marker_pos[:, 0, 1, ...]], dim=-1
        )
        r_marker_pos = torch.cat(
            [marker_pos[:, 1, 0, ...], marker_pos[:, 1, 1, ...]], dim=-1
        )

        l_point_flow_fea = self.point_net_feature_extractor(l_marker_pos)
        r_point_flow_fea = self.point_net_feature_extractor(
            r_marker_pos
        )  # (batch_num, pointnet_feature_dim)
        point_flow_fea = torch.cat([l_point_flow_fea, r_point_flow_fea], dim=-1)

        feature = [
            point_flow_fea,
        ]
        
        if self.use_state:
            relative_motion = obs["relative_motion"]  # 4
            state = torch.cat([relative_motion], dim=-1)
            state_feat = self.state_mlp(state)
            if len(state_feat.shape) == 1:
                state_feat = state_feat.unsqueeze(0) 
            feature.append(state_feat)

        if self.use_relative_motion:
            relative_motion = obs["relative_motion"]
            if relative_motion.ndim == 1:
                relative_motion = torch.unsqueeze(relative_motion, dim=0)
            # repeat_num = l_point_flow_fea.shape[-1] // 4
            # xz = xz.repeat(1, repeat_num)
            feature.append(relative_motion)

        feature = torch.cat(feature, dim=-1)
        pred = self.mlp_policy(feature)
        return pred


class CustomCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            **kwargs,
        )

        action_dim = get_action_dim(self.action_space)
        self.features_dim = features_dim

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = nn.Sequential(
                *create_mlp(self.features_dim + action_dim, 1, net_arch, activation_fn)
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(False):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](torch.cat([features, actions], dim=1))
