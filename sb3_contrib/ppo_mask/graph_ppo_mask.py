"""
Graph Maskable PPO: Proximal Policy Optimization for graph-structured
observations with Invalid Action Masking.

Combines the graph-observation handling of :class:`~sb3_contrib.ppo_mask.graph_ppo.GraphPPO`
(``gymnasium.spaces.Graph`` observations converted to
``torch_geometric.data.Batch`` and processed by a GNN features extractor)
with the invalid-action-masking machinery of
:class:`~sb3_contrib.ppo_mask.ppo_mask.MaskablePPO`.

The environment's observation space must be a ``gymnasium.spaces.Graph`` and it
must expose an ``action_masks()`` method (see ``ActionMasker`` wrapper).

Requirements:
    pip install torch_geometric
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any, ClassVar, NamedTuple, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import (
    GNNFeaturesExtractor,
    GraphRolloutBuffer,
    graphs_to_batch,
)
try:
    from gymnasium.spaces.utils import GraphInstance
except ImportError:
    from gymnasium.spaces.graph import GraphInstance  # type: ignore[no-redef]



try:
    from torch_geometric.data import Batch
except ImportError as exc:
    raise ImportError(
        "torch_geometric is required for GraphMaskablePPO. "
        "See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    ) from exc

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO

SelfGraphMaskablePPO = TypeVar("SelfGraphMaskablePPO", bound="GraphMaskablePPO")


# ---------------------------------------------------------------------------
# Graph Maskable Rollout Buffer
# ---------------------------------------------------------------------------


class GraphMaskableRolloutBufferSamples(NamedTuple):
    observations: Batch
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor


class GraphMaskableRolloutBuffer(GraphRolloutBuffer):
    """
    Rollout buffer for ``gymnasium.spaces.Graph`` observations that also stores
    the invalid action masks associated with each observation.

    Extends :class:`~sb3_contrib.ppo_mask.graph_ppo.GraphRolloutBuffer` with the
    masking storage logic of
    :class:`~sb3_contrib.common.maskable.buffers.MaskableRolloutBuffer`.
    """

    action_masks: np.ndarray
    mask_dims: int

    def reset(self) -> None:
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = int(self.action_space.n)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        super().reset()
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)

    def add(self, *args, action_masks: np.ndarray | None = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))
        super().add(*args, **kwargs)

    def get(self, batch_size: int | None = None) -> Generator[GraphMaskableRolloutBufferSamples, None, None]:  # type: ignore[override]
        assert self.full, "Buffer must be completely filled before sampling"

        total = self.buffer_size * self.n_envs
        indices = np.random.permutation(total)

        # Flatten scalar arrays to env-major order (matches swap_and_flatten).
        flat_actions = self.swap_and_flatten(self.actions)
        flat_values = self.swap_and_flatten(self.values)
        flat_log_probs = self.swap_and_flatten(self.log_probs)
        flat_advantages = self.swap_and_flatten(self.advantages)
        flat_returns = self.swap_and_flatten(self.returns)
        flat_action_masks = self.swap_and_flatten(self.action_masks)

        # Flatten obs to match the same env-major ordering.
        flat_obs: list = [None] * total  # type: ignore[list-item]
        for env_idx in range(self.n_envs):
            for step in range(self.buffer_size):
                flat_obs[env_idx * self.buffer_size + step] = self.obs_store[step][env_idx]

        if batch_size is None:
            batch_size = total

        start_idx = 0
        while start_idx < total:
            batch_inds = indices[start_idx : start_idx + batch_size]
            obs_batch = graphs_to_batch([flat_obs[i] for i in batch_inds], self.device)
            yield GraphMaskableRolloutBufferSamples(
                observations=obs_batch,
                actions=th.as_tensor(flat_actions[batch_inds].astype(np.float32, copy=False), device=self.device),
                old_values=th.as_tensor(flat_values[batch_inds].flatten(), device=self.device),
                old_log_prob=th.as_tensor(flat_log_probs[batch_inds].flatten(), device=self.device),
                advantages=th.as_tensor(flat_advantages[batch_inds].flatten(), device=self.device),
                returns=th.as_tensor(flat_returns[batch_inds].flatten(), device=self.device),
                action_masks=th.as_tensor(
                    flat_action_masks[batch_inds].reshape(-1, self.mask_dims), device=self.device
                ),
            )
            start_idx += batch_size


# ---------------------------------------------------------------------------
# Graph Maskable Actor-Critic Policy
# ---------------------------------------------------------------------------


class GraphMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """
    Maskable Actor-Critic policy for ``gymnasium.spaces.Graph`` observations.

    Combines:

    - The graph feature extraction of
      :class:`~sb3_contrib.ppo_mask.graph_ppo.GraphActorCriticPolicy`
      (``GraphInstance`` → ``torch_geometric.data.Batch`` → :class:`GNNFeaturesExtractor`,
      bypassing ``preprocess_obs`` which does not support ``spaces.Graph``).
    - The invalid-action-masking action distribution of
      :class:`~sb3_contrib.common.maskable.policies.MaskableActorCriticPolicy`.

    :param observation_space: Must be a ``gymnasium.spaces.Graph``.
    :param action_space: Action space (Discrete, MultiDiscrete or MultiBinary).
    :param lr_schedule: Learning rate schedule.
    :param gnn_features_dim: Output dimension of the GNN extractor.
    :param gnn_hidden_dim: Hidden dimension of GNN layers.
    :param num_gnn_layers: Number of GNN message-passing layers.
    :param kwargs: Forwarded to :class:`MaskableActorCriticPolicy`.
    """

    def __init__(
        self,
        observation_space: spaces.Graph,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        gnn_features_dim: int = 64,
        gnn_hidden_dim: int = 64,
        num_gnn_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        assert isinstance(observation_space, spaces.Graph), (
            "GraphMaskableActorCriticPolicy requires a gymnasium.spaces.Graph observation space"
        )
        # Inject GNNFeaturesExtractor; the user should not override this.
        kwargs["features_extractor_class"] = GNNFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {
            "features_dim": gnn_features_dim,
            "gnn_hidden_dim": gnn_hidden_dim,
            "num_gnn_layers": num_gnn_layers,
        }
        # Images are not involved; disable normalisation.
        kwargs.setdefault("normalize_images", False)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # ------------------------------------------------------------------
    # Feature extraction — bypass preprocess_obs
    # ------------------------------------------------------------------

    def _gnn_extract(self, obs: Batch, features_extractor: GNNFeaturesExtractor) -> th.Tensor:
        """Pass a PyG Batch directly to a features extractor (no preprocessing)."""
        return features_extractor(obs)

    def extract_features(  # type: ignore[override]
        self,
        obs: Batch,
        features_extractor: GNNFeaturesExtractor | None = None,
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        if self.share_features_extractor:
            ext = self.features_extractor if features_extractor is None else features_extractor
            return self._gnn_extract(obs, ext)
        else:
            return (
                self._gnn_extract(obs, self.pi_features_extractor),
                self._gnn_extract(obs, self.vf_features_extractor),
            )

    def predict_values(self, obs: Batch) -> th.Tensor:  # type: ignore[override]
        features = self._gnn_extract(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def get_distribution(self, obs: Batch, action_masks: np.ndarray | None = None):  # type: ignore[override]
        features = self._gnn_extract(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    # ------------------------------------------------------------------
    # Observation → Batch conversion
    # ------------------------------------------------------------------

    def obs_to_tensor(  # type: ignore[override]
        self,
        observation: GraphInstance | np.ndarray | list,
    ) -> tuple[Batch, bool]:
        """
        Convert a single ``GraphInstance`` or an array/list of them to a
        ``torch_geometric.data.Batch``.

        :param observation: One ``GraphInstance`` (non-vectorised) or a
            length-``n_envs`` array/list of them (vectorised).
        :return: (batch, vectorized_env)
        """
        if isinstance(observation, GraphInstance):
            instances = [observation]
            vectorized = False
        else:
            instances = list(observation)
            vectorized = len(instances) > 1 or (
                isinstance(observation, np.ndarray) and observation.ndim >= 1
            )
        batch = graphs_to_batch(instances, self.device)
        return batch, vectorized


# ---------------------------------------------------------------------------
# Graph Maskable PPO
# ---------------------------------------------------------------------------


class GraphMaskablePPO(MaskablePPO):
    """
    Proximal Policy Optimization with Invalid Action Masking for environments
    with ``gymnasium.spaces.Graph`` observations.

    Each observation is a ``GraphInstance`` (a namedtuple with ``nodes``,
    ``edges`` and ``edge_links`` fields) that is converted on-the-fly to a
    ``torch_geometric.data.Batch`` and processed by a :class:`GNNFeaturesExtractor`.
    Action masking is applied exactly as in
    :class:`~sb3_contrib.ppo_mask.ppo_mask.MaskablePPO`.

    All standard (Maskable)PPO hyper-parameters apply.  GNN architecture
    parameters are forwarded to :class:`GraphMaskableActorCriticPolicy` via
    ``policy_kwargs``.

    :param policy: Policy class or ``"GraphMaskablePolicy"`` string alias.
    :param env: Training environment whose observation space is
        ``gymnasium.spaces.Graph`` and which exposes an ``action_masks()`` method.
    :param gnn_features_dim: Output dimension of the GNN features extractor.
    :param gnn_hidden_dim: Hidden channels for each GNN layer.
    :param num_gnn_layers: Number of GNN message-passing layers.
    :param kwargs: Forwarded to :class:`~sb3_contrib.ppo_mask.ppo_mask.MaskablePPO`.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "GraphMaskablePolicy": GraphMaskableActorCriticPolicy,
    }

    def __init__(
        self,
        policy: str | type[GraphMaskableActorCriticPolicy] = "GraphMaskablePolicy",
        env: GymEnv | str | None = None,
        gnn_features_dim: int = 64,
        gnn_hidden_dim: int = 64,
        num_gnn_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        # Inject GNN params into policy_kwargs (do not overwrite user values).
        policy_kwargs: dict[str, Any] = kwargs.pop("policy_kwargs", None) or {}
        policy_kwargs.setdefault("gnn_features_dim", gnn_features_dim)
        policy_kwargs.setdefault("gnn_hidden_dim", gnn_hidden_dim)
        policy_kwargs.setdefault("num_gnn_layers", num_gnn_layers)

        # Always use the graph maskable rollout buffer.
        kwargs["rollout_buffer_class"] = GraphMaskableRolloutBuffer

        super().__init__(policy=policy, env=env, policy_kwargs=policy_kwargs, **kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Rollout collection — graph obs + action masking
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: GraphMaskableRolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Collect ``n_rollout_steps`` transitions per environment and store them
        in the :class:`GraphMaskableRolloutBuffer`.

        Graph observations are converted to ``torch_geometric.data.Batch``
        objects before being passed to the policy, and invalid action masks are
        queried from the environment and applied to the action distribution.
        """
        assert isinstance(rollout_buffer, GraphMaskableRolloutBuffer), (
            "GraphMaskablePPO requires a GraphMaskableRolloutBuffer"
        )
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_batch = graphs_to_batch(self._last_obs, self.device)

                if use_masking:
                    action_masks = get_action_masks(env)

                actions, values, log_probs = self.policy(obs_batch, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Bootstrap value for timed-out episodes.
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = infos[idx]["terminal_observation"]
                    if isinstance(terminal_obs, GraphInstance):
                        terminal_obs = [terminal_obs]
                    terminal_batch = graphs_to_batch(terminal_obs, self.device)
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_batch)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            last_batch = graphs_to_batch(new_obs, self.device)
            values = self.policy.predict_values(last_batch)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()
        return True

    def learn(  # type: ignore[override]
        self: SelfGraphMaskablePPO,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        tb_log_name: str = "GraphMaskablePPO",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ) -> SelfGraphMaskablePPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            use_masking=use_masking,
            progress_bar=progress_bar,
        )
