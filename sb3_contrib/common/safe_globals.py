"""Safe deserialization registration for SB3-Contrib types.

This module registers all SB3-Contrib types (policies, distributions,
buffers, custom layers, named tuples) with the SB3 safe deserialization
allowlist, so that SB3-Contrib saved models can be loaded with
``deserialization_mode="safe"`` (the default since SB3 2.10).

The registration is performed automatically when sb3_contrib is imported.
"""

from __future__ import annotations

import torch as th
from stable_baselines3.common.safe_globals import add_safe_globals, get_safe_globals


def _register() -> None:
    """Register all sb3-contrib types with the safe deserialization allowlist."""
    from sb3_contrib.ars.policies import ARSLinearPolicy, ARSPolicy
    from sb3_contrib.common.maskable.buffers import (
        MaskableDictRolloutBuffer,
        MaskableRolloutBuffer,
    )
    from sb3_contrib.common.maskable.distributions import (
        MaskableBernoulliDistribution,
        MaskableCategorical,
        MaskableCategoricalDistribution,
        MaskableDistribution,
        MaskableMultiCategoricalDistribution,
    )
    from sb3_contrib.common.maskable.policies import (
        MaskableActorCriticCnnPolicy,
        MaskableActorCriticPolicy,
        MaskableMultiInputActorCriticPolicy,
    )
    from sb3_contrib.common.recurrent.buffers import (
        RecurrentDictRolloutBuffer,
        RecurrentRolloutBuffer,
    )
    from sb3_contrib.common.recurrent.policies import (
        RecurrentActorCriticCnnPolicy,
        RecurrentActorCriticPolicy,
        RecurrentMultiInputActorCriticPolicy,
    )
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
    from sb3_contrib.common.torch_layers import BatchRenorm, BatchRenorm1d
    from sb3_contrib.crossq.policies import (
        Actor as CrossQActor,
    )
    from sb3_contrib.crossq.policies import (
        CrossQCritic,
        CrossQPolicy,
    )
    from sb3_contrib.qrdqn.policies import QRDQNPolicy, QuantileNetwork
    from sb3_contrib.tqc.policies import Actor as TQCActor
    from sb3_contrib.tqc.policies import Critic as TQCCritic
    from sb3_contrib.tqc.policies import TQCPolicy

    contrib_types = [
        # ARS policies
        ARSPolicy,
        ARSLinearPolicy,
        # Maskable distributions
        MaskableCategorical,
        MaskableDistribution,
        MaskableCategoricalDistribution,
        MaskableMultiCategoricalDistribution,
        MaskableBernoulliDistribution,
        # Maskable policies
        MaskableActorCriticPolicy,
        MaskableActorCriticCnnPolicy,
        MaskableMultiInputActorCriticPolicy,
        # Maskable buffers
        MaskableRolloutBuffer,
        MaskableDictRolloutBuffer,
        # Recurrent policies
        RecurrentActorCriticPolicy,
        RecurrentActorCriticCnnPolicy,
        RecurrentMultiInputActorCriticPolicy,
        # Recurrent buffers
        RecurrentRolloutBuffer,
        RecurrentDictRolloutBuffer,
        # Recurrent type aliases (NamedTuples)
        RNNStates,
        # TQC policies
        TQCActor,
        TQCCritic,
        TQCPolicy,
        # CrossQ policies
        CrossQActor,
        CrossQCritic,
        CrossQPolicy,
        # CrossQ custom layers
        BatchRenorm,
        BatchRenorm1d,
        # QR-DQN policies
        QRDQNPolicy,
        QuantileNetwork,
    ]

    add_safe_globals(contrib_types)
    th.serialization.add_safe_globals(contrib_types)  # type: ignore[arg-type]

    # Verify at least one type was registered (sanity check)
    _allowlist = get_safe_globals()
    assert any("sb3_contrib" in entry for entry in _allowlist), "sb3_contrib types were not registered with the safe allowlist"


# Auto-register on import
_register()
