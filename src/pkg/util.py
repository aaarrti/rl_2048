from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from pkg.models import ActorCritic


def get_initial_params(key: jax.Array, model: ActorCritic) -> dict[str, Any]:
    input_dims = (1, 4, 4)
    init_shape = jnp.ones(input_dims, jnp.float32)
    initial_params = model.init(key, init_shape)["params"]
    return initial_params  # type: ignore
