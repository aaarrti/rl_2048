from flax import linen as nn
import jax
import jax.numpy as jnp


class DuelingDQN(nn.Module):
    n_actions: int = 4

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.relu(nn.Dense(256)(x))
        adv = nn.relu(nn.Dense(128)(x))
        val = nn.relu(nn.Dense(128)(x))

        adv = nn.Dense(self.n_actions)(adv)
        val = nn.Dense(1)(val)

        q = val + (adv - jnp.mean(adv, axis=-1, keepdims=True))
        return q
