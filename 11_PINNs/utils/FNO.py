import jax 
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PyTree
from typing import Callable, List
from jax.tree_util import tree_map

class SpectralConv2d(eqx.Module):
    real_weights: jax.Array
    imag_weights: jax.Array
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int

    def __init__(
            self,
            in_channels,
            out_channels,
            modes1,
            modes2,
            *,
            key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)

        real_key, imag_key = jax.random.split(key)
        self.real_weights = jax.random.uniform(
            real_key,
            (in_channels, out_channels, self.modes1, self.modes2),
            minval=-scale,
            maxval=+scale,
        )
        self.imag_weights = jax.random.uniform(
            imag_key,
            (in_channels, out_channels, self.modes1, self.modes2),
            minval=-scale,
            maxval=+scale,
        )

    def complex_mult2d(
            self,
            x_hat,
            w,
    ):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return jnp.einsum("ixy,ioxy->oxy", x_hat, w)


    def __call__(
            self,
            x
    ):
        channels, spatial_points_0, spatial_points_1 = x.shape

        # shape of x_hat is (in_channels, spatial_points_0, spatial_points_1//2+1)
        x_hat = jnp.fft.rfft2(x)
        # shape of x_hat_under_modes is (in_channels, self.modes1, self.modes2)
        x_hat_under_modes = x_hat[:, :self.modes1, :self.modes2]
        weights = self.real_weights + 1j * self.imag_weights

        # shape of out_hat_under_modes is (out_channels, self.modes)
        out_hat_under_modes = self.complex_mult2d(x_hat_under_modes, weights)

        # shape of out_hat is (out_channels, spatial_points_0, spatial_points_1//2+1)
        out_hat = jnp.zeros(
            (self.out_channels, x_hat.shape[-2], x_hat.shape[-1]),
            dtype=x_hat.dtype
        )
        out_hat = out_hat.at[:, :self.modes1, :self.modes2].set(out_hat_under_modes)

        out = jnp.fft.irfft2(out_hat, s=(x.shape[-2], x.shape[-1]))

        return out

class FNOBlock2d(eqx.Module):
    spectral_conv: SpectralConv2d
    bypass_conv: eqx.nn.Conv1d
    activation: Callable

    def __init__(
            self,
            in_channels,
            out_channels,
            modes1,
            modes2,
            activation=jax.nn.tanh,
            *,
            key,
    ):
        spectral_conv_key, bypass_key = jax.random.split(key)
        self.spectral_conv = SpectralConv2d(
            in_channels,
            out_channels,
            modes1,
            modes2,
            key=spectral_conv_key,
        )
        self.bypass_conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            1,  # Kernel size is one
            key=bypass_key,
        )

        self.activation = activation

    def __call__(
            self,
            x,
    ):
        return self.activation(
            self.spectral_conv(x) + self.bypass_conv(x)
        )


class FNO2d(eqx.Module):
    lifting: eqx.nn.Conv2d
    fno_blocks: List[FNOBlock2d]
    projection1: eqx.nn.Conv2d
    projection2: eqx.nn.Conv2d
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int
    width: int
    n_blocks: int
    activation: Callable
    bias1: jax.Array
    bias2: jax.Array

    def __init__(
            self,
            in_channels,
            out_channels,
            modes1,
            modes2,
            width,
            activation=jax.nn.tanh,
            n_blocks = 4,
            *,
            key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_blocks = n_blocks
        self.activation = activation
        key, lifting_key = jax.random.split(key)
        self.lifting = eqx.nn.Conv2d(
            in_channels,
            width,
            1,
            key=lifting_key,
        )

        self.fno_blocks = []
        for i in range(n_blocks):
            key, subkey = jax.random.split(key)
            if i != n_blocks - 1:
                self.fno_blocks.append(FNOBlock2d(
                    width,
                    width,
                    modes1,
                    modes2,
                    activation,
                    key=subkey,
                ))
            else: 
                self.fno_blocks.append(FNOBlock2d(
                    width,
                    width,
                    modes1,
                    modes2,
                    activation = lambda x: x,
                    key=subkey,
                ))

        key, projection1_key, projection2_key = jax.random.split(key, 3)
        self.projection1 = eqx.nn.Conv2d(
            width,
            width*4,
            1,
            key=projection1_key,
        )
        self.projection2 = eqx.nn.Conv2d(
            width*4,
            out_channels,
            1,
            key=projection2_key,
        )
        key, b1key, b2key = jax.random.split(key, 3)
        self.bias1 = jr.normal(b1key)
        self.bias2 = jr.normal(b2key)

    @eqx.filter_jit
    def __call__(
            self,
            x,
    ):
        assert(x.shape[0] == self.in_channels)

        x = self.lifting(x)

        for fno_block in self.fno_blocks:
            x = fno_block(x)

        x = self.projection1(x) + self.bias1
        x = self.activation(x) 
        x = self.projection2(x) + self.bias2

        #return x if self.out_channels != 1 else x.squeeze(0)
        return x.squeeze()

