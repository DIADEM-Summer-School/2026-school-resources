'''
Author: Chih-Kang Huang && chih-kang.huang@hotmail.com
Date: 2026-08-30 17:19:32
LastEditors: Chih-Kang Huang && chih-kang.huang@hotmail.com
LastEditTime: 2026-09-01 17:33:27
FilePath: /hands-on/solutions/utils/helper.py
Description: 


'''
import jax 
import jax.numpy as jnp
import numpy as np
from jax import jit
import equinox as eqx
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable, Dict, Any, NamedTuple, Optional

@eqx.filter_jit
def energy(u_sol, N_x, N_y, dx, eps): 
    u_pred = u_sol
    u_h = jnp.fft.rfft2(u_pred)
    wave = jnp.fft.fftfreq(N_x, d= dx) * 2j* jnp.pi 
    wave_real = jnp.fft.rfftfreq(N_y, d= dx)* 2j* jnp.pi
    k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')
    u_x_h =  k_x * u_h
    u_y_h =  k_y * u_h

    u_x = jnp.fft.irfft2(u_x_h, s=(N_x, N_y))
    u_y = jnp.fft.irfft2(u_y_h, s=(N_x, N_y))
    # Constant from the double potential
    return (jnp.mean(eps* (u_x**2 + u_y**2)/2 + ((u_pred**2 - 1)**2)/(4*eps)))/(4/(3*jnp.sqrt(2)))

@eqx.filter_jit 
def splitting_next(u, N_x, N_y, dx, eps, dt, alpha=2.001): 
    """
    A x u_{n+1, h} = B(u_{n, h}) 
    """

    B = jnp.fft.rfft2( 
        u  + dt* ( u - u**3 + alpha*u)
    )

    I = jnp.ones_like(B)

    wave = jnp.fft.fftfreq(N_x, d=dx) * 2j* jnp.pi
    wave_real = jnp.fft.rfftfreq(N_y, d=dx)* 2j* jnp.pi
    k_x, k_y = jnp.meshgrid(wave, wave_real, indexing='ij')

    A = I - dt* ( (eps**2)*(k_x**2 + k_y**2) - alpha**I) 
    
    return  jnp.fft.irfft2(B*(A**(-1)), s=(N_x, N_y))


# ==============================================================================
# Physics & Domain Hyperparameters
# ==============================================================================
class Config(NamedTuple):
    x_lb: float = 0.0
    x_rb: float = 1.0
    y_lb: float = 0.0
    y_ub: float = 1.0
    eps: float = 2.5e-2  # Interface width parameter
    T0: float = 0.0
    Tf: float = 16.0  # Reduced Tf for stable baseline execution
    N_t: int = 40  # Number of temporal slices
    N_x: int = 160  # Spatial resolution x
    N_y: int = 160  # Spatial resolution y
    n_PDE: int = 250  # Spatial points per time slice
    n_IC: int = 300  # Initial condition collocation points
    n_BC: int = 300  # Boundary condition collocation points
    lr: float = 1e-3
    n_epochs: int = 1000
    lambda_IC: float = 100.0
    lambda_BC: float = 1.0
    eps_causal: float = 100.0
    r3_freq: int = 50  # Epoch interval for R3 resampling

# ==============================================================================
# Neural Network Architecture
# ==============================================================================
class PINN(eqx.Module):
    """
    Physics-Informed Neural Network architecture built using Equinox MLP.
    
    Attributes:
        mlp (eqx.nn.MLP): Multi-layer perceptron mapping (t, x, y) -> u(t, x, y).
    """
    mlp: eqx.nn.MLP

    def __init__(self, in_size: int = 3, out_size: str = "scalar", width_size: int = 100, depth: int = 4, *, key: jax.Array):
        """
        Args:
            in_size (int): Input dimensionality (3 for t, x, y).
            out_size (str): Output dimension ('scalar' for single scalar field).
            width_size (int): Number of neurons per hidden layer.
            depth (int): Number of hidden layers.
            key (jax.Array): PRNG key for weight initialization.
        """
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            key=key
        )

    def __call__(self, point: jax.Array) -> jax.Array:
        """
        Forward pass for a single spatio-temporal coordinate.
        
        Args:
            point (jax.Array): Coordinates tensor of shape (3,) representing [t, x, y].
            
        Returns:
            jax.Array: Scalar prediction u(t, x, y) of shape ().
        """
        return self.mlp(point)

# ==============================================================================
# Visualization
# ==============================================================================
def plot_sampling_points(
    points_init: jax.Array, points_final: jax.Array, title: str = "Collocation Points Adaptation"
) -> None:
    """
    Plots spatial distribution comparison between initial and R3-adapted sampling points.
    
    Args:
        points_init (jax.Array): Initial points coordinates array of shape (..., 3).
        points_final (jax.Array): Adapted points coordinates array of shape (..., 3).
        title (str): Figure title header.
    """
    pts0 = points_init.reshape(-1, 3)
    pts1 = points_final.reshape(-1, 3)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(pts0[:, 1], pts0[:, 2], s=6, c="crimson", alpha=0.7, label="Initial Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Initial Distribution: {title}")
    plt.grid(True, linestyle=":")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(pts1[:, 1], pts1[:, 2], s=6, c="navy", alpha=0.7, label="R3 Adapted Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"R3 Adapted: {title}")
    plt.grid(True, linestyle=":")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_loss_history(loss_history: List[float]) -> None:
    """
    Plots logarithmic convergence curve for PINN training history.
    
    Args:
        loss_history (List[float]): Sequence of scalar loss values per epoch.
    """
    plt.figure(figsize=(8, 4), dpi=120)
    plt.plot(jnp.arange(1, len(loss_history) + 1), loss_history, label="Causal Loss", color="teal")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("PINN Optimization Loss Convergence")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_solution_slices(
    model: PINN, config: Config, num_slices: int = 9
) -> None:
    """
    Renders 2D spatio-temporal solution grid evaluations across time snapshots.
    
    Args:
        model (PINN): Trained model parameters.
        config (Config): Configuration containing domain dimensions.
        num_slices (int): Number of time snapshots to plot.
    """
    x = jnp.linspace(config.x_lb, config.x_rb, config.N_x)
    y = jnp.linspace(config.y_lb, config.y_ub, config.N_y)
    xs, ys = jnp.meshgrid(x, y)
    
    t_vals = jnp.linspace(config.T0, config.Tf, num_slices)
    rows = int(jnp.ceil(jnp.sqrt(num_slices)))
    cols = int(jnp.ceil(num_slices / rows))

    plt.figure(figsize=(4 * cols, 3.5 * rows), dpi=120)
    for idx, t_i in enumerate(t_vals):
        ts = jnp.full_like(xs, t_i)
        grid_pts = jnp.stack([ts, xs, ys], axis=-1)  # Shape: (N_y, N_x, 3)
        
        # Vectorized evaluation over spatial grid
        u_pred = jax.vmap(jax.vmap(model))(grid_pts)
        
        plt.subplot(rows, cols, idx + 1)
        im = plt.imshow(
            u_pred,
            extent=[config.x_lb, config.x_rb, config.y_lb, config.y_ub],
            cmap="jet",
            origin="lower",
            vmin=-1.0,
            vmax=1.0
        )
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"t = {t_i:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")

    plt.suptitle("PINN Predicted Allen-Cahn Field Evolution u(t, x, y)", fontsize=14)
    plt.tight_layout()
    plt.show()