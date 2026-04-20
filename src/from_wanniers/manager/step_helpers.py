import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def get_complex_rgb(z, mag_power=0.7):
    """Generates an RGB image where Hue=Phase and Value=Magnitude."""
    mag = np.abs(z)
    phase = np.angle(z)
    
    # Normalize magnitude with power law for better contrast in low-density regions
    mag_norm = (mag / (mag.max() + 1e-12))**mag_power
    
    # Map phase [-pi, pi] to [0, 1]
    hue = (phase + np.pi) / (2 * np.pi)
    
    hsv = np.stack([hue, np.ones_like(hue), mag_norm], axis=-1)
    return hsv_to_rgb(hsv)

def add_phase_wheel(ax):
    """Adds a small inset unit circle showing the phase-color mapping."""
    res = 50
    r, th = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 2*np.pi, res))
    # Create the wheel data
    hue = th / (2 * np.pi)
    sat = np.ones_like(hue)
    val = r 
    hsv_wheel = np.stack([hue, sat, val], axis=-1)
    rgb_wheel = hsv_to_rgb(hsv_wheel)
    
    # Plot in polar coordinates
    ax_inset = ax.inset_axes([0.75, 0.02, 0.22, 0.22], projection='polar')
    ax_inset.pcolormesh(th, r, rgb_wheel, shading='auto')
    ax_inset.set_yticklabels([]) # Hide radius labels
    ax_inset.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax_inset.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$'], fontsize=8)
    ax_inset.spines['polar'].set_visible(False)
    ax_inset.grid(False)

def psi_slice_to_3_panels(psi_slice):
    fig, axes = plt.subplots(1, 3, figsize=(12, 8), constrained_layout=True)
    
    # 1. Magnitude Squared (Probability Density)
    prob_density = np.abs(psi_slice)**2
    im0 = axes[0].imshow(prob_density, origin="lower", cmap="viridis")
    axes[0].set_title(r"$|\psi|^2$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Phase Plot
    # We use 'twilight' or 'hsv' because they are cyclic (end colors match)
    phase_data = np.angle(psi_slice)
    im1 = axes[1].imshow(
        phase_data, 
        origin="lower", 
        cmap="twilight",
        vmin=-np.pi, 
        vmax=np.pi
    )
    axes[1].set_title(r"Phase [arg($\psi$)]")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_ticks([-np.pi, 0, np.pi])
    cbar1.set_ticklabels([r'-$\pi$', '0', r'$\pi$'])
    
    # 3. Domain Coloring (Combined)
    rgb_image = get_complex_rgb(psi_slice)
    axes[2].imshow(rgb_image, origin="lower")
    axes[2].set_title(r"$\psi$ Complex Hue/Value Map")
    add_phase_wheel(axes[2]) # This adds the "unit circle" legend

    return fig, axes