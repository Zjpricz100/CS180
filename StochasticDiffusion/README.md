# Diffusion Models Visualization Tool

An interactive visualization tool that demonstrates the difference between deterministic flow matching and stochastic DDPM (Denoising Diffusion Probabilistic Models) sampling processes.

## Features

- **Visual Comparison**: See how particles flow from a noisy Gaussian distribution (left) to a clean bimodal distribution (right)
- **Interactive Controls**:
  - **Stochasticity Slider**: Adjust from 0 (deterministic flow matching) to 1 (full DDPM stochasticity)
  - **Particle Count Slider**: Control how many particles spawn per second
- **Real-time Animation**: Watch particles move through the diffusion process in real-time
- **Trajectory Visualization**: Each particle leaves a colored trail showing its path

## How to Use

1. Open `index.html` in a modern web browser (Chrome, Firefox, Safari, or Edge)
2. Use the sliders to adjust:
   - **Stochasticity**: At 0, particles follow deterministic trajectories. At 1, particles exhibit full stochastic DDPM behavior with random diffusion
   - **Particles per Second**: Control the density of particles in the visualization
3. Click **Reset** to clear all particles
4. Click **Pause/Resume** to pause the animation

## Understanding the Visualization

- **Left side (t=0)**: Noisy distribution - a single Gaussian bell curve representing noisy data
- **Right side (t=1)**: Clean distribution - a bimodal distribution with two peaks representing clean data
- **Particles**: Colored dots that spawn from the noisy distribution and flow to the clean distribution
- **Trajectories**: Colored lines showing the path each particle takes

### Key Concepts

- **Flow Matching (Stochasticity = 0)**: Particles follow a deterministic vector field, moving smoothly along learned paths
- **DDPM (Stochasticity = 1)**: Particles follow stochastic trajectories with added diffusion noise, creating more varied paths
- **Intermediate Values**: Mix deterministic and stochastic behavior

## Technical Details

The tool implements:
- A simplified diffusion model with interpolated distributions
- A learned vector field that guides particles from noise to clean data
- Stochastic noise injection for DDPM sampling
- Real-time particle system with trajectory tracking

## Local Development

Simply open `index.html` in your browser. No build process or server required - it runs entirely client-side.

## Future Integration

This tool is designed to be easily integrated into a website. The code is modular and can be embedded in a larger web application.

