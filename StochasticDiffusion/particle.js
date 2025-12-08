/**
 * Particle System
 * Manages individual particles and their trajectories
 */

class Particle {
    constructor(x, t = 0) {
        this.x = x; // Position in distribution space
        this.t = t; // Time in [0, 1]
        this.trajectory = [{ x: x, t: t }]; // Store trajectory for visualization
        this.active = true;
        this.color = this.generateColor();
    }

    /**
     * Generate a random color for the particle
     */
    generateColor() {
        const hue = Math.random() * 360;
        const saturation = 60 + Math.random() * 40;
        const lightness = 50 + Math.random() * 30;
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }

    /**
     * Update particle position using diffusion model
     */
    update(diffusionModel, dt, stochasticity) {
        if (!this.active || this.t >= 1.0) {
            this.active = false;
            return;
        }

        const newX = diffusionModel.evolveParticle(this.x, this.t, dt, stochasticity);
        const newT = Math.min(this.t + dt, 1.0);

        this.x = newX;
        this.t = newT;
        
        // Store trajectory point (limit to recent history for performance)
        this.trajectory.push({ x: newX, t: newT });
        if (this.trajectory.length > 100) {
            this.trajectory.shift();
        }
    }

    /**
     * Check if particle has reached the end
     */
    isComplete() {
        return this.t >= 1.0;
    }
}

class ParticleSystem {
    constructor() {
        this.particles = [];
        this.maxParticles = 1000; // Limit total particles for performance
    }

    /**
     * Spawn new particles from the noisy distribution
     */
    spawnParticles(count, diffusionModel) {
        for (let i = 0; i < count; i++) {
            if (this.particles.length >= this.maxParticles) {
                // Remove oldest completed particles
                const completed = this.particles.findIndex(p => p.isComplete());
                if (completed !== -1) {
                    this.particles.splice(completed, 1);
                } else {
                    break; // No room for new particles
                }
            }
            
            const x = diffusionModel.sampleNoisy();
            this.particles.push(new Particle(x, 0));
        }
    }

    /**
     * Update all active particles
     */
    update(diffusionModel, dt, stochasticity) {
        this.particles.forEach(particle => {
            particle.update(diffusionModel, dt, stochasticity);
        });
    }

    /**
     * Get all active particles
     */
    getActiveParticles() {
        return this.particles.filter(p => p.active);
    }

    /**
     * Clear all particles
     */
    clear() {
        this.particles = [];
    }

    /**
     * Remove completed particles
     */
    cleanup() {
        this.particles = this.particles.filter(p => !p.isComplete() || p.trajectory.length > 0);
    }
}

