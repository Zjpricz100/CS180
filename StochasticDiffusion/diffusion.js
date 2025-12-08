/**
 * Diffusion Models Math
 * Implements both deterministic flow matching and stochastic DDPM sampling
 */

class DiffusionModel {
    constructor() {
        // Distribution parameters
        // Make noise distribution wider to span full vertical range
        this.noiseMean = 0.0;
        this.noiseStd = 0.7; // Increased from 0.3 to span more of [-2, 2] range
        
        // Clean distribution: bimodal (2 Gaussians)
        // Increase separation between modes and make them wider
        this.cleanMode1 = -1.0; // Moved farther apart from 0.4
        this.cleanMode2 = 1.0;  // Moved farther apart from 0.4
        this.cleanStd = 0.3;    // Increased from 0.15 to make modes more visible
        this.modeWeight = 0.5; // Equal weight for both modes
        
        // Time parameters
        this.tMin = 0.0;
        this.tMax = 1.0;
    }

    /**
     * Compute the noisy distribution at time t
     * Interpolates between noise (t=0) and clean (t=1)
     */
    getNoisyDistribution(x, t) {
        const noiseDist = this.gaussian(x, this.noiseMean, this.noiseStd);
        const cleanDist = this.getCleanDistribution(x);
        
        // Linear interpolation in probability space (simplified)
        // In practice, this would be more complex
        return (1 - t) * noiseDist + t * cleanDist;
    }

    /**
     * Get the clean (target) distribution
     * Bimodal: two Gaussians
     */
    getCleanDistribution(x) {
        const g1 = this.gaussian(x, this.cleanMode1, this.cleanStd);
        const g2 = this.gaussian(x, this.cleanMode2, this.cleanStd);
        return this.modeWeight * g1 + (1 - this.modeWeight) * g2;
    }

    /**
     * Gaussian probability density function
     */
    gaussian(x, mean, std) {
        const variance = std * std;
        const coefficient = 1.0 / Math.sqrt(2 * Math.PI * variance);
        const exponent = -0.5 * Math.pow((x - mean) / std, 2);
        return coefficient * Math.exp(exponent);
    }

    /**
     * Compute the learned vector field (velocity field) for flow matching
     * This is the deterministic direction particles should follow
     * 
     * The vector field points from the noisy distribution towards the clean distribution
     * Creates curved trajectories and allows particles to go to both modes
     */
    computeVectorField(x, t) {
        // At t=0, we're at the noise distribution (centered at 0)
        // At t=1, we're at the clean distribution (bimodal at -1.0 and 1.0)
        
        // Weighted combination towards both modes - allows particles to go to either
        // The weights depend on position and time to create curved paths
        const distToMode1 = Math.abs(x - this.cleanMode1);
        const distToMode2 = Math.abs(x - this.cleanMode2);
        
        // Compute weights - particles closer to a mode are more likely to go there
        // But we add time-dependent mixing so paths can curve
        const weight1 = Math.exp(-distToMode1 * 2) * (1 + 0.3 * Math.sin(t * Math.PI));
        const weight2 = Math.exp(-distToMode2 * 2) * (1 + 0.3 * Math.cos(t * Math.PI));
        const totalWeight = weight1 + weight2;
        
        // Weighted target position
        const targetX = (weight1 * this.cleanMode1 + weight2 * this.cleanMode2) / totalWeight;
        
        // Base velocity towards weighted target
        const direction = targetX - x;
        
        // Time-dependent velocity magnitude (stronger at beginning)
        // Scale up the velocity to make particles move more noticeably
        const timeScale = 1.0 - t * 0.2;
        
        // Add curvature component for curved paths
        const curvature = Math.sin(t * Math.PI * 2) * 0.3 * (1 - t);
        
        // Position-dependent component for smooth curves
        const centerComponent = -x * 0.4 * Math.exp(-t * 1.5) * (1 - t);
        
        // Combine for curved trajectory - scaled to move at appropriate speed
        // The velocity should be strong enough to move particles from center to modes
        const velocity = direction * timeScale * 1.0 + 
                        curvature * Math.sign(x) * (1 - t) +
                        centerComponent;
        
        return velocity;
    }

    /**
     * Deterministic flow matching step
     * Follows the vector field exactly
     */
    flowMatchingStep(x, t, dt) {
        const velocity = this.computeVectorField(x, t);
        return x + velocity * dt;
    }

    /**
     * Stochastic DDPM step
     * Adds noise to the deterministic flow
     */
    ddpmStep(x, t, dt, stochasticity) {
        // Deterministic component
        const deterministic = this.flowMatchingStep(x, t, dt);
        
        // Stochastic component: diffusion noise
        // Noise is stronger when we're further from the clean distribution (smaller t)
        const noiseScale = Math.sqrt(dt) * (1 - t) * stochasticity;
        const noise = this.sampleGaussian(0, noiseScale);
        
        return deterministic + noise;
    }

    /**
     * Sample from a Gaussian distribution using Box-Muller transform
     */
    sampleGaussian(mean, std) {
        // Box-Muller transform for generating Gaussian random numbers
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return mean + std * z0;
    }

    /**
     * Sample from the noisy distribution (t=0)
     */
    sampleNoisy() {
        return this.sampleGaussian(this.noiseMean, this.noiseStd);
    }

    /**
     * Evolve a particle from time t to t+dt
     * Combines deterministic and stochastic components based on stochasticity parameter
     */
    evolveParticle(x, t, dt, stochasticity) {
        if (stochasticity === 0) {
            // Pure deterministic flow matching
            return this.flowMatchingStep(x, t, dt);
        } else {
            // Stochastic DDPM with varying noise level
            return this.ddpmStep(x, t, dt, stochasticity);
        }
    }

    /**
     * Get the maximum probability density for scaling visualization
     */
    getMaxDensity() {
        // Approximate maximum of the distributions
        const noiseMax = this.gaussian(this.noiseMean, this.noiseMean, this.noiseStd);
        const cleanMax1 = this.gaussian(this.cleanMode1, this.cleanMode1, this.cleanStd);
        const cleanMax2 = this.gaussian(this.cleanMode2, this.cleanMode2, this.cleanStd);
        return Math.max(noiseMax, cleanMax1, cleanMax2) * 1.2; // Add some padding
    }
}

