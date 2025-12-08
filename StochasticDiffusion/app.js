/**
 * Main Application
 * Coordinates visualization, particle system, and user interactions
 */

class VisualizationApp {
    constructor() {
        this.canvas = document.getElementById('visualization-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Set canvas size
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Initialize models
        this.diffusionModel = new DiffusionModel();
        this.particleSystem = new ParticleSystem();
        
        // Animation state
        this.animationId = null;
        this.isPaused = false;
        this.lastSpawnTime = performance.now();
        this.lastFrameTime = performance.now();
        
        // Parameters
        this.stochasticity = 0.5;
        this.particlesPerSecond = 10;
        this.dt = 0.02; // Time step for particle evolution (increased for more responsive movement)
        this.showTrails = true; // Show particle trajectories
        
        // Visualization parameters
        // x-axis = time (0 to 1), y-axis = position
        // Expand range to show full Gaussian tails (about 3-4 standard deviations)
        this.positionRange = [-2.0, 2.0]; // Position range (vertical) - expanded to show full distributions
        this.tRange = [0, 1]; // Time range (horizontal)
        this.maxDensity = this.diffusionModel.getMaxDensity();
        
        // Padding for distributions (so they don't touch edges)
        this.horizontalPadding = 0.05; // 5% padding on left/right
        this.verticalPadding = 0.1; // 10% padding on top/bottom
        
        // Setup controls
        this.setupControls();
        
        // Start animation
        this.animate();
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        const padding = 80;
        this.canvas.width = Math.min(1200, window.innerWidth - padding);
        this.canvas.height = 600;
    }

    setupControls() {
        const stochasticitySlider = document.getElementById('stochasticity-slider');
        const particleCountSlider = document.getElementById('particle-count-slider');
        const stochasticityValue = document.getElementById('stochasticity-value');
        const particleCountValue = document.getElementById('particle-count-value');
        const resetBtn = document.getElementById('reset-btn');
        const pauseBtn = document.getElementById('pause-btn');

        stochasticitySlider.addEventListener('input', (e) => {
            this.stochasticity = parseFloat(e.target.value);
            stochasticityValue.textContent = this.stochasticity.toFixed(2);
        });

        particleCountSlider.addEventListener('input', (e) => {
            this.particlesPerSecond = parseInt(e.target.value);
            particleCountValue.textContent = this.particlesPerSecond;
        });

        resetBtn.addEventListener('click', () => {
            this.particleSystem.clear();
            this.lastSpawnTime = performance.now();
        });

        pauseBtn.addEventListener('click', () => {
            this.isPaused = !this.isPaused;
            pauseBtn.textContent = this.isPaused ? 'Resume' : 'Pause';
            if (!this.isPaused) {
                this.animate();
            }
        });

        const showTrailsCheckbox = document.getElementById('show-trails-checkbox');
        showTrailsCheckbox.addEventListener('change', (e) => {
            this.showTrails = e.target.checked;
        });
    }

    /**
     * Convert world coordinates to canvas coordinates
     * t = time (horizontal), x = position (vertical)
     * Particles flow through the heatmap area
     */
    worldToCanvas(t, x) {
        const paddingX = this.canvas.width * this.horizontalPadding;
        const paddingY = this.canvas.height * this.verticalPadding;
        const usableWidth = this.canvas.width - 2 * paddingX;
        const usableHeight = this.canvas.height - 2 * paddingY;
        
        const distributionWidth = this.canvas.width * 0.12;
        const heatmapStartX = paddingX + distributionWidth;
        const heatmapWidth = usableWidth - 2 * distributionWidth;
        
        // Map time to heatmap area (t=0 at left edge of heatmap, t=1 at right edge)
        const canvasX = heatmapStartX + ((t - this.tRange[0]) / (this.tRange[1] - this.tRange[0])) * heatmapWidth;
        const canvasY = paddingY + ((this.positionRange[1] - x) / (this.positionRange[1] - this.positionRange[0])) * usableHeight;
        return { x: canvasX, y: canvasY };
    }

    /**
     * Convert canvas coordinates to world coordinates
     */
    canvasToWorld(canvasX, canvasY) {
        const paddingX = this.canvas.width * this.horizontalPadding;
        const paddingY = this.canvas.height * this.verticalPadding;
        const usableWidth = this.canvas.width - 2 * paddingX;
        const usableHeight = this.canvas.height - 2 * paddingY;
        
        const t = ((canvasX - paddingX) / usableWidth) * (this.tRange[1] - this.tRange[0]) + this.tRange[0];
        const x = this.positionRange[1] - ((canvasY - paddingY) / usableHeight) * (this.positionRange[1] - this.positionRange[0]);
        return { x, t };
    }

    /**
     * Draw a vertical distribution curve (Gaussian or bimodal)
     * Positioned on left or right edge, width represents probability density
     */
    drawVerticalDistribution(t, isLeft) {
        const steps = 400; // More steps for smoother curves across full range
        const dx = (this.positionRange[1] - this.positionRange[0]) / steps;
        
        const paddingX = this.canvas.width * this.horizontalPadding;
        const paddingY = this.canvas.height * this.verticalPadding;
        const usableHeight = this.canvas.height - 2 * paddingY;
        const distributionWidth = this.canvas.width * 0.12; // Width of distribution curves
        
        // Position: left edge for t=0, right edge for t=1
        const canvasX = isLeft ? paddingX : (this.canvas.width - paddingX);
        
        const path = [];
        
        // Draw across the FULL vertical range to show complete Gaussian tails
        for (let i = 0; i <= steps; i++) {
            const x = this.positionRange[0] + i * dx;
            const density = this.diffusionModel.getNoisyDistribution(x, t);
            
            // Normalize density - ensure we show even very small densities
            const normalizedDensity = Math.max(0, density / this.maxDensity);
            // Use a minimum width threshold so tails are visible
            const minWidth = distributionWidth * 0.01; // 1% minimum width for visibility
            const width = Math.max(minWidth, normalizedDensity * distributionWidth);
            
            // Convert position to canvas Y - map full range to full height
            const canvasY = paddingY + ((this.positionRange[1] - x) / (this.positionRange[1] - this.positionRange[0])) * usableHeight;
            
            // For left distribution, extend to the right. For right, extend to the left
            const curveX = isLeft ? (canvasX + width) : (canvasX - width);
            
            path.push({ x: curveX, y: canvasY });
        }
        
        // Draw filled shape
        this.ctx.beginPath();
        
        // Draw the curve
        for (let i = 0; i < path.length; i++) {
            if (i === 0) {
                this.ctx.moveTo(path[i].x, path[i].y);
            } else {
                this.ctx.lineTo(path[i].x, path[i].y);
            }
        }
        
        // Close the shape by going back along the base line
        this.ctx.lineTo(canvasX, paddingY + usableHeight);
        this.ctx.lineTo(canvasX, paddingY);
        this.ctx.closePath();
        
        // Color gradient based on density (red/orange for high, blue for low)
        const gradient = this.ctx.createLinearGradient(
            isLeft ? canvasX : canvasX - distributionWidth,
            0,
            isLeft ? canvasX + distributionWidth : canvasX,
            0
        );
        
        // Heatmap colors: dark blue -> light blue -> green -> yellow -> orange -> red
        gradient.addColorStop(0, 'rgba(0, 0, 100, 0.8)');      // Dark blue
        gradient.addColorStop(0.3, 'rgba(0, 150, 255, 0.7)');  // Light blue
        gradient.addColorStop(0.5, 'rgba(0, 255, 150, 0.7)');  // Green
        gradient.addColorStop(0.7, 'rgba(255, 255, 0, 0.8)');  // Yellow
        gradient.addColorStop(0.9, 'rgba(255, 150, 0, 0.9)');  // Orange
        gradient.addColorStop(1, 'rgba(255, 50, 0, 1.0)');    // Red
        
        this.ctx.fillStyle = gradient;
        this.ctx.fill();
        
        // Outline
        this.ctx.beginPath();
        for (let i = 0; i < path.length; i++) {
            if (i === 0) {
                this.ctx.moveTo(path[i].x, path[i].y);
            } else {
                this.ctx.lineTo(path[i].x, path[i].y);
            }
        }
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Add label
        this.ctx.fillStyle = 'rgba(255, 255, 255, 1.0)';
        this.ctx.font = 'bold 16px sans-serif';
        this.ctx.textAlign = 'center';
        const label = isLeft ? 'Prior' : 'Data';
        this.ctx.fillText(label, canvasX, paddingY - 10);
    }

    /**
     * Draw the heatmap showing probability density evolution
     */
    drawHeatmap() {
        const paddingX = this.canvas.width * this.horizontalPadding;
        const paddingY = this.canvas.height * this.verticalPadding;
        const usableWidth = this.canvas.width - 2 * paddingX;
        const usableHeight = this.canvas.height - 2 * paddingY;
        
        const distributionWidth = this.canvas.width * 0.12;
        const heatmapStartX = paddingX + distributionWidth;
        const heatmapWidth = usableWidth - 2 * distributionWidth;
        
        const timeSteps = 150;
        const positionSteps = 250;
        
        // Draw heatmap pixel by pixel
        for (let ti = 0; ti < timeSteps; ti++) {
            const t = ti / (timeSteps - 1);
            const canvasX = heatmapStartX + (ti / (timeSteps - 1)) * heatmapWidth;
            const nextCanvasX = heatmapStartX + ((ti + 1) / (timeSteps - 1)) * heatmapWidth;
            const pixelWidth = Math.max(1, nextCanvasX - canvasX);
            
            for (let pi = 0; pi < positionSteps; pi++) {
                const x = this.positionRange[0] + (pi / (positionSteps - 1)) * (this.positionRange[1] - this.positionRange[0]);
                const density = this.diffusionModel.getNoisyDistribution(x, t);
                const normalizedDensity = density / this.maxDensity;
                
                const canvasY = paddingY + ((this.positionRange[1] - x) / (this.positionRange[1] - this.positionRange[0])) * usableHeight;
                const nextX = this.positionRange[0] + ((pi + 1) / (positionSteps - 1)) * (this.positionRange[1] - this.positionRange[0]);
                const nextCanvasY = paddingY + ((this.positionRange[1] - nextX) / (this.positionRange[1] - this.positionRange[0])) * usableHeight;
                const pixelHeight = Math.abs(nextCanvasY - canvasY);
                
                // Color based on density (heatmap style)
                const color = this.densityToColor(normalizedDensity);
                
                this.ctx.fillStyle = color;
                this.ctx.fillRect(canvasX, Math.min(canvasY, nextCanvasY), pixelWidth, pixelHeight);
            }
        }
        
        // Add axis label
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        this.ctx.font = '14px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Timestep (t)', heatmapStartX + heatmapWidth / 2, this.canvas.height - 5);
    }
    
    /**
     * Convert normalized density to heatmap color
     */
    densityToColor(density) {
        // Clamp density to [0, 1]
        density = Math.max(0, Math.min(1, density));
        
        // Heatmap: dark blue -> light blue -> green -> yellow -> orange -> red
        let r, g, b;
        
        if (density < 0.2) {
            // Dark blue to light blue
            const t = density / 0.2;
            r = 0;
            g = Math.floor(150 * t);
            b = Math.floor(100 + 155 * t);
        } else if (density < 0.4) {
            // Light blue to green
            const t = (density - 0.2) / 0.2;
            r = 0;
            g = Math.floor(150 + 105 * t);
            b = Math.floor(255 - 105 * t);
        } else if (density < 0.6) {
            // Green to yellow
            const t = (density - 0.4) / 0.2;
            r = Math.floor(255 * t);
            g = 255;
            b = Math.floor(150 - 150 * t);
        } else if (density < 0.8) {
            // Yellow to orange
            const t = (density - 0.6) / 0.2;
            r = 255;
            g = Math.floor(255 - 105 * t);
            b = 0;
        } else {
            // Orange to red
            const t = (density - 0.8) / 0.2;
            r = 255;
            g = Math.floor(150 - 100 * t);
            b = 0;
        }
        
        return `rgba(${r}, ${g}, ${b}, 0.9)`;
    }
    
    /**
     * Draw only the two distributions: left (t=0) and right (t=1)
     */
    drawDistributions() {
        // Draw left distribution (Prior/Noisy Gaussian)
        this.drawVerticalDistribution(0, true);
        // Draw right distribution (Data/Clean Bimodal)
        this.drawVerticalDistribution(1, false);
    }

    /**
     * Draw a single particle
     */
    drawParticle(particle) {
        const { x: canvasX, y: canvasY } = this.worldToCanvas(particle.t, particle.x);
        
        // Draw trajectory only if enabled
        if (this.showTrails && particle.trajectory.length > 1) {
            this.ctx.beginPath();
            this.ctx.strokeStyle = particle.color;
            this.ctx.lineWidth = 2;
            this.ctx.globalAlpha = 0.7;
            
            for (let i = 0; i < particle.trajectory.length; i++) {
                const point = particle.trajectory[i];
                const { x: px, y: py } = this.worldToCanvas(point.t, point.x);
                
                if (i === 0) {
                    this.ctx.moveTo(px, py);
                } else {
                    this.ctx.lineTo(px, py);
                }
            }
            this.ctx.stroke();
            this.ctx.globalAlpha = 1.0;
        }
        
        // Draw particle
        this.ctx.beginPath();
        this.ctx.arc(canvasX, canvasY, 4, 0, Math.PI * 2);
        this.ctx.fillStyle = particle.color;
        this.ctx.shadowBlur = 8;
        this.ctx.shadowColor = particle.color;
        this.ctx.fill();
        this.ctx.shadowBlur = 0;
    }

    /**
     * Draw all particles
     */
    drawParticles() {
        const activeParticles = this.particleSystem.getActiveParticles();
        activeParticles.forEach(particle => {
            this.drawParticle(particle);
        });
    }


    /**
     * Main render function
     */
    render() {
        // Clear canvas
        this.ctx.fillStyle = '#0a0a1a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw heatmap first (background)
        this.drawHeatmap();
        
        // Draw the two distributions on left and right
        this.drawDistributions();
        
        // Draw particles on top
        this.drawParticles();
    }

    /**
     * Update simulation
     */
    update(deltaTime) {
        if (this.isPaused) return;
        
        // Spawn new particles
        const currentTime = performance.now();
        const timeSinceLastSpawn = currentTime - this.lastSpawnTime;
        const spawnInterval = 1000 / this.particlesPerSecond; // milliseconds
        
        if (timeSinceLastSpawn >= spawnInterval) {
            this.particleSystem.spawnParticles(1, this.diffusionModel);
            this.lastSpawnTime = currentTime;
        }
        
        // Update particles
        // Use fixed time step for consistent physics, but scale by frame rate
        const frameScale = Math.min(deltaTime / 16, 2.0); // Cap at 2x to prevent large jumps
        const dt = this.dt * frameScale;
        this.particleSystem.update(this.diffusionModel, dt, this.stochasticity);
        
        // Cleanup old particles
        if (Math.random() < 0.1) { // Occasionally cleanup
            this.particleSystem.cleanup();
        }
    }

    /**
     * Animation loop
     */
    animate() {
        if (this.isPaused) return;
        
        const currentTime = performance.now();
        const deltaTime = currentTime - (this.lastFrameTime || currentTime);
        this.lastFrameTime = currentTime;
        
        this.update(deltaTime);
        this.render();
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

// Initialize app when page loads
window.addEventListener('DOMContentLoaded', () => {
    new VisualizationApp();
});

