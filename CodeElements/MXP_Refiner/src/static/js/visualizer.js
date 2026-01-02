class MacroVisualizer {
    constructor(canvasId, scale = 0.4) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.scale = scale;
        this.macros = [];
        this.edges = {};
        this.layers = {
            phys: true,
            align: true,
            logic: false
        };
        this.canvasWidth = 1000; // Original coordinates
        this.canvasHeight = 1000;
        
        // Setup scaling
        this.canvas.width = this.canvasWidth * this.scale;
        this.canvas.height = this.canvasHeight * this.scale;
    }

    setData(macros, edges = {}) {
        this.macros = macros;
        this.edges = edges;
        this.draw();
    }

    setLayers(layers) {
        this.layers = { ...this.layers, ...layers };
        this.draw();
    }

    getCenter(idx) {
        const m = this.macros[idx];
        if (!m) return { x: 0, y: 0 };
        return {
            x: (m.x + m.w / 2) * this.scale,
            y: (m.y + m.h / 2) * this.scale
        };
    }

    draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw boundary
        ctx.strokeStyle = '#999';
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.setLineDash([]);

        // 1. Draw Edges (Background)
        
        // Logic (Red, very faint)
        if (this.layers.logic && this.edges.logic) {
            ctx.strokeStyle = 'rgba(220, 53, 69, 0.2)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            this.edges.logic.forEach(e => {
                const p1 = this.getCenter(e[0]);
                const p2 = this.getCenter(e[1]);
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
            });
            ctx.stroke();
        }

        // Physical (Green, faint)
        if (this.layers.phys && this.edges.phys) {
            ctx.strokeStyle = 'rgba(40, 167, 69, 0.4)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            this.edges.phys.forEach(e => {
                const p1 = this.getCenter(e[0]);
                const p2 = this.getCenter(e[1]);
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
            });
            ctx.stroke();
        }

        // Align (Blue)
        if (this.layers.align && this.edges.align) {
            ctx.strokeStyle = 'rgba(0, 123, 255, 0.6)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            this.edges.align.forEach(e => {
                const p1 = this.getCenter(e[0]);
                const p2 = this.getCenter(e[1]);
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
            });
            ctx.stroke();
        }

        // 2. Draw Macros
        this.macros.forEach(m => {
            const x = m.x * this.scale;
            const y = m.y * this.scale;
            const w = m.w * this.scale;
            const h = m.h * this.scale;

            ctx.fillStyle = 'rgba(240, 240, 240, 0.8)';
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;

            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);

            // Draw ID if large enough
            if (w > 15) {
                ctx.fillStyle = '#000';
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(m.id, x + w / 2, y + h / 2 + 4);
            }
        });
    }

    getMacroAt(mouseX, mouseY) {
        // Simple hit test
        for (let i = this.macros.length - 1; i >= 0; i--) {
            const m = this.macros[i];
            const x = m.x * this.scale;
            const y = m.y * this.scale;
            const w = m.w * this.scale;
            const h = m.h * this.scale;

            if (mouseX >= x && mouseX <= x + w && mouseY >= y && mouseY <= y + h) {
                return m;
            }
        }
        return null;
    }
}
