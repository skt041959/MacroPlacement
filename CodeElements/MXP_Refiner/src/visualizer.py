import json
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from config import Config

class DashboardGenerator:
    def __init__(self, output_path="dashboard.html"):
        self.output_path = output_path

    def _plot_to_base64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    def generate(self, history, layout_snapshots, netlist=None):
        """
        history: dict with keys 'rewards', 'losses' (list of floats)
        layout_snapshots: list of lists (snapshots of macros at each step of an episode)
                          each macro is a dict {'x', 'y', 'w', 'h', 'id'}
        netlist: list of tuples (u, v, weight) [Optional]
        """
        
        # 1. Generate Training Curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(history['rewards'])
        ax1.set_title("Total Reward per Episode")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True)
        
        ax2.plot(history['losses'])
        ax2.set_title("Loss per Step")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss")
        ax2.grid(True)
        
        metrics_img = self._plot_to_base64(fig)
        
        # 2. Prepare Data for JS Visualization
        # Normalize coordinates for canvas if necessary, but we can stick to 1000x1000
        canvas_width = Config.CANVAS_WIDTH
        canvas_height = Config.CANVAS_HEIGHT
        
        snapshots_json = json.dumps(layout_snapshots)
        netlist_json = json.dumps(netlist) if netlist else "[]"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MXP Refiner Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }}
        h1, h2 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: #fff; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }}
        .metrics {{ text-align: center; margin-bottom: 30px; }}
        .metrics img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .visualization {{ display: flex; flex-direction: column; align-items: center; }}
        #layoutCanvas {{ border: 1px solid #333; background-color: #f0f0f0; margin-top: 10px; }}
        .controls {{ margin-top: 10px; }}
        button {{ padding: 8px 16px; cursor: pointer; background-color: #007bff; color: white; border: none; border-radius: 4px; font-size: 14px; }}
        button:hover {{ background-color: #0056b3; }}
        button:disabled {{ background-color: #ccc; }}
        #stepInfo {{ margin-left: 10px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MXP Refiner Training Dashboard</h1>
        
        <div class="metrics">
            <h2>Training Metrics</h2>
            <img src="data:image/png;base64,{metrics_img}" alt="Training Metrics">
        </div>
        
        <div class="visualization">
            <h2>Layout Evolution (Last Episode)</h2>
            <canvas id="layoutCanvas" width="{int(canvas_width/2)}" height="{int(canvas_height/2)}"></canvas>
            <div class="controls">
                <button id="btnPrev" onclick="prevStep()">Previous</button>
                <button id="btnPlay" onclick="togglePlay()">Play</button>
                <button id="btnNext" onclick="nextStep()">Next</button>
                <span id="stepInfo">Step: 0 / 0</span>
            </div>
        </div>
    </div>

    <script>
        const snapshots = {snapshots_json};
        const netlist = {netlist_json};
        const canvas = document.getElementById('layoutCanvas');
        const ctx = canvas.getContext('2d');
        const scale = 0.5; // Scale down for display
        
        let currentStep = 0;
        let isPlaying = false;
        let playInterval;
        
        function drawLayout(stepIndex) {{
            if (stepIndex < 0 || stepIndex >= snapshots.length) return;
            
            const macros = snapshots[stepIndex];
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw boundary
            ctx.strokeStyle = '#999';
            ctx.strokeRect(0, 0, {canvas_width} * scale, {canvas_height} * scale);
            
            // Draw Netlist Connections (Background)
            if (netlist && netlist.length > 0) {{
                ctx.beginPath();
                ctx.strokeStyle = 'rgba(150, 150, 150, 0.2)'; // Very faint
                ctx.lineWidth = 1;
                
                netlist.forEach(net => {{
                    const u = net[0];
                    const v = net[1];
                    
                    if (macros[u] && macros[v]) {{
                        const x1 = (macros[u].x + macros[u].w / 2) * scale;
                        const y1 = (macros[u].y + macros[u].h / 2) * scale;
                        const x2 = (macros[v].x + macros[v].w / 2) * scale;
                        const y2 = (macros[v].y + macros[v].h / 2) * scale;
                        
                        ctx.moveTo(x1, y1);
                        ctx.lineTo(x2, y2);
                    }}
                }});
                ctx.stroke();
            }}
            
            // Draw Macros
            macros.forEach(macro => {{
                const x = macro.x * scale;
                const y = macro.y * scale;
                const w = macro.w * scale;
                const h = macro.h * scale;
                
                ctx.fillStyle = 'rgba(0, 123, 255, 0.6)';
                ctx.strokeStyle = '#0056b3';
                ctx.lineWidth = 1;
                
                ctx.fillRect(x, y, w, h);
                ctx.strokeRect(x, y, w, h);
                
                // Draw ID
                ctx.fillStyle = '#000';
                ctx.font = '10px Arial';
                ctx.fillText(macro.id, x + 2, y + 10);
            }});
            
            document.getElementById('stepInfo').innerText = `Step: ${{stepIndex}} / ${{snapshots.length - 1}}`;
        }}
        
        function prevStep() {{
            if (currentStep > 0) {{
                currentStep--;
                drawLayout(currentStep);
            }}
        }}
        
        function nextStep() {{
            if (currentStep < snapshots.length - 1) {{
                currentStep++;
                drawLayout(currentStep);
            }}
        }}
        
        function togglePlay() {{
            const btn = document.getElementById('btnPlay');
            if (isPlaying) {{
                clearInterval(playInterval);
                btn.innerText = 'Play';
                isPlaying = false;
            }} else {{
                btn.innerText = 'Pause';
                isPlaying = true;
                playInterval = setInterval(() => {{
                    if (currentStep < snapshots.length - 1) {{
                        nextStep();
                    }} else {{
                        togglePlay(); // Stop at end
                    }}
                }}, 200);
            }}
        }}
        
        // Initial draw
        drawLayout(0);
    </script>
</body>
</html>
        """
        
        with open(self.output_path, "w") as f:
            f.write(html_content)
        
        print(f"Dashboard generated at: {os.path.abspath(self.output_path)}")
