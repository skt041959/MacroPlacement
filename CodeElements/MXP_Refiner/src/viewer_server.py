import torch
import os
from flask import Flask, jsonify, request, render_template
from config import Config

class DataLoader:
    def __init__(self, results_path):
        self.results_path = results_path
        self.data = []
        if os.path.exists(results_path):
            print(f"Loading results from {results_path}")
            self.data = torch.load(results_path, weights_only=False)
        else:
            print(f"Warning: {results_path} not found.")

    def get_total_count(self):
        return len(self.data)

    def get_page(self, page=1, per_page=10, category=None, min_mse=None):
        filtered_data = self.data
        
        if category:
            filtered_data = [d for d in filtered_data if d.get('category') == category]
            
        if min_mse:
            filtered_data = [d for d in filtered_data if d['metrics']['mse'] >= float(min_mse)]
            
        start = (page - 1) * per_page
        end = start + per_page
        return filtered_data[start:end], len(filtered_data)

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Default path
results_path = os.path.join(Config.DATASET_DIR, "val_results.pt")
app.data_loader = DataLoader(results_path)

@app.route('/')
def index():
    return render_template('viewer.html')

@app.route('/api/samples')
def get_samples():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    category = request.args.get('category')
    min_mse = request.args.get('min_mse')
    
    samples, total = app.data_loader.get_page(page, per_page, category, min_mse)
    
    return jsonify({
        'samples': samples,
        'total': total,
        'page': page,
        'per_page': per_page
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
