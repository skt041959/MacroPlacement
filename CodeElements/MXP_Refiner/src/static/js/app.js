let currentState = {
    samples: [],
    total: 0,
    page: 1,
    perPage: 10,
    selectedSampleIdx: null,
    category: '',
    minMse: ''
};

const vizRef = new MacroVisualizer('canvasRef');
const vizDist = new MacroVisualizer('canvasDist');
const vizRest = new MacroVisualizer('canvasRest');

const visualizers = [vizRef, vizDist, vizRest];

async function loadSamples() {
    const loader = document.getElementById('loadingSpinner');
    loader.style.display = 'inline-block';
    
    const params = new URLSearchParams({
        page: currentState.page,
        per_page: currentState.perPage,
        category: currentState.category,
        min_mse: currentState.minMse
    });

    try {
        const response = await fetch(`/api/samples?${params.toString()}`);
        const data = await response.json();
        
        currentState.samples = data.samples;
        currentState.total = data.total;
        
        renderSampleList();
        updatePaginationUI();
    } catch (err) {
        console.error("Failed to load samples:", err);
    } finally {
        loader.style.display = 'none';
    }
}

function renderSampleList() {
    const list = document.getElementById('sampleList');
    list.innerHTML = '';

    currentState.samples.forEach((sample, idx) => {
        const item = document.createElement('a');
        item.className = `list-group-item list-group-item-action sample-item ${currentState.selectedSampleIdx === idx ? 'active' : ''}`;
        
        // Use a small part of metrics for the label
        const mse = sample.metrics.mse.toFixed(2);
        item.innerHTML = `
            <div class="d-flex w-100 justify-content-between">
                <h6 class="mb-1">Sample #${sample.id}</h6>
                <small class="text-muted">MSE: ${mse}</small>
            </div>
            <small>${sample.category || 'N/A'}</small>
        `;
        
        item.onclick = () => selectSample(idx);
        list.appendChild(item);
    });

    if (currentState.samples.length === 0) {
        list.innerHTML = '<div class="p-3 text-center text-muted">No samples found</div>';
    }
}

function updatePaginationUI() {
    document.getElementById('pageInfo').innerText = `Page ${currentState.page} of ${Math.ceil(currentState.total / currentState.perPage) || 1}`;
    document.getElementById('prevPage').parentElement.classList.toggle('disabled', currentState.page <= 1);
    document.getElementById('nextPage').parentElement.classList.toggle('disabled', currentState.page >= Math.ceil(currentState.total / currentState.perPage));
    document.getElementById('statsInfo').innerText = `Total: ${currentState.total} samples`;
}

function selectSample(idx) {
    currentState.selectedSampleIdx = idx;
    renderSampleList();

    const sample = currentState.samples[idx];
    document.getElementById('noSampleSelected').style.display = 'none';
    document.getElementById('sampleDetail').style.display = 'block';
    
    document.getElementById('detailTitle').innerText = `Sample #${sample.id}`;
    
    // Render metrics
    const mContainer = document.getElementById('detailMetrics');
    mContainer.innerHTML = `
        <span class="badge bg-danger metric-badge">MSE: ${sample.metrics.mse.toFixed(4)}</span>
        <span class="badge bg-warning text-dark metric-badge">Overlap Reduction: ${((1 - sample.metrics.overlap_restored/(sample.metrics.overlap_disturbed+1e-6))*100).toFixed(1)}%</span>
        <span class="badge bg-success metric-badge">Align Recovery: ${(sample.metrics.alignment_recovery * 100).toFixed(1)}%</span>
    `;

    // Fetch edges (In a real app, these might be in the sample data or fetched separately)
    // For now, assume they are processed or we can get them from aligned/disturbed?
    // In our generate_val_results.py, we didn't save edges to keep file size small.
    // Let's assume for now they are not available or we need to add them.
    // If edges are missing, we just show macros.
    
    vizRef.setData(sample.aligned, sample.edges);
    vizDist.setData(sample.disturbed, sample.edges);
    vizRest.setData(sample.restored, sample.edges);
}

// Event Listeners
document.getElementById('prevPage').onclick = (e) => {
    e.preventDefault();
    if (currentState.page > 1) {
        currentState.page--;
        loadSamples();
    }
};

document.getElementById('nextPage').onclick = (e) => {
    e.preventDefault();
    currentState.page++;
    loadSamples();
};

document.getElementById('btnApplyFilters').onclick = () => {
    currentState.category = document.getElementById('filterCategory').value;
    currentState.minMse = document.getElementById('filterMSE').value;
    currentState.page = 1;
    loadSamples();
};

// Layer Toggles
['Phys', 'Align', 'Logic'].forEach(type => {
    document.getElementById(`toggle${type}`).onchange = (e) => {
        const layers = {};
        layers[type.toLowerCase()] = e.target.checked;
        visualizers.forEach(v => v.setLayers(layers));
    };
});

// Tooltip / Inspection
function handleMouseMove(e, viz) {
    const rect = viz.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const macro = viz.getMacroAt(x, y);
    const tooltip = document.getElementById('tooltip');
    
    if (macro) {
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 10) + 'px';
        tooltip.style.top = (e.clientY + 10) + 'px';
        tooltip.innerHTML = `
            <strong>Macro ID: ${macro.id}</strong><br>
            X: ${macro.x.toFixed(1)} Y: ${macro.y.toFixed(1)}<br>
            W: ${macro.w.toFixed(1)} H: ${macro.h.toFixed(1)}
        `;
    } else {
        tooltip.style.display = 'none';
    }
}

[vizRef, vizDist, vizRest].forEach(viz => {
    viz.canvas.onmousemove = (e) => handleMouseMove(e, viz);
    viz.canvas.onmouseout = () => { document.getElementById('tooltip').style.display = 'none'; };
});

// Initial Load
loadSamples();
