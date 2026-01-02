import pytest
from unittest.mock import MagicMock, patch
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

@pytest.fixture
def mock_torch_load():
    with patch('torch.load') as mock:
        # Mock data structure matching generate_val_results.py output
        mock.return_value = [
            {
                'id': i,
                'metrics': {'mse': 0.1 * i, 'overlap_disturbed': 100, 'overlap_restored': 10},
                'aligned': [{'id': 0, 'w': 10, 'h': 10}], # simplified
                'disturbed': [{'id': 0, 'w': 10, 'h': 10}],
                'restored': [{'id': 0, 'w': 10, 'h': 10}],
                'category': 'grid' if i % 2 == 0 else 'clustered' # hypothetically added to data
            }
            for i in range(20)
        ]
        yield mock

@patch('os.path.exists')
@patch('torch.load')
def test_data_loader(mock_torch_load, mock_exists):
    from viewer_server import DataLoader
    
    mock_exists.return_value = True
    mock_torch_load.return_value = [{'id': i, 'metrics': {'mse': 0.1}} for i in range(20)]
    
    loader = DataLoader("dummy_path")
    assert len(loader.data) == 20
    assert loader.get_total_count() == 20
    
    # Test pagination
    page1, total = loader.get_page(page=1, per_page=5)
    assert len(page1) == 5
    assert total == 20
    assert page1[0]['id'] == 0
    
    page2, total = loader.get_page(page=2, per_page=5)
    assert len(page2) == 5
    assert page2[0]['id'] == 5

@patch('os.path.exists')
@patch('torch.load')
def test_api_samples(mock_torch_load, mock_exists):
    from viewer_server import app, DataLoader
    
    mock_exists.return_value = True
    mock_torch_load.return_value = [{'id': i, 'metrics': {'mse': 0.1}} for i in range(20)]
    
    # Inject mock loader
    app.config['TESTING'] = True
    loader = DataLoader("dummy_path")
    app.data_loader = loader 
    
    client = app.test_client()
    
    response = client.get('/api/samples?page=1&per_page=10')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data['samples']) == 10
    assert data['total'] == 20
    assert data['page'] == 1

@patch('os.path.exists')
@patch('torch.load')
def test_api_filter(mock_torch_load, mock_exists):
    from viewer_server import app, DataLoader
    
    mock_exists.return_value = True
    mock_torch_load.return_value = [
        {'id': i, 'category': 'grid' if i % 2 == 0 else 'rows', 'metrics': {'mse': i}} 
        for i in range(20)
    ]
    
    app.config['TESTING'] = True
    loader = DataLoader("dummy_path")
    app.data_loader = loader
    
    client = app.test_client()
    
    # Filter by category
    response = client.get('/api/samples?category=grid')
    data = json.loads(response.data)
    assert data['total'] == 10
    assert all(s['category'] == 'grid' for s in data['samples'])
    
    # Filter by min_mse
    response = client.get('/api/samples?min_mse=15')
    data = json.loads(response.data)
    assert data['total'] == 5 # 15, 16, 17, 18, 19
