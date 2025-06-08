from flask import Flask, render_template, url_for, jsonify, request
import torch
import torch.nn as nn
from graphviz import Digraph
from graphviz.backend import ExecutableNotFound
import os
import json
import numpy as np
try:
    import scipy.stats
except ImportError:
    scipy = None

# Define the model classes for different saved models
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class AmharicTransformerModel(nn.Module):
    def __init__(self, vocab_size=8000, hidden_size=256, num_layers=2, num_heads=4, max_length=128, dim_feedforward=None, has_final_layer_norm=False):
        super(AmharicTransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.has_final_layer_norm = has_final_layer_norm
        
        # If dim_feedforward is not specified, try to infer it from typical sizes
        if dim_feedforward is None:
            dim_feedforward = hidden_size * 2  # Default to 2x hidden size for smaller models
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Optional final layer normalization
        if has_final_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        # This is a simplified forward pass for visualization purposes
        seq_len = x.size(1) if len(x.shape) > 1 else x.size(0)
        positions = torch.arange(seq_len).unsqueeze(0)
        
        x = self.embedding(x) + self.position_embedding(positions)
        x = self.transformer(x)
        
        if self.has_final_layer_norm:
            x = self.layer_norm(x)
            
        x = self.output_projection(x)
        return x

# Set Graphviz path if not in system PATH
graphviz_path = r"C:\Program Files\Graphviz\bin"
os.environ['PATH'] = graphviz_path + os.pathsep + os.environ.get('PATH', '')

app = Flask(__name__)
# Ensure static images directory exists
os.makedirs(os.path.join(app.root_path, 'static', 'images'), exist_ok=True)

MODEL_PATH = os.path.join(os.getcwd(), 'model_epoch10.pt')
STATE_DICT_PATH = os.path.join(os.getcwd(), 'model_state_dict.pt')

PT_FILES_DIR = os.getcwd() # Directory where .pt files are located

# Function to list .pt files
def get_pt_files():
    return [f for f in os.listdir(PT_FILES_DIR) if f.endswith('.pt')]

# Function to intelligently load different model types
def smart_load_model(model_path):
    """
    Intelligently load different types of PyTorch models.
    Returns: (model, model_name_suffix, error_message)
    """
    try:
        # First, try to load and inspect the file
        loaded_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Case 1: It's already a complete nn.Module
        if isinstance(loaded_data, nn.Module):
            return loaded_data, " (Full Model)", None
              # Case 2: It's a dictionary - check its structure
        elif isinstance(loaded_data, dict):
            # Check if it has nested model structure (like Amharic model)
            if 'model_state_dict' in loaded_data:
                state_dict = loaded_data['model_state_dict']
                config = loaded_data.get('config', {})
                vocab_size = loaded_data.get('vocab_size', 8000)
                
                # Detect if it's a transformer model
                if any('transformer' in key for key in state_dict.keys()) or any('layers.' in key and 'self_attn' in key for key in state_dict.keys()):
                    # Analyze dimensions from the saved weights
                    hidden_size = config.get('hidden_size', 256)
                    
                    # Detect max_length from position embedding
                    max_length = 128  # default
                    if 'position_embedding.weight' in state_dict:
                        max_length = state_dict['position_embedding.weight'].shape[0]
                    
                    # Check if there's a final layer norm
                    has_final_layer_norm = 'layer_norm.weight' in state_dict
                    
                    # Try to detect feedforward dimension from the model weights
                    # Handle both naming conventions: transformer.layers.0 and layers.0
                    linear1_key = 'transformer.layers.0.linear1.weight' if 'transformer.layers.0.linear1.weight' in state_dict else 'layers.0.linear1.weight'
                    if linear1_key in state_dict:
                        dim_feedforward = state_dict[linear1_key].shape[0]
                    else:
                        dim_feedforward = hidden_size * 2  # fallback
                    
                    # Count number of layers
                    num_layers = 0
                    layer_prefix = 'transformer.layers.' if 'transformer.layers.0.self_attn.in_proj_weight' in state_dict else 'layers.'
                    for key in state_dict.keys():
                        if key.startswith(layer_prefix) and 'self_attn.in_proj_weight' in key:
                            # Extract layer number properly
                            if 'transformer' in key:
                                # Format: transformer.layers.X.self_attn.in_proj_weight
                                layer_num = int(key.split('.')[2])
                            else:
                                # Format: layers.X.self_attn.in_proj_weight
                                layer_num = int(key.split('.')[1])
                            num_layers = max(num_layers, layer_num + 1)
                    
                    if num_layers == 0:
                        num_layers = config.get('num_layers', 2)
                    
                    model = AmharicTransformerModel(
                        vocab_size=vocab_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        num_heads=config.get('num_heads', 4),
                        max_length=max_length,
                        dim_feedforward=dim_feedforward,
                        has_final_layer_norm=has_final_layer_norm
                    )
                    
                    # If the model uses different naming convention, we need to map the keys
                    if 'layers.0.self_attn.in_proj_weight' in state_dict and 'transformer.layers.0.self_attn.in_proj_weight' not in state_dict:
                        # This model uses 'layers.X' instead of 'transformer.layers.X'
                        # We need to adjust the state_dict keys to match our model structure
                        new_state_dict = {}
                        for key, value in state_dict.items():
                            if key.startswith('layers.'):
                                new_key = 'transformer.' + key
                                new_state_dict[new_key] = value
                            else:
                                new_state_dict[key] = value
                        state_dict = new_state_dict
                    
                    model.load_state_dict(state_dict)
                    return model, f" (Transformer Model, {num_layers} layers, vocab={vocab_size}, ff_dim={dim_feedforward})", None
                else:
                    # Try to load as SimpleNN
                    model = SimpleNN()
                    model.load_state_dict(state_dict)
                    return model, " (Simple NN from nested state_dict)", None
                    
            # Case 3: Direct state_dict
            else:
                # Check if it matches SimpleNN structure
                simple_nn_keys = {'layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias', 'layer3.weight', 'layer3.bias'}
                if simple_nn_keys.issubset(set(loaded_data.keys())):
                    model = SimpleNN()
                    model.load_state_dict(loaded_data)
                    return model, " (Simple NN from state_dict)", None
                else:
                    # Return the state_dict directly for visualization
                    return loaded_data, " (State Dict - Direct Visualization)", None
        
        # Case 4: Unknown format
        else:
            return None, "", f"Unknown model format: {type(loaded_data)}"
            
    except Exception as e:
        return None, "", f"Error loading model: {str(e)}"

# Function to visualize the model
def visualize_model(model):
    # If model is a state_dict (dict of parameters), visualize hierarchical keys
    if isinstance(model, dict):
        return visualize_state_dict(model) # Delegate to state_dict visualization
          # Otherwise assume nn.Module
    dot = Digraph(comment='Neural Network', 
                  graph_attr={
                      'rankdir': 'TB', 
                      'splines': 'ortho', 
                      'nodesep': '0.8', 
                      'ranksep': '1.2',
                      'fontname': 'Arial, sans-serif',
                      'fontsize': '14',
                      'bgcolor': 'white',
                      'pad': '0.5',
                      'margin': '0.3'
                  },
                  node_attr={
                      'fontname': 'Arial, sans-serif', 
                      'fontsize': '11',
                      'margin': '0.1,0.05'
                  },
                  edge_attr={
                      'fontname': 'Arial, sans-serif', 
                      'fontsize': '9',
                      'penwidth': '1.5'
                  }
                  )

    module_nodes = {} # To keep track of created module nodes and their original names    # Add nodes for modules
    for name, module in model.named_modules():
        node_id = name if name else 'model'
        module_type = module.__class__.__name__
        
        # Create more descriptive labels
        if not name:  # Root model
            label = f'Model\\n({module_type})'
        else:
            label = f'{name}\\n({module_type})'
        
        # Add layer-specific information for common layers
        if isinstance(module, nn.Linear):
            label += f'\\n{module.in_features} → {module.out_features}'
        elif isinstance(module, nn.Conv2d):
            label += f'\\n{module.in_channels}→{module.out_channels}\\nkernel: {module.kernel_size}'
        elif isinstance(module, nn.Dropout):
            label += f'\\np={module.p}'
        
        # Enhanced styling based on module type
        if not name:  # Root model
            shape = 'doubleoctagon'
            fillcolor = '#4CAF50'  # Green
            style = 'filled'
            fontcolor = 'white'
            penwidth = '3'
        elif isinstance(module, nn.Linear):
            shape = 'box'
            fillcolor = '#2196F3'  # Blue
            style = 'filled,rounded'
            fontcolor = 'white'
            penwidth = '2'
        elif isinstance(module, nn.Conv2d):
            shape = 'box'
            fillcolor = '#9C27B0'  # Purple
            style = 'filled,rounded'
            fontcolor = 'white'
            penwidth = '2'
        elif isinstance(module, nn.ReLU):
            shape = 'ellipse'
            fillcolor = '#FF9800'  # Orange
            style = 'filled'
            fontcolor = 'black'
            penwidth = '2'
        elif isinstance(module, nn.Dropout):
            shape = 'diamond'
            fillcolor = '#F44336'  # Red
            style = 'filled'
            fontcolor = 'white'
            penwidth = '2'
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            shape = 'hexagon'
            fillcolor = '#607D8B'  # Blue Grey
            style = 'filled'
            fontcolor = 'white'
            penwidth = '2'
        else:
            shape = 'box'
            fillcolor = '#9E9E9E'  # Grey
            style = 'filled,rounded'
            fontcolor = 'black'
            penwidth = '1'
        
        dot.node(node_id, label=label, shape=shape, style=style, 
                fillcolor=fillcolor, fontcolor=fontcolor, color='black', penwidth=penwidth)
        module_nodes[node_id] = node_id

        # Add edges between parent and child modules
        if '.' in name: 
            parent_name = name.rsplit('.', 1)[0]
            if parent_name in module_nodes: 
                dot.edge(module_nodes[parent_name], node_id, style='solid', color='#424242', penwidth='2')
            else: 
                dot.edge('model', node_id, style='solid', color='#424242', penwidth='2')
        elif name and node_id != 'model': 
             if 'model' in module_nodes: 
                dot.edge(module_nodes['model'], node_id, style='solid', color='#424242', penwidth='2')
    # Add nodes for parameters and connect them to their respective modules
    for param_name, param in model.named_parameters():
        parts = param_name.split('.')
        actual_param_name = parts[-1] 
        
        if len(parts) > 1:
            module_path_for_param = '.'.join(parts[:-1])
        else:
            module_path_for_param = 'model' 

        # Create more concise parameter labels
        param_shape = list(param.shape)
        param_size = param.numel()
        
        # Show fewer values and format better
        weights_flat = param.data.detach().cpu().flatten() 
        if weights_flat.numel() > 2:
            weights_str = f"[{weights_flat[0]:.3f}, {weights_flat[1]:.3f}, ...]"
        else:
            weights_str = f"[{', '.join([f'{x:.3f}' for x in weights_flat.tolist()])}]"
        
        param_label = f"{actual_param_name}\\nShape: {param_shape}\\nSize: {param_size:,}\\nSample: {weights_str}"
        
        param_node_name = f"param_{param_name.replace('.', '_')}"
        
        # Style parameters based on type
        if 'weight' in actual_param_name.lower():
            fillcolor = '#E3F2FD'  # Light blue
            fontcolor = '#1565C0'  # Dark blue
        elif 'bias' in actual_param_name.lower():
            fillcolor = '#FFF3E0'  # Light orange
            fontcolor = '#EF6C00'  # Dark orange
        else:
            fillcolor = '#F3E5F5'  # Light purple
            fontcolor = '#7B1FA2'  # Dark purple
        
        dot.node(param_node_name, label=param_label, shape='note', style='filled', 
                fillcolor=fillcolor, fontcolor=fontcolor, color='#616161', penwidth='1')
        
        target_module_node_id = module_nodes.get(module_path_for_param)

        if target_module_node_id:
            dot.edge(target_module_node_id, param_node_name, style='dashed', 
                    arrowhead='none', color='#757575', penwidth='1')
        elif module_path_for_param == 'model' and module_nodes.get('model'): 
             dot.edge(module_nodes['model'], param_node_name, style='dashed', 
                     arrowhead='none', color='#757575', penwidth='1')
        else:
            print(f"Warning: Module path '{module_path_for_param}' for param '{param_name}' not found. Connecting to root 'model'.")
            if module_nodes.get('model'):
                dot.edge(module_nodes['model'], param_node_name, style='dashed', 
                        arrowhead='none', color='#757575', penwidth='1')
            
    return dot

def visualize_state_dict(state_dict):
    dot = Digraph(comment='Neural Network (from state_dict)', 
                  graph_attr={
                      'rankdir': 'TB', 
                      'splines': 'ortho', 
                      'nodesep': '0.8', 
                      'ranksep': '1.2',
                      'fontname': 'Arial, sans-serif',
                      'fontsize': '14',
                      'bgcolor': 'white',
                      'pad': '0.5',
                      'margin': '0.3'
                  },
                  node_attr={
                      'fontname': 'Arial, sans-serif', 
                      'fontsize': '11',
                      'margin': '0.1,0.05'
                  },
                  edge_attr={
                      'fontname': 'Arial, sans-serif', 
                      'fontsize': '9',
                      'penwidth': '1.5'
                  }
                  )
    
    module_params = {} 
    all_module_paths = set() 

    has_root_params = any('.' not in k for k in state_dict.keys())
    if has_root_params or not any('.' in k for k in state_dict.keys()):
        all_module_paths.add('model') # Ensure 'model' is treated as the root/default module path

    for key, value in state_dict.items():
        parts = key.split('.')
        param_name = parts[-1]
        
        current_module_path_parts = []
        if len(parts) > 1:
            for i in range(len(parts) - 1):
                current_module_path_parts.append(parts[i])
                all_module_paths.add('.'.join(current_module_path_parts))
            module_path = '.'.join(parts[:-1])
        else:
            module_path = 'model'
        
        if module_path not in module_params:
            module_params[module_path] = []
        
        weights_flat = value.cpu().flatten() 
        param_size = value.numel()
        
        # Better formatting for weights display
        if value.numel() > 2:
            weights_str = f"[{weights_flat[0]:.3f}, {weights_flat[1]:.3f}, ...]"
        else:
            weights_str = f"[{', '.join([f'{x:.3f}' for x in weights_flat.tolist()])}]"
        
        module_params[module_path].append({
            'name': param_name, 
            'shape': list(value.shape), 
            'size': param_size,
            'weights': weights_str,
            'full_key': key 
        })

    sorted_module_paths = sorted(list(all_module_paths), key=lambda x: (x.count('.'), x))
    created_nodes = set()

    for module_path in sorted_module_paths:
        if module_path not in created_nodes:
            label = f"{module_path}\\n(Module)"
            
            # Enhanced styling for modules
            if module_path == 'model':
                shape = 'doubleoctagon'
                fillcolor = '#4CAF50'  # Green for root
                style = 'filled'
                fontcolor = 'white'
                penwidth = '3'
            elif 'layer' in module_path.lower():
                shape = 'box'
                fillcolor = '#2196F3'  # Blue for layers
                style = 'filled,rounded'
                fontcolor = 'white'
                penwidth = '2'
            else:
                shape = 'box'
                fillcolor = '#FF9800'  # Orange for other modules
                style = 'filled,rounded'
                fontcolor = 'black'
                penwidth = '2'

            dot.node(module_path, label=label, shape=shape, style=style, 
                    fillcolor=fillcolor, fontcolor=fontcolor, color='black', penwidth=penwidth)
            created_nodes.add(module_path)

        if '.' in module_path:
            parent_path = module_path.rsplit('.', 1)[0]
            if parent_path in created_nodes: 
                 dot.edge(parent_path, module_path, style='solid', color='#424242', penwidth='2')
        elif module_path != 'model' and 'model' in created_nodes:
            is_param_key_at_root = module_path in state_dict 
            if not is_param_key_at_root:
                 dot.edge('model', module_path, style='solid', color='#424242', penwidth='2')


    for module_path, params_list in module_params.items():
        if module_path not in created_nodes:
            label = f"{module_path}\\n(Module)"
            dot.node(module_path, label=label, shape='box', style='filled,rounded', fillcolor='#FFEB3B', color='black')
            created_nodes.add(module_path)
            if '.' in module_path:
                parent_path = module_path.rsplit('.', 1)[0]
                if parent_path in created_nodes:
                    dot.edge(parent_path, module_path, style='solid', color='dimgray')

        for p_info in params_list:
            param_label = f"{p_info['name']}\\nShape: {p_info['shape']}\\nData: {p_info['weights']}" # Shortened
            param_node_name = f"param_{p_info['full_key'].replace('.', '_')}"
            dot.node(param_node_name, label=param_label, shape='note', style='filled', fillcolor='#FFF9C4', color='darkgoldenrod')
            dot.edge(module_path, param_node_name, style='dashed', arrowhead='none', color='gray50')
            
    return dot

# Function to extract 3D visualization data from model
def extract_3d_model_data(model):
    """
    Extract data structure for 3D visualization
    Returns: dict with nodes, edges, and metadata
    """
    nodes = []
    edges = []
    layers = []
    
    if isinstance(model, dict):
        # Handle state_dict visualization
        return extract_3d_from_state_dict(model)
    
    # Process nn.Module
    layer_positions = {}
    layer_count = 0
    
    for name, module in model.named_modules():
        node_id = name if name else 'model'
        module_type = module.__class__.__name__
        
        # Calculate layer depth
        depth = name.count('.') if name else 0
        
        # Determine layer properties
        if isinstance(module, nn.Linear):
            layer_type = 'linear'
            input_size = module.in_features
            output_size = module.out_features
            color = '#2196F3'
        elif isinstance(module, nn.Conv2d):
            layer_type = 'conv2d'
            input_size = module.in_channels
            output_size = module.out_channels
            color = '#9C27B0'
        elif isinstance(module, nn.ReLU):
            layer_type = 'activation'
            input_size = 0
            output_size = 0
            color = '#FF9800'
        elif isinstance(module, nn.Dropout):
            layer_type = 'dropout'
            input_size = 0
            output_size = 0
            color = '#F44336'
        elif isinstance(module, nn.TransformerEncoder):
            layer_type = 'transformer'
            input_size = getattr(module, 'num_layers', 0)
            output_size = getattr(module, 'num_layers', 0)
            color = '#4CAF50'
        else:
            layer_type = 'module'
            input_size = 0
            output_size = 0
            color = '#9E9E9E'
        
        # Position calculation for 3D space
        x = (layer_count % 5) * 150 - 300  # Spread across X
        y = depth * -100  # Depth in Y
        z = (layer_count // 5) * 100 - 200  # Layers in Z
        
        node = {
            'id': node_id,
            'label': f"{name or 'Model'}\\n({module_type})",
            'type': layer_type,
            'depth': depth,
            'input_size': input_size,
            'output_size': output_size,
            'color': color,
            'position': {'x': x, 'y': y, 'z': z},
            'module_type': module_type
        }
        
        nodes.append(node)
        layer_positions[node_id] = layer_count
        layer_count += 1
        
        # Create edges for hierarchical connections
        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            if parent_name in [n['id'] for n in nodes[:-1]]:
                edges.append({
                    'source': parent_name,
                    'target': node_id,
                    'type': 'hierarchical'
                })
        elif name and node_id != 'model':
            edges.append({
                'source': 'model',
                'target': node_id,
                'type': 'hierarchical'
            })
    
    # Add parameter information
    param_count = 0
    for param_name, param in model.named_parameters() if hasattr(model, 'named_parameters') else []:
        param_count += param.numel()
    
    return {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'total_parameters': param_count,
            'num_layers': len(nodes),
            'model_type': type(model).__name__
        }
    }

def extract_3d_from_state_dict(state_dict):
    """Extract 3D data from state dictionary"""
    nodes = []
    edges = []
    
    # Group parameters by module
    modules = {}
    for key, tensor in state_dict.items():
        parts = key.split('.')
        if len(parts) > 1:
            module_path = '.'.join(parts[:-1])
            param_name = parts[-1]
        else:
            module_path = 'model'
            param_name = key
            
        if module_path not in modules:
            modules[module_path] = []
        
        modules[module_path].append({
            'name': param_name,
            'shape': list(tensor.shape),
            'size': tensor.numel()
        })
    
    # Create nodes for modules
    layer_count = 0
    for module_path, params in modules.items():
        depth = module_path.count('.') if module_path != 'model' else 0
        
        # Calculate total parameters in this module
        total_params = sum(p['size'] for p in params)
        
        # Position in 3D space
        x = (layer_count % 4) * 200 - 300
        y = depth * -120
        z = (layer_count // 4) * 150 - 200
        
        node = {
            'id': module_path,
            'label': module_path,
            'type': 'module',
            'depth': depth,
            'parameter_count': total_params,
            'parameters': params,
            'color': '#2196F3' if 'layer' in module_path.lower() else '#4CAF50',
            'position': {'x': x, 'y': y, 'z': z}
        }
        
        nodes.append(node)
        layer_count += 1
        
        # Create hierarchical edges
        if '.' in module_path:
            parent_path = module_path.rsplit('.', 1)[0]
            if parent_path in [n['id'] for n in nodes[:-1]]:
                edges.append({
                    'source': parent_path,
                    'target': module_path,
                    'type': 'hierarchical'
                })
        elif module_path != 'model':
            edges.append({
                'source': 'model',
                'target': module_path,
                'type': 'hierarchical'
            })
    
    total_params = sum(tensor.numel() for tensor in state_dict.values())
    
    return {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'total_parameters': total_params,
            'num_modules': len(nodes),
            'model_type': 'StateDict'
        }
    }

@app.route('/api/model_data_3d')
def get_model_data_3d():
    """API endpoint to get 3D model data as JSON"""
    from flask import request, jsonify
    selected_file = request.args.get('model_file')
    
    if not selected_file:
        return jsonify({'error': 'No model file specified'}), 400
    
    model_path = os.path.join(PT_FILES_DIR, selected_file)
    
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model file {selected_file} not found'}), 404
    
    try:
        # Load the model using our smart loading function
        model, model_name_suffix, load_error = smart_load_model(model_path)
        
        if load_error:
            return jsonify({'error': load_error}), 500
        
        # Extract 3D visualization data
        model_data = extract_3d_model_data(model)
        model_data['model_name'] = selected_file + model_name_suffix
        
        return jsonify(model_data)
        
    except Exception as e:
        return jsonify({'error': f'Error processing model: {str(e)}'}), 500

@app.route('/visualize_3d')
def visualize_3d():
    """Route for 3D visualization page"""
    model_files = get_pt_files()
    return render_template('visualize_3d.html', model_files=model_files)

@app.route('/visualize_hierarchical')
def visualize_hierarchical():
    """Route for hierarchical tree visualization"""
    model_files = get_pt_files()
    return render_template('visualize_hierarchical.html', model_files=model_files)

@app.route('/visualize_force')
def visualize_force():
    """Route for force-directed network visualization"""
    model_files = get_pt_files()
    return render_template('visualize_force.html', model_files=model_files)

@app.route('/visualize_interactive')
def visualize_interactive():
    """Route for interactive inspection view"""
    model_files = get_pt_files()
    return render_template('visualize_interactive.html', model_files=model_files)

@app.route('/')
def index():
    """Analytics dashboard as home page"""
    try:
        model_files = get_pt_files()
        return render_template('analytics_dashboard.html', model_files=model_files)
    except Exception as e:
        return f"Error loading analytics dashboard: {str(e)}", 500

@app.route('/visualize_selected', methods=['GET'])
def visualize_selected():
    from flask import request
    selected_file = request.args.get('model_file')
    model_files = get_pt_files()
    
    if not selected_file or selected_file not in model_files:
        return render_template('index.html', show_options=True, model_files=model_files, error="Please select a valid model file.", selected_model_file=None)

    model_path = os.path.join(PT_FILES_DIR, selected_file)
    image_url = None
    error_message = None
    model_name_suffix = ""
    load_error = None

    if os.path.exists(model_path):
        try:
            # Use the smart loading function
            model, model_name_suffix, load_error = smart_load_model(model_path)
            
            if load_error:
                error_message = f"Error loading {selected_file}: {load_error}"
            else:
                model_name = selected_file + model_name_suffix
                
                # Generate visualization
                dot = visualize_model(model)
                
                # Sanitize filename from selected_file
                base_filename = os.path.splitext(selected_file)[0].replace('.', '_').replace(' ', '_')
                viz_filename = f'{base_filename}.gv'
                
                try:
                    dot.render(filename=viz_filename, directory=os.path.join(app.root_path, 'static', 'images'), format='png', cleanup=True)
                    image_url = url_for('static', filename=f'images/{viz_filename}.png')
                except ExecutableNotFound:
                    error_message = "Graphviz 'dot' executable not found. Please install Graphviz and add it to your PATH."
                except Exception as e_render:
                    error_message = f"Error rendering graph: {str(e_render)}"

        except Exception as e_load:
            error_message = f"Error processing {selected_file}: {str(e_load)}"
    else:
        error_message = f"{selected_file} file not found."

    return render_template('index.html', 
                             image_url=image_url,
                             model_name=selected_file + (model_name_suffix if load_error is None and 'model_name_suffix' in locals() else ""),
                             show_options=True, 
                             model_files=model_files, 
                             selected_model_file=selected_file,
                             error=error_message)


@app.route('/export_svg')
def export_svg():
    from flask import request, send_file
    selected_file = request.args.get('model_file')
    model_files = get_pt_files()
    
    if not selected_file or selected_file not in model_files:
        return "Invalid model file", 400

    model_path = os.path.join(PT_FILES_DIR, selected_file)
    
    if os.path.exists(model_path):
        try:
            # Use the smart loading function
            model, model_name_suffix, load_error = smart_load_model(model_path)
            
            if load_error:
                return f"Error loading {selected_file}: {load_error}", 500
            else:
                # Generate visualization
                dot = visualize_model(model)
                
                # Sanitize filename from selected_file
                base_filename = os.path.splitext(selected_file)[0].replace('.', '_').replace(' ', '_')
                viz_filename = f'{base_filename}.gv'
                
                try:
                    # Render as SVG
                    svg_path = dot.render(filename=viz_filename, directory=os.path.join(app.root_path, 'static', 'images'), format='svg', cleanup=True)
                    return send_file(svg_path, as_attachment=True, download_name=f'{base_filename}.svg')
                except Exception as e_render:
                    return f"Error rendering SVG: {str(e_render)}", 500

        except Exception as e_load:
            return f"Error processing {selected_file}: {str(e_load)}", 500
    else:
        return f"{selected_file} file not found.", 404

@app.route('/visualize/<model_type>')
def visualize(model_type):
    if model_type == 'full':
        model_path = MODEL_PATH
        model_name = 'Full Model'
    elif model_type == 'state_dict':
        model_path = STATE_DICT_PATH
        model_name = 'State Dict'
    else:
        return "Invalid model type", 400
    
    if os.path.exists(model_path):
        try:
            if model_type == 'full':
                model = torch.load(model_path, weights_only=False)
            else:
                # Load state dict and create model instance
                state_dict = torch.load(model_path, weights_only=True)
                model = SimpleNN()
                model.load_state_dict(state_dict)
            
            # Generate visualization
            dot = visualize_model(model)
            filename = f'model_{model_type}.gv'
            
            # Render the graph to a PNG image
            try:
                dot.render(filename=filename, directory=os.path.join(app.root_path, 'static', 'images'), format='png', cleanup=True)
            except ExecutableNotFound:
                return ("Graphviz 'dot' executable not found. "
                        "Please install Graphviz and add it to your PATH."), 500
            
            return render_template('index.html', 
                                 image_url=url_for('static', filename=f'images/{filename}.png'),
                                 model_name=model_name,
                                 show_options=True)
        except Exception as e:
            return f"Error loading {model_name}: {str(e)}", 500
    else:
        return f"{model_name} file not found.", 404

@app.route('/api/model_structure')
def get_model_structure():
    """API endpoint to get detailed model structure for interactive inspection"""
    selected_file = request.args.get('model_file')
    
    if not selected_file:
        return jsonify({'error': 'No model file specified'}), 400
    
    model_path = os.path.join(PT_FILES_DIR, selected_file)
    
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model file {selected_file} not found'}), 404
    
    try:
        # Load the model using our smart loading function
        model, model_name_suffix, load_error = smart_load_model(model_path)
        
        if load_error:
            return jsonify({'error': load_error}), 500
        
        # Extract detailed structure for interactive inspection
        structure_data = extract_detailed_structure(model)
        structure_data['model_name'] = selected_file + model_name_suffix
        
        return jsonify(structure_data)
        
    except Exception as e:
        return jsonify({'error': f'Error processing model: {str(e)}'}), 500

def extract_detailed_structure(model):
    """Extract detailed model structure with parameter inspection capabilities"""
    structure = {
        'modules': [],
        'parameters': [],
        'architecture': {},
        'statistics': {}
    }
    
    if isinstance(model, dict):
        # Handle state_dict
        total_params = 0
        param_groups = {}
        
        for param_name, param_tensor in model.items():
            total_params += param_tensor.numel()
            
            # Group parameters by module
            module_path = '.'.join(param_name.split('.')[:-1]) if '.' in param_name else 'root'
            if module_path not in param_groups:
                param_groups[module_path] = []
            
            param_info = {
                'name': param_name,
                'shape': list(param_tensor.shape),
                'size': param_tensor.numel(),
                'dtype': str(param_tensor.dtype),
                'requires_grad': param_tensor.requires_grad if hasattr(param_tensor, 'requires_grad') else False,
                'min_value': float(param_tensor.min()) if param_tensor.numel() > 0 else 0,
                'max_value': float(param_tensor.max()) if param_tensor.numel() > 0 else 0,
                'mean_value': float(param_tensor.mean()) if param_tensor.numel() > 0 else 0,
                'std_value': float(param_tensor.std()) if param_tensor.numel() > 0 else 0
            }
            
            param_groups[module_path].append(param_info)
            structure['parameters'].append(param_info)
        
        # Create module structure from parameter groups
        for module_path, params in param_groups.items():
            module_info = {
                'name': module_path,
                'type': 'StateDict',
                'parameters': params,
                'total_params': sum(p['size'] for p in params)
            }
            structure['modules'].append(module_info)
        
        structure['statistics'] = {
            'total_parameters': total_params,
            'num_modules': len(param_groups),
            'model_type': 'StateDict'
        }
        
    else:
        # Handle nn.Module
        total_params = 0
        
        # Extract module hierarchy
        for name, module in model.named_modules():
            module_type = module.__class__.__name__
            module_params = []
            module_param_count = 0
            
            # Get parameters for this specific module (not children)
            for param_name, param in module.named_parameters(recurse=False):
                total_params += param.numel()
                module_param_count += param.numel()
                
                param_info = {
                    'name': param_name,
                    'shape': list(param.shape),
                    'size': param.numel(),
                    'dtype': str(param.dtype),
                    'requires_grad': param.requires_grad,
                    'min_value': float(param.min()) if param.numel() > 0 else 0,
                    'max_value': float(param.max()) if param.numel() > 0 else 0,
                    'mean_value': float(param.mean()) if param.numel() > 0 else 0,
                    'std_value': float(param.std()) if param.numel() > 0 else 0
                }
                
                module_params.append(param_info)
                structure['parameters'].append(param_info)
            
            # Extract module-specific information
            module_info = {
                'name': name or 'model',
                'type': module_type,
                'parameters': module_params,
                'total_params': module_param_count,
                'depth': name.count('.') if name else 0
            }
            
            # Add layer-specific details
            if isinstance(module, nn.Linear):
                module_info['details'] = {
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'bias': module.bias is not None
                }
            elif isinstance(module, nn.Conv2d):
                module_info['details'] = {
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding
                }
            elif isinstance(module, nn.Dropout):
                module_info['details'] = {
                    'p': module.p,
                    'inplace': module.inplace
                }
            
            structure['modules'].append(module_info)
        
        structure['statistics'] = {
            'total_parameters': total_params,
            'num_modules': len(list(model.named_modules())),
            'model_type': type(model).__name__
        }
          # Extract architecture summary
        structure['architecture'] = {
            'input_layers': [],
            'hidden_layers': [],
            'output_layers': [],
            'activation_functions': [],
            'regularization': []
        }
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if 'input' in name.lower() or name.count('.') == 0:
                    structure['architecture']['input_layers'].append(name)
                elif 'output' in name.lower() or 'classifier' in name.lower():
                    structure['architecture']['output_layers'].append(name)
                else:
                    structure['architecture']['hidden_layers'].append(name)
            elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU)):
                structure['architecture']['activation_functions'].append(name)
            elif isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d)):
                structure['architecture']['regularization'].append(name)
    
    return structure

# Comprehensive Neural Network Analysis Helper Functions

def calculate_model_health(model):
    """Calculate comprehensive model health score"""
    health_metrics = {
        'overall_score': 0,
        'weight_health': 0,
        'architecture_health': 0,
        'parameter_efficiency': 0,
        'initialization_quality': 0,
        'issues': [],
        'recommendations': []
    }
    
    try:
        # Weight health analysis
        weight_stats = []
        dead_neurons = 0
        total_neurons = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weights = param.data.cpu().numpy().flatten()
                
                # Check for dead weights (all zeros)
                if np.all(weights == 0):
                    health_metrics['issues'].append(f"Dead weights detected in {name}")
                
                # Check for exploding weights
                if np.max(np.abs(weights)) > 10:
                    health_metrics['issues'].append(f"Potential exploding weights in {name}")
                
                # Check for vanishing weights
                if np.max(np.abs(weights)) < 1e-6:
                    health_metrics['issues'].append(f"Potential vanishing weights in {name}")
                
                weight_stats.append({
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights)
                })
        
        # Calculate weight health score
        if weight_stats:
            avg_std = np.mean([w['std'] for w in weight_stats])
            weight_health = min(100, max(0, 100 - len(health_metrics['issues']) * 20))
            health_metrics['weight_health'] = weight_health
        
        # Architecture health
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Parameter efficiency
        param_efficiency = (trainable_params / total_params) * 100 if total_params > 0 else 0
        health_metrics['parameter_efficiency'] = param_efficiency
        
        # Architecture complexity analysis
        depth = len([m for m in model.modules() if len(list(m.children())) == 0])
        if depth > 50:
            health_metrics['issues'].append("Very deep architecture may cause gradient issues")
        
        # Overall score calculation
        health_metrics['overall_score'] = (
            health_metrics['weight_health'] * 0.4 +
            param_efficiency * 0.3 +
            min(100, 100 - len(health_metrics['issues']) * 10) * 0.3
        )
        
        # Generate recommendations
        if health_metrics['overall_score'] < 70:
            health_metrics['recommendations'].append("Consider model optimization")
        if len(health_metrics['issues']) > 3:
            health_metrics['recommendations'].append("Multiple issues detected - review architecture")
        if param_efficiency < 80:
            health_metrics['recommendations'].append("Consider parameter pruning")
        
    except Exception as e:
        health_metrics['error'] = str(e)
    
    return health_metrics

def analyze_layer_insights(model):
    """Deep layer-by-layer analysis"""
    insights = {
        'layer_details': [],
        'bottlenecks': [],
        'redundancies': [],
        'recommendations': []
    }
    
    try:
        layer_sizes = []
        activation_types = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                layer_info = {
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                    'input_size': None,
                    'output_size': None,
                    'efficiency_score': 0,
                    'bottleneck_risk': False
                }
                
                # Get layer dimensions
                if hasattr(module, 'in_features'):
                    layer_info['input_size'] = module.in_features
                    layer_info['output_size'] = module.out_features
                    layer_sizes.append(module.out_features)
                    
                    # Check for bottlenecks
                    if module.out_features < module.in_features * 0.1:
                        layer_info['bottleneck_risk'] = True
                        insights['bottlenecks'].append(name)
                
                elif hasattr(module, 'in_channels'):
                    layer_info['input_size'] = module.in_channels
                    layer_info['output_size'] = module.out_channels
                
                # Track activation functions
                activation_types[type(module).__name__] = activation_types.get(type(module).__name__, 0) + 1
                
                # Calculate efficiency score
                if layer_info['parameters'] > 0:
                    param_ratio = layer_info['parameters'] / sum(p.numel() for p in model.parameters())
                    layer_info['efficiency_score'] = min(100, (1 - param_ratio) * 100)
                
                insights['layer_details'].append(layer_info)
        
        # Analyze patterns
        if len(set(layer_sizes)) < len(layer_sizes) * 0.5:
            insights['redundancies'].append("Similar layer sizes detected - consider optimization")
        
        # Generate recommendations
        if len(insights['bottlenecks']) > 0:
            insights['recommendations'].append("Bottleneck layers detected - may limit model capacity")
        
        if activation_types.get('ReLU', 0) == 0 and activation_types.get('LeakyReLU', 0) == 0:
            insights['recommendations'].append("Consider adding ReLU activations for better training")
        
    except Exception as e:
        insights['error'] = str(e)
    
    return insights

def analyze_training_state(model):
    """Analyze training state and patterns"""
    training_insights = {
        'initialization_analysis': {},
        'gradient_readiness': {},
        'training_recommendations': [],
        'potential_issues': []
    }
    
    try:
        # Analyze weight initialization patterns
        weight_distributions = []
        bias_distributions = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                data = param.data.cpu().numpy().flatten()
                
                if 'weight' in name:
                    weight_distributions.append({
                        'name': name,
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data)),
                        'zero_fraction': float(np.mean(data == 0))
                    })
                elif 'bias' in name:
                    bias_distributions.append({
                        'name': name,
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data))
                    })
        
        training_insights['initialization_analysis'] = {
            'weight_distributions': weight_distributions,
            'bias_distributions': bias_distributions,
            'initialization_health': 'good' if len(weight_distributions) > 0 else 'unknown'
        }
        
        # Check for common training issues
        for wd in weight_distributions:
            if wd['std'] > 2.0:
                training_insights['potential_issues'].append(f"High weight variance in {wd['name']}")
            if wd['zero_fraction'] > 0.1:
                training_insights['potential_issues'].append(f"Many zero weights in {wd['name']}")
        
        # Generate training recommendations
        if len(weight_distributions) > 10:
            training_insights['training_recommendations'].append("Consider using learning rate scheduling")
        
        if any(wd['std'] < 0.01 for wd in weight_distributions):
            training_insights['training_recommendations'].append("Some layers may need better initialization")
        
    except Exception as e:
        training_insights['error'] = str(e)
    
    return training_insights

def analyze_architecture_patterns(model):
    """Analyze architectural patterns and design"""
    patterns = {
        'architecture_type': 'unknown',
        'design_patterns': [],
        'layer_patterns': {},
        'connectivity_analysis': {},
        'architectural_score': 0
    }
    
    try:
        # Analyze layer types and patterns
        layer_types = {}
        layer_sequence = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                module_type = type(module).__name__
                layer_types[module_type] = layer_types.get(module_type, 0) + 1
                layer_sequence.append(module_type)
        
        patterns['layer_patterns'] = layer_types
        
        # Detect common architectures
        if 'TransformerEncoder' in layer_types or 'MultiheadAttention' in layer_types:
            patterns['architecture_type'] = 'transformer'
            patterns['design_patterns'].append('attention_mechanism')
        elif 'Conv2d' in layer_types:
            patterns['architecture_type'] = 'convolutional'
            if 'BatchNorm2d' in layer_types:
                patterns['design_patterns'].append('batch_normalization')
        elif 'Linear' in layer_types:
            patterns['architecture_type'] = 'feedforward'
        
        # Analyze regularization patterns
        if 'Dropout' in layer_types:
            patterns['design_patterns'].append('dropout_regularization')
        if any('BatchNorm' in lt for lt in layer_types):
            patterns['design_patterns'].append('batch_normalization')
        
        # Calculate architectural score
        score = 50  # Base score
        if len(patterns['design_patterns']) > 0:
            score += len(patterns['design_patterns']) * 10
        if patterns['architecture_type'] != 'unknown':
            score += 20
        
        patterns['architectural_score'] = min(100, score)
        
    except Exception as e:
        patterns['error'] = str(e)
    
    return patterns

def generate_optimization_report(model):
    """Generate comprehensive optimization recommendations"""
    report = {
        'optimization_opportunities': [],
        'performance_improvements': [],
        'memory_optimizations': [],
        'computational_optimizations': [],
        'priority_score': {},
        'estimated_benefits': {}
    }
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        
        # Memory optimization opportunities
        large_layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                param_count = module.weight.numel()
                if param_count > total_params * 0.1:  # Layer uses >10% of parameters
                    large_layers.append((name, param_count))
        
        if large_layers:
            report['memory_optimizations'].append({
                'type': 'parameter_pruning',
                'target_layers': [name for name, _ in large_layers],
                'potential_reduction': '20-40%',
                'complexity': 'medium'
            })
        
        # Computational optimizations
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers.append(name)
        
        if len(linear_layers) > 3:
            report['computational_optimizations'].append({
                'type': 'quantization',
                'description': 'Convert to INT8 quantization',
                'potential_speedup': '2-4x',
                'accuracy_impact': 'minimal'
            })
        
        # Performance improvements
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
        if not has_dropout:
            report['performance_improvements'].append({
                'type': 'regularization',
                'recommendation': 'Add dropout layers',
                'benefit': 'Improved generalization'
            })
        
        # Priority scoring
        report['priority_score'] = {
            'memory_optimization': 8 if large_layers else 3,
            'quantization': 7 if len(linear_layers) > 5 else 4,
            'pruning': 6 if total_params > 1000000 else 2
        }
        
    except Exception as e:
        report['error'] = str(e)
    
    return report

def perform_weight_analysis(model):
    """Advanced weight distribution and health analysis"""
    analysis = {
        'weight_statistics': {},
        'distribution_analysis': {},
        'health_indicators': {},
        'anomaly_detection': {}
    }
    
    try:
        all_weights = []
        layer_stats = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weights = param.data.cpu().numpy().flatten()
                all_weights.extend(weights)
                  # Per-layer statistics
                layer_stats[name] = {
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights)),
                    'min': float(np.min(weights)),
                    'max': float(np.max(weights)),
                    'sparsity': float(np.mean(weights == 0)),
                    'kurtosis': float(scipy.stats.kurtosis(weights)) if scipy and hasattr(scipy, 'stats') else 0,
                    'skewness': float(scipy.stats.skew(weights)) if scipy and hasattr(scipy, 'stats') else 0
                }
        
        analysis['weight_statistics'] = layer_stats
        
        # Overall distribution analysis
        if all_weights:
            all_weights = np.array(all_weights)
            analysis['distribution_analysis'] = {
                'global_mean': float(np.mean(all_weights)),
                'global_std': float(np.std(all_weights)),
                'global_sparsity': float(np.mean(all_weights == 0)),
                'outlier_percentage': float(np.mean(np.abs(all_weights) > 3 * np.std(all_weights)))
            }
        
        # Health indicators
        healthy_layers = sum(1 for stats in layer_stats.values() 
                           if 0.01 < stats['std'] < 2.0 and stats['sparsity'] < 0.5)
        total_layers = len(layer_stats)
        
        analysis['health_indicators'] = {
            'healthy_layer_ratio': healthy_layers / total_layers if total_layers > 0 else 0,
            'overall_health': 'good' if healthy_layers / total_layers > 0.8 else 'concerning'
        }
        
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis

def calculate_model_efficiency(model):
    """Calculate model efficiency metrics"""
    efficiency = {
        'parameter_efficiency': 0,
        'computational_efficiency': 0,
        'memory_efficiency': 0,
        'overall_efficiency': 0,
        'benchmarks': {}
    }
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Parameter efficiency
        efficiency['parameter_efficiency'] = (trainable_params / total_params * 100) if total_params > 0 else 0
        
        # Memory efficiency (simplified)
        param_memory = total_params * 4  # Assuming float32
        efficiency['memory_efficiency'] = min(100, 100 - (param_memory / (1024*1024)) / 100)  # Penalty for large models
        
        # Computational efficiency (layers vs parameters ratio)
        num_layers = len([m for m in model.modules() if len(list(m.children())) == 0])
        if num_layers > 0:
            params_per_layer = total_params / num_layers
            efficiency['computational_efficiency'] = min(100, 100 - np.log10(params_per_layer))
        
        # Overall efficiency
        efficiency['overall_efficiency'] = (
            efficiency['parameter_efficiency'] * 0.4 +
            efficiency['memory_efficiency'] * 0.3 +
            efficiency['computational_efficiency'] * 0.3
        )
        
        # Benchmark comparisons
        efficiency['benchmarks'] = {
            'parameter_count_category': 'small' if total_params < 1e6 else 'medium' if total_params < 1e8 else 'large',
            'efficiency_category': 'high' if efficiency['overall_efficiency'] > 80 else 'medium' if efficiency['overall_efficiency'] > 50 else 'low'
        }
        
    except Exception as e:
        efficiency['error'] = str(e)
    
    return efficiency

def get_basic_model_info(model):
    """Get basic model information"""
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_type': type(model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'num_layers': len(list(model.modules())),
            'memory_footprint_mb': (total_params * 4) / (1024 * 1024)  # Assuming float32
        }
    except Exception as e:
        return {'error': str(e)}

def get_optimization_priorities(model):
    """Get optimization priorities"""
    try:
        total_params = sum(p.numel() for p in model.parameters())
        priorities = []
        
        if total_params > 10e6:  # 10M+ parameters
            priorities.append({'type': 'pruning', 'priority': 'high', 'reason': 'Large parameter count'})
        
        # Check for quantization opportunities
        linear_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        if linear_count > 5:
            priorities.append({'type': 'quantization', 'priority': 'medium', 'reason': 'Many linear layers'})
        
        return priorities
    except Exception as e:
        return [{'error': str(e)}]

def analyze_layer_distribution(model):
    """Analyze layer type distribution"""
    try:
        layer_counts = {}
        for module in model.modules():
            if len(list(module.children())) == 0:  # Leaf modules
                layer_type = type(module).__name__
                layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        return {
            'distribution': layer_counts,
            'total_layers': sum(layer_counts.values()),
            'unique_types': len(layer_counts)
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_parameter_insights(model):
    """Analyze parameter insights"""
    try:
        param_insights = {
            'by_layer_type': {},
            'gradient_enabled': 0,
            'gradient_disabled': 0,
            'parameter_sharing': False
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_insights['gradient_enabled'] += param.numel()
            else:
                param_insights['gradient_disabled'] += param.numel()
        
        return param_insights
    except Exception as e:
        return {'error': str(e)}

@app.route('/api/model_analysis/<model_file>')
def get_model_analysis(model_file):
    """Get comprehensive model analysis including weights, gradients, and performance metrics"""
    try:
        model, model_type, error = smart_load_model(model_file)
        if error:
            return jsonify({'error': error}), 500
        
        analysis = {
            'layer_analysis': analyze_layers(model),
            'weight_distribution': analyze_weight_distribution(model),
            'gradient_flow': analyze_gradient_flow(model),
            'model_complexity': calculate_model_complexity(model),
            'memory_usage': estimate_memory_usage(model),
            'computational_cost': estimate_computational_cost(model),
            'optimization_suggestions': get_optimization_suggestions(model)
        }
        
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/layer_details/<model_file>/<path:layer_name>')
def get_layer_details(model_file, layer_name):
    """Get detailed analysis for a specific layer"""
    try:
        model, model_type, error = smart_load_model(model_file)
        if error:
            return jsonify({'error': error}), 500
        
        layer_info = analyze_specific_layer(model, layer_name)
        return jsonify(layer_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance_metrics/<model_file>')
def get_performance_metrics(model_file):
    """Get performance metrics and benchmarks"""
    try:
        model, model_type, error = smart_load_model(model_file)
        if error:
            return jsonify({'error': error}), 500
        
        metrics = {
            'inference_time': benchmark_inference_time(model),
            'memory_footprint': get_memory_footprint(model),
            'flops': calculate_flops(model),
            'parameter_efficiency': calculate_parameter_efficiency(model),
            'layer_timing': benchmark_layer_timing(model)
        }
        
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_comparison')
def compare_models():
    """Compare multiple models side by side"""
    try:
        model_files = request.args.getlist('models')
        comparison = {}
        
        for model_file in model_files:
            model, model_type, error = smart_load_model(model_file)
            if error:
                comparison[model_file] = {'error': error}
                continue
                
            comparison[model_file] = {
                'parameters': sum(p.numel() for p in model.parameters()),
                'layers': len(list(model.modules())),
                'memory_usage': estimate_memory_usage(model),
                'complexity_score': calculate_complexity_score(model)
            }
        
        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/legacy')
def legacy_index():
    """Legacy index page for traditional neural network visualization"""
    model_files = get_pt_files()
    return render_template('index.html', show_options=True, model_files=model_files, selected_model_file=None)

@app.route('/analytics_dashboard')
def analytics_dashboard():
    """Analytics dashboard for neural network management"""
    try:
        model_files = get_pt_files()
        return render_template('analytics_dashboard.html', model_files=model_files)
    except Exception as e:
        return f"Error loading analytics dashboard: {str(e)}", 500

# Advanced Neural Network Management and Analytics API Endpoints

@app.route('/api/model_health/<model_file>')
def get_model_health(model_file):
    """Comprehensive model health assessment"""
    try:
        model, model_type, error = smart_load_model(model_file)
        if error:
            return jsonify({'error': error}), 500
        
        health_score = calculate_model_health(model)
        return jsonify(health_score)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/layer_insights/<model_file>')
def get_layer_insights(model_file):
    """Deep layer-by-layer insights and recommendations"""
    try:
        model, model_type, error = smart_load_model(model_file)
        if error:
            return jsonify({'error': error}), 500
        
        insights = analyze_layer_insights(model)
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training_insights/<model_file>')
def get_training_insights(model_file):
    """Extract training insights from model state"""
    try:
        model, model_type, error = smart_load_model(model_file)
        if error:
            return jsonify({'error': error}), 500
        
        insights = analyze_training_state(model)
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/architecture_analysis/<model_file>')
def get_architecture_analysis(model_file):
    """Detailed architecture analysis and patterns"""
    try:
        model, model_type, error = smart_load_model(model_file)
        if error:
            return jsonify({'error': error}), 500
        
        analysis = analyze_architecture_patterns(model)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimization_report/<model_file>')
def get_optimization_report(model_file):
    """Comprehensive optimization recommendations"""
    try:
        model, model_type, error = smart_load_model(model_file)
        if error:
            return jsonify({'error': error}), 500
        
        report = generate_optimization_report(model)
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/weight_analysis/<model_file>')
def get_weight_analysis(model_file):
    """Advanced weight distribution and health analysis"""
    try:
        model, model_type, error = smart_load_model(model_file)
        if error:
            return jsonify({'error': error}), 500
        
        analysis = perform_weight_analysis(model)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_efficiency/<model_file>')
def get_model_efficiency(model_file):
    """Model efficiency metrics and benchmarks"""
    try:
        model, model_type, error = smart_load_model(model_file)
        if error:
            return jsonify({'error': error}), 500
        
        efficiency = calculate_model_efficiency(model)
        return jsonify(efficiency)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_insights_summary/<model_file>')
def get_model_insights_summary(model_file):
    """Comprehensive model insights dashboard data"""
    try:
        model, model_type, error = smart_load_model(model_file)
        if error:
            return jsonify({'error': error}), 500
        
        summary = {
            'basic_info': get_basic_model_info(model),
            'health_score': calculate_model_health(model),
            'efficiency_metrics': calculate_model_efficiency(model),
            'architecture_patterns': analyze_architecture_patterns(model),
            'optimization_priority': get_optimization_priorities(model),
            'layer_distribution': analyze_layer_distribution(model),
            'parameter_insights': analyze_parameter_insights(model)
        }
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Analysis helper functions
def analyze_layers(model):
    """Analyze each layer's properties"""
    layers_info = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            layer_info = {
                'name': name,
                'type': type(module).__name__,
                'parameters': sum(p.numel() for p in module.parameters()),
                'trainable_params': sum(p.numel() for p in module.parameters() if p.requires_grad),
                'input_shape': getattr(module, 'in_features', None) or getattr(module, 'in_channels', None),
                'output_shape': getattr(module, 'out_features', None) or getattr(module, 'out_channels', None),
                'activation_function': None,
                'regularization': {
                    'dropout': getattr(module, 'p', None) if hasattr(module, 'p') else None,
                    'batch_norm': 'BatchNorm' in type(module).__name__
                }
            }
            
            # Get weight statistics if available
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                layer_info['weight_stats'] = {
                    'mean': float(weight.mean()),
                    'std': float(weight.std()),
                    'min': float(weight.min()),
                    'max': float(weight.max()),
                    'shape': list(weight.shape),
                    'sparsity': float((weight == 0).float().mean())
                }
            
            layers_info.append(layer_info)
    
    return layers_info

def analyze_weight_distribution(model):
    """Analyze weight distributions across the model"""
    distributions = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data.cpu().numpy().flatten()
            distributions[name] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'histogram': np.histogram(data, bins=50)[0].tolist(),
                'histogram_bins': np.histogram(data, bins=50)[1].tolist(),
                'percentiles': {
                    '25': float(np.percentile(data, 25)),
                    '50': float(np.percentile(data, 50)),
                    '75': float(np.percentile(data, 75)),
                    '95': float(np.percentile(data, 95))
                }
            }
    
    return distributions

def analyze_gradient_flow(model):
    """Analyze gradient flow (simulated for saved models)"""
    gradient_info = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Since this is a saved model, we can't get actual gradients
            # But we can analyze the weights to infer potential gradient issues
            weight = param.data
            gradient_info[name] = {
                'parameter_count': param.numel(),
                'weight_magnitude': float(torch.norm(weight)),
                'potential_vanishing': float(torch.norm(weight)) < 1e-4,
                'potential_exploding': float(torch.norm(weight)) > 10,
                'initialization_health': analyze_initialization_health(weight)
            }
    
    return gradient_info

def analyze_initialization_health(weight):
    """Analyze if weights are well initialized"""
    std = weight.std()
    fan_in = weight.size(1) if len(weight.shape) > 1 else 1
    
    # Check for common initialization patterns
    xavier_std = np.sqrt(2.0 / fan_in)
    he_std = np.sqrt(2.0 / fan_in)
    
    return {
        'current_std': float(std),
        'xavier_expected': float(xavier_std),
        'he_expected': float(he_std),
        'likely_xavier': abs(float(std) - xavier_std) < 0.1,
        'likely_he': abs(float(std) - he_std) < 0.1
    }

def calculate_model_complexity(model):
    """Calculate various complexity metrics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate depth
    depth = len([m for m in model.modules() if len(list(m.children())) == 0])
    
    # Calculate width (max layer size)
    max_width = 0
    for module in model.modules():
        if hasattr(module, 'out_features'):
            max_width = max(max_width, module.out_features)
        elif hasattr(module, 'out_channels'):
            max_width = max(max_width, module.out_channels)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'depth': depth,
        'max_width': max_width,
        'parameter_density': trainable_params / depth if depth > 0 else 0,
        'complexity_score': np.log10(total_params) * depth / 100
    }

def estimate_memory_usage(model):
    """Estimate memory usage"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        'parameters_mb': param_size / 1024 / 1024,
        'buffers_mb': buffer_size / 1024 / 1024,
        'total_mb': (param_size + buffer_size) / 1024 / 1024,
        'fp32_inference_mb': (param_size + buffer_size) / 1024 / 1024,
        'fp16_inference_mb': (param_size + buffer_size) / 2 / 1024 / 1024
    }

def estimate_computational_cost(model):
    """Estimate computational cost"""
    total_ops = 0
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total_ops += module.in_features * module.out_features
        elif isinstance(module, nn.Conv2d):
            # Assuming typical input size - this is a rough estimate
            kernel_flops = module.in_channels * module.out_channels * np.prod(module.kernel_size)
            total_ops += kernel_flops * 224 * 224  # Assuming 224x224 input
    
    return {
        'estimated_ops': total_ops,
        'gflops': total_ops / 1e9,
        'computational_intensity': total_ops / sum(p.numel() for p in model.parameters())
    }

def get_optimization_suggestions(model):
    """Provide optimization suggestions"""
    suggestions = []
    total_params = sum(p.numel() for p in model.parameters())
    
    # Check for oversized models
    if total_params > 10_000_000:
        suggestions.append({
            'type': 'warning',
            'category': 'Model Size',
            'message': 'Model has >10M parameters. Consider pruning or quantization.',
            'impact': 'high'
        })
    
    # Check for potential overfitting indicators
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) > 5:
        suggestions.append({
            'type': 'info',
            'category': 'Architecture',
            'message': 'Many linear layers detected. Consider regularization techniques.',
            'impact': 'medium'
        })
    
    # Check for batch normalization usage
    has_batchnorm = any('BatchNorm' in str(type(m)) for m in model.modules())
    if not has_batchnorm and len(linear_layers) > 3:
        suggestions.append({
            'type': 'suggestion',
            'category': 'Regularization',
            'message': 'Consider adding batch normalization for better training stability.',
            'impact': 'medium'
        })
    
    return suggestions

def analyze_specific_layer(model, layer_name):
    """Get detailed analysis for a specific layer"""
    try:
        # Navigate to the specific layer
        layer = model
        for attr in layer_name.split('.'):
            layer = getattr(layer, attr)
        
        analysis = {
            'name': layer_name,
            'type': type(layer).__name__,
            'parameters': {},
            'architecture_details': {},
            'performance_characteristics': {}
        }
        
        # Get parameter details
        for name, param in layer.named_parameters():
            if param is not None:
                analysis['parameters'][name] = {
                    'shape': list(param.shape),
                    'requires_grad': param.requires_grad,
                    'dtype': str(param.dtype),
                    'device': str(param.device),
                    'memory_mb': param.numel() * param.element_size() / 1024 / 1024
               
                }
        
        # Get architecture-specific details
        if isinstance(layer, nn.Linear):
            analysis['architecture_details'] = {
                'input_features': layer.in_features,
                'output_features': layer.out_features,
                'bias': layer.bias is not None,
                'parameter_ratio': layer.out_features / layer.in_features
            }
        elif isinstance(layer, nn.Conv2d):
            analysis['architecture_details'] = {
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size,
                'stride': layer.stride,
                'padding': layer.padding,
                'dilation': layer.dilation,
                'groups': layer.groups,
                'bias': layer.bias is not None
            }
        
        return analysis
    except Exception as e:
        return {'error': f'Layer {layer_name} not found: {str(e)}'}

def benchmark_inference_time(model):
    """Benchmark inference time (simulated)"""
    total_params = sum(p.numel() for p in model.parameters())
    
    # Rough estimation based on parameter count
    estimated_ms = np.log10(total_params) * 5
    
    return {
        'estimated_inference_ms': estimated_ms,
        'parameters_factor': total_params / 1000000,
        'note': 'Estimated based on parameter count. Actual timing requires live model.'
    }

def get_memory_footprint(model):
    """Get detailed memory footprint"""
    memory_breakdown = {}
    total_memory = 0
    
    for name, param in model.named_parameters():
        memory = param.numel() * param.element_size()
        memory_breakdown[name] = {
            'bytes': memory,
            'mb': memory / 1024 / 1024,
            'percentage': 0  # Will be calculated after total
        }
        total_memory += memory
    
    # Calculate percentages
    for name in memory_breakdown:
        memory_breakdown[name]['percentage'] = (
            memory_breakdown[name]['bytes'] / total_memory * 100
        )
    
    return {
        'total_mb': total_memory / 1024 / 1024,
        'breakdown': memory_breakdown,
        'largest_layer': max(memory_breakdown.items(), key=lambda x: x[1]['bytes'])[0]
    }

def calculate_flops(model):
    """Calculate FLOPs (rough estimation)"""
    total_flops = 0
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total_flops += module.in_features * module.out_features * 2  # multiply-add
        elif isinstance(module, nn.Conv2d):
            # Assuming typical input size - this is a rough estimate
            kernel_flops = module.in_channels * module.out_channels * np.prod(module.kernel_size)
            total_flops += kernel_flops * 224 * 224  # Assuming 224x224 input
    
    return {
        'total_flops': total_flops,
        'gflops': total_flops / 1e9,
        'note': 'Rough estimation. Actual FLOPs depend on input size.'
    }

def calculate_parameter_efficiency(model):
    """Calculate parameter efficiency metrics"""
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count different layer types
    linear_layers = len([m for m in model.modules() if isinstance(m, nn.Linear)])
    conv_layers = len([m for m in model.modules() if isinstance(m, nn.Conv2d)])
    
    return {
        'params_per_layer': total_params / max(linear_layers + conv_layers, 1),
        'parameter_density': total_params / len(list(model.modules())),
        'efficiency_score': np.log10(total_params) / len(list(model.modules()))
    }

def benchmark_layer_timing(model):
    """Benchmark timing for each layer (simulated)"""
    layer_timings = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            param_count = sum(p.numel() for p in module.parameters())
            # Rough estimation based on parameter count and layer type
            base_time = np.log10(max(param_count, 1)) * 0.1
            
            if isinstance(module, nn.Linear):
                estimated_time = base_time * 1.0
            elif isinstance(module, nn.Conv2d):
                estimated_time = base_time * 2.0
            elif 'BatchNorm' in type(module).__name__:
                estimated_time = base_time * 0.3
            else:
                estimated_time = base_time * 0.5
            
            layer_timings[name] = {
                'estimated_ms': estimated_time,
                'parameter_count': param_count,
                'relative_cost': estimated_time / base_time if base_time > 0 else 1
            }
    
    return layer_timings

def calculate_complexity_score(model):
    """Calculate overall complexity score"""
    total_params = sum(p.numel() for p in model.parameters())
    depth = len([m for m in model.modules() if len(list(m.children())) == 0])
    # Weighted complexity score
    param_score = np.log10(total_params) * 10
    depth_score = depth * 5
    
    return param_score + depth_score

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
