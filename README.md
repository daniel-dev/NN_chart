# 🧠 Advanced Neural Network Visualizer

A comprehensive Flask-based application for visualizing PyTorch neural networks with multiple advanced viewing modes including 2D, 3D, hierarchical, force-directed, and interactive inspection views.

![Neural Network Visualizer](https://img.shields.io/badge/Neural%20Network-Visualizer-blue) ![Python](https://img.shields.io/badge/Python-3.7+-green) ![Flask](https://img.shields.io/badge/Flask-Web%20App-red) ![3D](https://img.shields.io/badge/3D-Visualization-purple)

## ✨ Features

### 🎯 Multiple Visualization Modes
- **📊 2D View**: Traditional Graphviz-based network diagrams with interactive zoom, pan, and fullscreen
- **🎯 3D View**: Interactive Three.js-powered 3D visualization with orbital controls
- **🌲 Hierarchical View**: Tree-based D3.js layout showing model architecture hierarchy
- **⚡ Force-Directed View**: Dynamic force simulation network with real-time physics
- **🔍 Interactive Inspector**: Detailed model analysis with parameter inspection

### 🚀 Advanced Features
- **Multi-Model Support**: Load and compare different PyTorch model architectures
- **Real-Time Model Loading**: Switch between models without page refresh
- **Interactive Controls**: Adjust visualization parameters with live sliders
- **Export Capabilities**: Download visualizations as PNG images
- **Mobile Responsive**: Touch-enabled controls for mobile devices
- **Professional UI**: Modern design with glassmorphism effects and smooth animations

### 📱 Supported Model Types
- Standard PyTorch `nn.Module` models
- Custom architectures (SimpleNN, AmharicTransformerModel)
- State dictionaries (`.pt` files with `state_dict`)
- Pre-trained models with complex hierarchies

## 🛠️ Prerequisites

- **Python 3.7+**
- **[Graphviz](https://graphviz.org/download/)** (for 2D visualization)
  - Windows: `choco install graphviz` or download installer
  - Ensure Graphviz `bin` directory is in PATH
- **Modern Web Browser** (Chrome, Firefox, Safari, Edge)

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Place Your Models
Add your PyTorch model files (`.pt`) to the project root:
```text
c:\RND\LLM\NN\NN_chart\
├── your_model.pt
├── another_model.pt
└── ...
```

### 3. Launch the Application
```powershell
python app.py
```

### 4. Open in Browser
Navigate to `http://127.0.0.1:5000/` and explore the different visualization modes!

## 🎮 Usage Guide

### 📊 2D View (Traditional)
- **Zoom**: Mouse wheel or zoom controls
- **Pan**: Click and drag the visualization
- **Fullscreen**: Click fullscreen button for immersive viewing
- **Export**: Download high-resolution PNG images
- **Minimap**: Toggle minimap for easy navigation of large networks

### 🎯 3D View (Interactive)
- **Camera Controls**: Click and drag to orbit around the model
- **Zoom**: Mouse wheel to zoom in/out
- **Node Inspection**: Click nodes to view detailed information
- **Layout Modes**: Switch between 3D, hierarchical, force-directed, and circular layouts
- **Visual Controls**: Adjust node size, spread, and animation speed
- **Real-time Loading**: Select models from dropdown for instant visualization

### 🌲 Hierarchical View
- **Tree Layout**: Visualizes model as an expandable tree structure
- **Interactive Nodes**: Hover for tooltips, click for detailed information
- **Responsive Design**: Adapts to different screen sizes
- **Smooth Animations**: Fluid transitions between states

### ⚡ Force-Directed View
- **Physics Simulation**: Real-time force-directed network layout
- **Interactive Physics**: Adjust link strength, charge force, and center gravity
- **Dynamic Visualization**: Nodes and connections move based on network topology
- **Performance Optimized**: Smooth 60fps animations even for large networks

### 🔍 Interactive Inspector
- **Detailed Analysis**: In-depth exploration of model parameters
- **Module Tree**: Navigate through model hierarchy
- **Parameter Inspection**: View weights, biases, and layer configurations
- **Statistics Dashboard**: Model metrics and architectural overview

## 📁 Project Structure

```text
NN_chart/
├── app.py                     # Main Flask application with all routes
├── requirements.txt           # Python dependencies
├── *.pt                      # Your PyTorch model files
├── templates/                # HTML templates
│   ├── index.html            # 2D visualization (enhanced)
│   ├── visualize_3d.html     # 3D Three.js visualization
│   ├── visualize_hierarchical.html  # D3.js tree view
│   ├── visualize_force.html  # Force-directed network
│   └── visualize_interactive.html   # Interactive inspector
├── static/
│   └── images/               # Generated 2D visualization images
└── __pycache__/              # Python cache files
```

## 🔧 API Endpoints

- **`/`** - Main 2D visualization interface
- **`/visualize_3d`** - 3D visualization mode
- **`/visualize_hierarchical`** - Hierarchical tree view
- **`/visualize_force`** - Force-directed network view
- **`/visualize_interactive`** - Interactive inspector
- **`/api/model_data_3d`** - JSON API for 3D model data
- **`/api/model_structure`** - JSON API for detailed model analysis
- **`/visualize_selected`** - Load specific model visualization

## 🎨 Customization

### Adding New Models
Simply place `.pt` files in the project root - they'll automatically appear in model selection dropdowns.

### Modifying Visualizations
- **Colors**: Edit CSS color schemes in template files
- **Layouts**: Adjust D3.js and Three.js parameters in JavaScript sections
- **Physics**: Modify force simulation parameters for different network behaviors

### Extending Functionality
- **New Views**: Add routes in `app.py` and create corresponding templates
- **Model Support**: Extend model loading logic in `extract_3d_model_data()` functions
- **Export Formats**: Add new export options (SVG, PDF, etc.)

## 🧪 Included Sample Models
- `final_amharic_religious_model.pt` - Complete transformer model
- `simple_amharic_model.pt` - Basic neural network
- `model_epoch10.pt` - Training checkpoint
- `model_state_dict.pt` - State dictionary format

## 🐛 Troubleshooting

### Common Issues

**Graphviz not found**
- Ensure Graphviz is installed and in PATH
- Restart terminal after installation
- Verify with: `dot -V`

**Models not loading**
- Check that `.pt` files are in project root
- Verify models are PyTorch-compatible
- Check console for detailed error messages

**3D visualization not working**
- Ensure modern browser with WebGL support
- Check browser console for JavaScript errors
- Try different browsers if issues persist

**Performance issues**
- Large models may take time to load
- Reduce visualization complexity in control panels
- Use force-directed view for better performance with large networks

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 🔗 Technologies Used

- **Backend**: Flask, PyTorch, Graphviz, NumPy
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Visualization**: Three.js, D3.js, Graphviz
- **UI/UX**: Modern CSS with glassmorphism and smooth animations
- **Responsive Design**: Mobile-first approach with touch controls
