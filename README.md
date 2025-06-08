# 🧠 Advanced Neural Network Visualizer & Analytics Dashboard

A comprehensive Flask-based application for visualizing PyTorch neural networks with an advanced analytics dashboard, plus multiple viewing modes including 2D, 3D, hierarchical, force-directed, and interactive inspection views.

![Neural Network Visualizer](https://img.shields.io/badge/Neural%20Network-Visualizer-blue) ![Python](https://img.shields.io/badge/Python-3.7+-green) ![Flask](https://img.shields.io/badge/Flask-Web%20App-red) ![Analytics](https://img.shields.io/badge/Analytics-Dashboard-orange) ![3D](https://img.shields.io/badge/3D-Visualization-purple)

## 📋 Table of Contents
- [🚀 Key Features & Recent Updates](#-key-features--recent-updates)
- [🎯 Analytics Dashboard - Main Feature](#-analytics-dashboard---main-feature)
  - [📊 Dashboard Overview](#-dashboard-overview)
  - [🧠 Advanced Analytics Features](#-advanced-analytics-features)
  - [🔄 Model Comparison](#-model-comparison)
  - [🔍 Interactive Model Inspector](#-interactive-model-inspector)
- [🎨 Additional Visualization Modes](#-additional-visualization-modes)
- [⚡ Quick Start](#-quick-start)
- [📚 API Endpoints](#-api-endpoints)
- [📷 Screenshots Added ✅](#-screenshots-added-)
- [🎨 Customization](#-customization)
- [🛠️ Technology Stack](#️-technology-stack)

## 🚀 Key Features & Recent Updates

✅ **Analytics Dashboard as Home Page** - Comprehensive neural network analysis interface  
✅ **Weight Distribution Charts** - Fixed loading issues and smart layer name matching  
✅ **Memory Analysis** - Complete implementation with dual visualization modes  
✅ **Performance Benchmarking** - Real-time inference time and memory footprint analysis  
✅ **Model Health Check** - AI-powered optimization suggestions and issue detection  
✅ **Interactive Training Insights** - Layer efficiency analysis and training recommendations  
✅ **Model Comparison** - Side-by-side analysis of multiple models  
✅ **Chart.js Error Protection** - Robust error handling for all visualizations  
✅ **Mobile Responsive Design** - Touch-enabled controls for all devices  
📸 **Complete Documentation** - README with actual screenshots and comprehensive guides

## 🎯 Analytics Dashboard - Main Feature

The **NN Analytics Dashboard** is now the default home page, providing comprehensive neural network analysis and insights with **live screenshots**:

### 📊 Dashboard Overview

![Analytics Dashboard - Main View](docs/analytics-dashboard-main.png)

*Comprehensive analytics interface showing real-time model metrics, layer analysis, weight distribution, and performance insights*

**Model Overview Card**: Real-time metrics including:
- 📊 **6.34M Total Parameters** & **6.34M Trainable Parameters**
- 🏗️ **47 Model Layers** with detailed architecture summary  
- 💾 **24.2 MB Memory Usage** analysis
- 📈 **98.4% Efficiency Score** & **100.0 Health Score**
- 🎯 **Architecture Summary**: Model type, parameter efficiency, memory efficiency

**Layer Analysis**: Interactive layer inspection with:
- 🔍 **Layer Selection**: Sort by name, parameters, or memory
- 📋 **Layer Details**: Type, parameters, input/output dimensions, efficiency
- 💡 **Recommendations**: Automated optimization suggestions

**Weight Distribution**: Advanced statistical analysis:
- 📈 **Interactive Charts**: Select any layer for weight statistics
- 📊 **Statistical Metrics**: Min, Mean, Max, Std Dev, Sparsity
- 🎨 **Visual Charts**: Bar charts showing weight distribution patterns

**Performance Metrics**: Comprehensive benchmarking:
- ⚡ **34.01 ms** Estimated Inference Time
- 💾 **24.2 MB** Memory Footprint  
- 🔢 **0.01 GFLOPS** Computational Cost
- 📊 **0.14** Efficiency Score
- 🔍 **Performance Insights**: Largest memory consumers, parameter density

### 🧠 Advanced Analytics Features

![Analytics Dashboard - Advanced Features](docs/analytics-dashboard-advanced.png)
![image](https://github.com/user-attachments/assets/379a431c-5056-4a99-9e68-752a9d6d9407)

*Advanced analytics including memory analysis, optimization suggestions, health checks, and training insights*

**Memory Analysis**: Dual-view memory breakdown:
- 🍩 **Memory Breakdown**: Doughnut chart showing layer-wise memory usage (Top 10)
- 📊 **Memory Efficiency**: Bar charts with color-coded efficiency metrics
- 🔄 **Switchable Views**: Toggle between breakdown and efficiency modes

**Optimization Suggestions**: AI-powered recommendations:
- 💡 **Smart Analysis**: Automated optimization detection
- 🎯 **Priority System**: High/Medium/Low priority suggestions
- 🔧 **Actionable Insights**: Specific optimization techniques (quantization, pruning, etc.)

**Model Health Check**: Comprehensive model assessment:
- ❤️ **Health Score**: Overall model health rating (100/100)
- ⚠️ **Issues Detection**: Automatic problem identification
- 📋 **Health Reports**: Detailed analysis of model architecture

**Training Insights**: Advanced training analysis:
- 🎯 **Training Recommendations**: Data-driven training suggestions
- 📊 **Layer Efficiency**: Per-layer training efficiency analysis
- ⚠️ **Low Efficiency Detection**: Automatic identification of underperforming layers
- 💡 **Optimization Tips**: Specific suggestions for model improvement

### 🔄 Model Comparison
**Side-by-Side Analysis**: Compare multiple models simultaneously:
- 📊 **Dual Selection**: Select main model and comparison model
- 🔍 **Comprehensive Metrics**: Compare all analytics features side-by-side
- 📈 **Performance Benchmarking**: Direct performance comparison

### 🔍 Interactive Model Inspector

![Interactive Model Inspector](docs/interactive-inspector.png)

*Interactive neural network inspector with detailed layer analysis and model structure exploration*

**Interactive Inspector Features**:
- 🌳 **Model Structure Tree**: Hierarchical view of all model components
- 📊 **Live Statistics**: Real-time parameter counts and model information
- 🔍 **Layer Details**: Click any layer for detailed inspection
- 📋 **Model Information**: Architecture type, vocabulary size, and dimensions
- 🎯 **Parameter Breakdown**: Detailed parameter distribution across layers

**Advanced Navigation**:
- 📂 **Expandable Tree**: Navigate through complex model hierarchies
- 🎨 **Color-Coded Layers**: Visual distinction between layer types
- 📊 **Architecture Overview**: Input, Hidden, Output, and Activation layer categorization
- 💾 **Memory Visualization**: Parameter count and memory usage per component
- 📊 **Metric Comparison**: Parameters, layers, memory, complexity scores
- 📈 **Performance Differential**: Direct comparison with difference calculations
- 💡 **Smart Recommendations**: AI-powered suggestions based on comparison results

### 🎮 Interactive Features
- 🔍 **Real-Time Analysis**: Click "Analyze Model" for instant comprehensive analysis
- 🔄 **Model Switching**: Seamless switching between different models
- 📊 **Live Charts**: Interactive Chart.js visualizations with hover details
- 🎛️ **Control Panels**: Intuitive controls for all dashboard functions

### 🔍 Interactive Neural Network Inspector

![Interactive Inspector](docs/interactive-inspector.png)

*Detailed model exploration with hierarchical structure tree, parameter analysis, and comprehensive statistics*

**Comprehensive Model Exploration**:
- 🎯 **Model Selection**: Load and analyze any PyTorch model file
- 📊 **6,344,000 Parameters** & **47 Modules** with detailed breakdown
- 🌳 **Model Structure Tree**: Hierarchical view of all model components
- 📋 **Detailed Statistics**: Parameter counts, module types, and relationships

**Module Analysis**:
- 🔍 **Interactive Tree**: Click any module for detailed inspection
- 📊 **Parameter Details**: embedding (2,046,000 params), position_embedding (131,072 params)
- 🏗️ **Architecture Layers**: transformer, layers, self_attn, out_proj analysis
- 💾 **Memory Footprint**: Per-module memory usage and efficiency

**Architecture Overview**:
- 📐 **Input Layers**: 1 layer analysis
- 🧠 **Hidden Layers**: 12 layers with detailed breakdown  
- 📤 **Output Layers**: 0 layers (decoder architecture)
- ⚡ **Activations**: 0 functions analysis

## ✨ Additional Visualization Features

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
Navigate to `http://127.0.0.1:5000/` to access the **Analytics Dashboard** (default home page), or explore other visualization modes!

## 🎮 Usage Guide

### 🎯 Analytics Dashboard (Default Home Page)
The main analytics interface provides comprehensive model analysis:

**Getting Started**:
1. **Select Model**: Choose from dropdown "Select Model for Analysis"
2. **Click "Analyze Model"**: Comprehensive analysis loads automatically
3. **Explore Cards**: Model Overview, Layer Analysis, Weight Distribution, Performance Metrics
4. **Memory Analysis**: View memory breakdown and efficiency charts
5. **Health Check**: Run model health assessment with optimization suggestions
6. **Training Insights**: Get AI-powered training recommendations

**Advanced Features**:
- **Model Comparison**: Select second model and click "Compare Models"
- **Weight Analysis**: Select any layer to view detailed weight statistics
- **Performance Benchmark**: Click "Run Benchmark" for detailed performance metrics
- **Export Options**: Save analysis results and charts

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
├── app.py                     # Main Flask application with analytics & visualization routes
├── requirements.txt           # Python dependencies
├── *.pt                      # Your PyTorch model files
├── docs/                     # Documentation and screenshots
│   ├── analytics-dashboard-main.png      # Main dashboard screenshot ✅
│   ├── analytics-dashboard-advanced.png  # Advanced features screenshot ✅
│   └── interactive-inspector.png         # Interactive inspector screenshot ✅
├── templates/                # HTML templates
│   ├── analytics_dashboard.html # Analytics Dashboard (Main Feature)
│   ├── index.html            # Legacy 2D visualization
│   ├── visualize_3d.html     # 3D Three.js visualization
│   ├── visualize_hierarchical.html  # D3.js tree view
│   ├── visualize_force.html  # Force-directed network
│   └── visualize_interactive.html   # Interactive inspector
├── static/
│   └── images/               # Generated 2D visualization images
└── __pycache__/              # Python cache files
```

## 🔧 API Endpoints

### Main Application Routes
- **`/`** - Analytics Dashboard (Default Home Page)
- **`/analytics_dashboard`** - Alternative analytics dashboard route
- **`/legacy`** - Original 2D visualization interface
- **`/visualize_3d`** - 3D visualization mode
- **`/visualize_hierarchical`** - Hierarchical tree view
- **`/visualize_force`** - Force-directed network view
- **`/visualize_interactive`** - Interactive inspector

### Analytics API Endpoints
- **`/api/model_insights_summary/<model_file>`** - Comprehensive model analysis
- **`/api/layer_insights/<model_file>`** - Layer-by-layer analysis
- **`/api/weight_analysis/<model_file>`** - Weight distribution statistics
- **`/api/performance_metrics/<model_file>`** - Performance benchmarking
- **`/api/model_comparison`** - Multi-model comparison
- **`/api/model_health/<model_file>`** - Model health assessment
- **`/api/training_insights/<model_file>`** - Training optimization insights
- **`/api/architecture_analysis/<model_file>`** - Architecture pattern analysis
- **`/api/optimization_report/<model_file>`** - Optimization recommendations

### Legacy API Endpoints
- **`/api/model_data_3d`** - JSON API for 3D model data
- **`/api/model_structure`** - JSON API for detailed model analysis
- **`/visualize_selected`** - Load specific model visualization

## 📷 Screenshots Added ✅

The README now includes actual screenshots of the Neural Network Analytics Dashboard in action:

### Current Screenshots
✅ **Analytics Dashboard Main View** (`docs/analytics-dashboard-main.png`)
   - Model Overview with real metrics (6.34M parameters, 47 layers, 24.2MB memory)
   - Layer Analysis with interactive layer selection and details
   - Weight Distribution charts with statistical analysis
   - Performance Metrics showing inference time and efficiency scores

✅ **Analytics Dashboard Advanced View** (`docs/analytics-dashboard-advanced.png`)
   - Memory Analysis with doughnut chart breakdown
   - Optimization Suggestions with AI-powered recommendations
   - Model Health Check with 100/100 health score
   - Training Insights with layer efficiency analysis

✅ **Interactive Inspector** (`docs/interactive-inspector.png`)
   - Complete model structure tree with expandable components
   - Real-time parameter counts and model statistics
   - Interactive layer exploration with detailed information
   - Architecture overview with color-coded layer types

### Update Screenshots (Optional)
To capture new screenshots with different models or updated features:

### 1. Capture Screenshots
1. **Run the application**: `python app.py`
2. **Open browser**: Navigate to `http://127.0.0.1:5000/`

3. **Analytics Dashboard Main View**:
   - Select a model (e.g., `final_amharic_religious_model.pt`)
   - Click "Analyze Model" 
   - Wait for all cards to load with data
   - Take screenshot and **replace** `docs/analytics-dashboard-main.png`

4. **Analytics Dashboard Advanced View**:
   - Scroll down to show Memory Analysis, Optimization Suggestions, Health Check, Training Insights
   - Take screenshot and **replace** `docs/analytics-dashboard-advanced.png`

5. **Interactive Inspector**:
   - Navigate to `/visualize_interactive` 
   - Select a model and click "Load & Analyze"
   - Take screenshot showing model structure tree and statistics
   - **Replace** `docs/interactive-inspector.png`

### 2. File Locations
The placeholder files are already created at:
- `docs/analytics-dashboard-main.png`
- `docs/analytics-dashboard-advanced.png`
- `docs/interactive-inspector.png`

Simply replace these files with your actual screenshots to update the README automatically.

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

- **Backend**: Flask, PyTorch, Graphviz, NumPy, SciPy
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Chart.js
- **Analytics**: Advanced neural network analysis with statistical computations
- **Visualization**: Three.js, D3.js, Chart.js, Graphviz
- **UI/UX**: Modern CSS with glassmorphism, card-based layouts, and smooth animations
- **Responsive Design**: Mobile-first approach with touch controls and adaptive layouts
- **Performance**: Optimized loading, lazy chart rendering, and efficient memory management

## 🎉 Project Status: Complete ✅

This Neural Network Visualizer & Analytics Dashboard project has been successfully completed with all major features implemented and thoroughly tested:

### ✅ **Completed Features**
- 🏠 **Analytics Dashboard as Home Page** - Fully functional with comprehensive model analysis
- 📊 **Weight Distribution Charts** - Smart layer matching with robust error handling
- 💾 **Memory Analysis** - Dual-view memory visualization with breakdown and efficiency charts
- ⚡ **Performance Benchmarking** - Real-time inference time and computational cost analysis
- 🔍 **Model Health Check** - AI-powered optimization suggestions and issue detection
- 📈 **Training Insights** - Layer efficiency analysis with actionable recommendations
- 🔄 **Model Comparison** - Side-by-side analysis with comprehensive metrics
- 🔧 **Error Resolution** - All Chart.js context errors and innerHTML issues resolved
- 📱 **Responsive Design** - Mobile-friendly interface with touch controls
- 📸 **Complete Documentation** - README with actual screenshots and comprehensive guides

### 🌟 **Key Achievements**
- **Zero JavaScript Errors** - All Chart.js context acquisition and innerHTML errors resolved
- **Smart Layer Matching** - Advanced algorithm for matching dropdown names with weight statistics
- **Comprehensive Analytics** - Full-featured dashboard with multiple visualization modes
- **Professional Documentation** - Complete README with live screenshots and detailed guides
- **Production Ready** - Robust error handling, loading states, and user feedback

The application is now ready for production use with a complete analytics dashboard, multiple visualization modes, and comprehensive documentation.
