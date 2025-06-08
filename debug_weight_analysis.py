#!/usr/bin/env python3
"""
Debug script for weight analysis data structure investigation.
This script helps identify what layer names are available vs what's being requested.
"""

import requests
import json
import sys

def test_api_endpoint(url, description):
    """Test an API endpoint and return the response."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    try:
        response = requests.get(url, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response received successfully")
            return data
        else:
            print(f"Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Error text: {response.text[:500]}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def analyze_weight_statistics(data):
    """Analyze the weight statistics structure."""
    print(f"\n{'='*40}")
    print("WEIGHT STATISTICS ANALYSIS")
    print(f"{'='*40}")
    
    if not data:
        print("❌ No data to analyze")
        return
    
    if 'error' in data:
        print(f"❌ API Error: {data['error']}")
        return
    
    print("✅ Data structure:")
    print(f"   Top-level keys: {list(data.keys())}")
    
    if 'weight_statistics' in data:
        weight_stats = data['weight_statistics']
        print(f"   Weight statistics type: {type(weight_stats)}")
        
        if isinstance(weight_stats, dict):
            print(f"   Available layers in weight_statistics: {len(weight_stats)} layers")
            layer_names = list(weight_stats.keys())
            print(f"   Layer names:")
            for i, name in enumerate(layer_names[:10]):  # Show first 10
                print(f"     {i+1:2d}. '{name}'")
            
            if len(layer_names) > 10:
                print(f"     ... and {len(layer_names) - 10} more layers")
            
            # Analyze a sample layer
            if layer_names:
                sample_layer = layer_names[0]
                sample_data = weight_stats[sample_layer]
                print(f"\n   Sample layer '{sample_layer}':")
                print(f"   Structure: {sample_data}")
                
                if isinstance(sample_data, dict):
                    print(f"   Available statistics: {list(sample_data.keys())}")
        else:
            print(f"   ❌ weight_statistics is not a dictionary: {weight_stats}")
    else:
        print("   ❌ No 'weight_statistics' key found")
    
    # Check for layer analysis to compare layer names
    if 'layer_analysis' in data:
        layer_analysis = data['layer_analysis']
        print(f"\n   Layer analysis available: {type(layer_analysis)}")
        if isinstance(layer_analysis, dict) and 'layers' in layer_analysis:
            analysis_layers = [layer.get('name', '') for layer in layer_analysis['layers']]
            print(f"   Layers in analysis: {analysis_layers[:5]}{'...' if len(analysis_layers) > 5 else ''}")

def compare_layer_sources(model_name):
    """Compare layer names from different API endpoints."""
    print(f"\n{'='*50}")
    print("LAYER NAME COMPARISON")
    print(f"{'='*50}")
    
    base_url = "http://localhost:5000"
    
    # Get weight analysis
    weight_data = test_api_endpoint(
        f"{base_url}/api/weight_analysis/{model_name}",
        "Weight Analysis API"
    )
    
    # Get layer insights
    layer_data = test_api_endpoint(
        f"{base_url}/api/layer_insights/{model_name}",
        "Layer Insights API"
    )
    
    # Get model insights summary
    summary_data = test_api_endpoint(
        f"{base_url}/api/model_insights_summary/{model_name}",
        "Model Insights Summary API"
    )
    
    # Extract layer names from each source
    weight_layers = set()
    if weight_data and 'weight_statistics' in weight_data:
        weight_layers = set(weight_data['weight_statistics'].keys())
    
    insight_layers = set()
    if layer_data:
        if isinstance(layer_data, list):
            insight_layers = set(layer.get('name', '') for layer in layer_data)
        elif isinstance(layer_data, dict) and 'layer_details' in layer_data:
            insight_layers = set(layer.get('name', '') for layer in layer_data['layer_details'])
    
    summary_layers = set()
    if summary_data and 'layer_insights' in summary_data:
        layer_insights = summary_data['layer_insights']
        if 'layer_details' in layer_insights:
            summary_layers = set(layer.get('name', '') for layer in layer_insights['layer_details'])
    
    print(f"\nLayer name comparison:")
    print(f"  Weight analysis layers: {len(weight_layers)} layers")
    print(f"  Layer insights layers:  {len(insight_layers)} layers")
    print(f"  Summary layers:         {len(summary_layers)} layers")
    
    if weight_layers:
        print(f"\n  Sample weight layer names:")
        for name in list(weight_layers)[:5]:
            print(f"    - '{name}'")
    
    if insight_layers:
        print(f"\n  Sample insight layer names:")
        for name in list(insight_layers)[:5]:
            print(f"    - '{name}'")
    
    # Find intersections and differences
    if weight_layers and insight_layers:
        common = weight_layers & insight_layers
        weight_only = weight_layers - insight_layers
        insight_only = insight_layers - weight_layers
        
        print(f"\n  Common layers: {len(common)}")
        print(f"  Weight-only layers: {len(weight_only)}")
        print(f"  Insight-only layers: {len(insight_only)}")
        
        if weight_only:
            print(f"  Weight-only examples: {list(weight_only)[:3]}")
        if insight_only:
            print(f"  Insight-only examples: {list(insight_only)[:3]}")

def test_specific_model_analysis():
    """Test analysis for a specific model that should have data."""
    print(f"\n{'='*60}")
    print("TESTING SPECIFIC MODEL ANALYSIS")
    print(f"{'='*60}")
    
    base_url = "http://localhost:5000"
    
    # First, get list of available models
    models_data = test_api_endpoint(f"{base_url}/api/models", "Available Models")
    
    if not models_data or 'models' not in models_data:
        print("❌ Cannot get model list")
        return
    
    models = models_data['models']
    print(f"Available models: {len(models)}")
    
    if not models:
        print("❌ No models available")
        return
    
    # Test with the first model
    test_model = models[0]['filename']
    print(f"\nTesting with model: {test_model}")
    
    # Get weight analysis for this specific model
    weight_data = test_api_endpoint(
        f"{base_url}/api/weight_analysis/{test_model}",
        f"Weight Analysis for {test_model}"
    )
    
    analyze_weight_statistics(weight_data)
    compare_layer_sources(test_model)

def main():
    """Main test function."""
    print("🔍 Neural Network Weight Analysis Debug Tool")
    print("=" * 60)
    
    # Test server availability
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        print("✅ Server is running")
    except:
        print("❌ Server is not running. Please start the Flask app first.")
        return
    
    # Run specific model analysis
    test_specific_model_analysis()
    
    print(f"\n{'='*60}")
    print("DEBUG SUMMARY")
    print(f"{'='*60}")
    print("This script helps identify:")
    print("1. ✅ Chart.js context issues (fixed)")
    print("2. 🔍 Weight statistics data structure")
    print("3. 🔍 Layer name mismatches between APIs")
    print("4. 🔍 Missing weight data for specific layers")
    
    print(f"\n📋 Next steps:")
    print("1. Check the layer name format differences")
    print("2. Verify backend weight analysis is returning correct layer names")
    print("3. Ensure transformer layers are included in weight analysis")
    print("4. Test the fixed Chart.js implementation in browser")

if __name__ == "__main__":
    main()
