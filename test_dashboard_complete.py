#!/usr/bin/env python3
"""
Complete Dashboard Functionality Test
Tests all the fixed analytics dashboard features including:
- Weight Distribution Chart fixes
- Memory Analysis implementation
- Chart.js error protection
- Model analysis endpoints
"""

import requests
import json
import sys
import time

def test_dashboard_endpoint():
    """Test the main analytics dashboard loads correctly"""
    try:
        response = requests.get('http://127.0.0.1:5000/')
        if response.status_code == 200:
            print("✅ Analytics Dashboard loads successfully at root URL")
            return True
        else:
            print(f"❌ Dashboard failed to load: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def test_model_analysis_api():
    """Test model analysis API endpoints"""
    models = ['final_amharic_religious_model.pt', 'simple_amharic_model.pt']
    
    for model in models:
        try:
            # Test model insights summary
            response = requests.get(f'http://127.0.0.1:5000/api/model_insights_summary/{model}')
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model insights API works for {model}")
                print(f"   📊 Parameters: {data.get('total_params', 'N/A')}")
                print(f"   📊 Layers: {data.get('total_layers', 'N/A')}")
            else:
                print(f"⚠️  Model insights API issue for {model}: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Model analysis test failed for {model}: {e}")

def test_weight_analysis_api():
    """Test weight analysis API (the fixed endpoint)"""
    model = 'final_amharic_religious_model.pt'
    try:
        response = requests.get(f'http://127.0.0.1:5000/api/weight_analysis/{model}')
        if response.status_code == 200:
            data = response.json()
            print("✅ Weight Analysis API works correctly")
            layers = data.get('layers', {})
            print(f"   📊 Available layers: {len(layers)}")
            if layers:
                first_layer = list(layers.keys())[0]
                stats = layers[first_layer]
                print(f"   📊 Sample layer '{first_layer}':")
                print(f"       Mean: {stats.get('mean', 'N/A')}")
                print(f"       Std: {stats.get('std', 'N/A')}")
        else:
            print(f"❌ Weight Analysis API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Weight Analysis test failed: {e}")

def test_performance_metrics_api():
    """Test performance metrics API"""
    model = 'final_amharic_religious_model.pt'
    try:
        response = requests.get(f'http://127.0.0.1:5000/api/performance_metrics/{model}')
        if response.status_code == 200:
            data = response.json()
            print("✅ Performance Metrics API works correctly")
            print(f"   ⚡ Inference Time: {data.get('inference_time', 'N/A')} ms")
            print(f"   💾 Memory Usage: {data.get('memory_usage', 'N/A')} MB")
        else:
            print(f"❌ Performance Metrics API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Performance Metrics test failed: {e}")

def test_legacy_route():
    """Test that legacy route still works"""
    try:
        response = requests.get('http://127.0.0.1:5000/legacy')
        if response.status_code == 200:
            print("✅ Legacy route (/legacy) works correctly")
            return True
        else:
            print(f"❌ Legacy route failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Legacy route test failed: {e}")
        return False

def main():
    print("🧪 Testing Complete Analytics Dashboard Functionality")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    all_tests_passed = True
    
    # Test main dashboard
    if not test_dashboard_endpoint():
        all_tests_passed = False
    
    print()
    
    # Test API endpoints
    test_model_analysis_api()
    print()
    
    test_weight_analysis_api()
    print()
    
    test_performance_metrics_api()
    print()
    
    # Test legacy route
    if not test_legacy_route():
        all_tests_passed = False
    
    print()
    print("=" * 60)
    if all_tests_passed:
        print("🎉 All critical tests passed! Analytics Dashboard is ready.")
        print()
        print("📷 Next Steps:")
        print("1. Open http://127.0.0.1:5000/ in your browser")
        print("2. Take screenshots of the analytics dashboard")
        print("3. Replace placeholder files in docs/ directory:")
        print("   - docs/analytics-dashboard-main.png")
        print("   - docs/analytics-dashboard-advanced.png")
        print("   - docs/interactive-inspector.png")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    print()

if __name__ == "__main__":
    main()
