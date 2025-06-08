#!/usr/bin/env python3

import requests
import json

def test_complete_analytics_flow():
    """Test the complete analytics dashboard flow that simulates clicking 'Analyze model'"""
    base_url = "http://127.0.0.1:5000"
    model_file = "simple_amharic_model.pt"
    
    print("=== Testing Complete Analytics Dashboard Flow ===")
    print("This simulates what happens when 'Analyze model' button is clicked\n")
    
    # Step 1: Model insights summary (first API call)
    print("Step 1: Loading model insights summary...")
    try:
        response = requests.get(f"{base_url}/api/model_insights_summary/{model_file}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Model insights summary loaded successfully")
            print(f"   Model size: {data.get('model_size_mb', 'N/A')} MB")
            total_params = data.get('total_parameters', 'N/A')
            if isinstance(total_params, (int, float)):
                print(f"   Total parameters: {total_params:,}")
            else:
                print(f"   Total parameters: {total_params}")
        else:
            print(f"❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Step 2: Layer insights (second API call)
    print("\nStep 2: Loading layer insights...")
    try:
        response = requests.get(f"{base_url}/api/layer_insights/{model_file}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Layer insights loaded successfully")
            layer_details = data if isinstance(data, list) else data.get('layer_details', [])
            print(f"   Found {len(layer_details)} layers")
            if layer_details:
                print(f"   First layer: {layer_details[0].get('name', 'N/A')}")
        else:
            print(f"❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Step 3: Weight analysis (third API call - this was failing before)
    print("\nStep 3: Loading weight analysis...")
    try:
        response = requests.get(f"{base_url}/api/weight_analysis/{model_file}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Weight analysis loaded successfully")
            if 'weight_statistics' in data:
                layer_count = len(data['weight_statistics'])
                print(f"   Analyzed {layer_count} weight layers")
                
                # Check if scipy statistics are working
                first_layer_stats = next(iter(data['weight_statistics'].values()))
                has_kurtosis = 'kurtosis' in first_layer_stats
                has_skewness = 'skewness' in first_layer_stats
                print(f"   Scipy statistics available: {has_kurtosis and has_skewness}")
                
            if 'health_indicators' in data:
                health = data['health_indicators'].get('overall_health', 'unknown')
                print(f"   Model health: {health}")
        else:
            print(f"❌ Failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n🎉 SUCCESS: All analytics dashboard API calls completed successfully!")
    print("The JavaScript error 'Cannot read properties of undefined (reading 'length')' should now be resolved.")
    print("\nTo test manually:")
    print("1. Open http://127.0.0.1:5000/analytics_dashboard in your browser")
    print("2. Select 'simple_amharic_model.pt' from the dropdown")
    print("3. Click 'Analyze Model' button")
    print("4. The analysis should load without JavaScript errors")
    
    return True

if __name__ == "__main__":
    test_complete_analytics_flow()
