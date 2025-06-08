#!/usr/bin/env python3

import requests
import json

def test_weight_analysis():
    """Test the weight analysis API endpoint"""
    base_url = "http://127.0.0.1:5000"
    
    print("Testing weight analysis API...")
    
    try:
        # Test the weight analysis endpoint
        response = requests.get(f"{base_url}/api/weight_analysis/")
        print(f"Weight Analysis Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Weight analysis API working!")
            print(f"Response keys: {list(data.keys())}")
            
            # Check for weight statistics
            if 'weight_statistics' in data:
                print(f"✅ Weight statistics found with {len(data['weight_statistics'])} layers")
                
                # Check first layer for kurtosis and skewness
                first_layer = next(iter(data['weight_statistics'].values()))
                print(f"First layer stats: {first_layer}")
                
                if 'kurtosis' in first_layer and 'skewness' in first_layer:
                    print("✅ Kurtosis and skewness values present (scipy fix working)")
                else:
                    print("⚠️ Kurtosis/skewness missing")
            else:
                print("⚠️ No weight statistics in response")
                
        else:
            print(f"❌ Weight analysis failed: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask server")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_weight_analysis()
