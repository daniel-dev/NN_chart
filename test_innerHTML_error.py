#!/usr/bin/env python3
"""
Test the specific scenario that's causing the innerHTML error when both 
main model and comparison model are selected.
"""

import requests
import json

def test_model_comparison_scenario():
    """Test the scenario described by the user"""
    base_url = "http://127.0.0.1:5000"
    
    print("=" * 70)
    print("TESTING: Model Comparison Scenario")
    print("Reproducing: 'Cannot set properties of null (setting 'innerHTML')'")
    print("=" * 70)
    
    # Test models
    main_model = "final_amharic_religious_model.pt"
    comparison_model = "simple_amharic_model.pt"
    
    print(f"\n🎯 SCENARIO:")
    print(f"Main Model: {main_model}")
    print(f"Compare With: {comparison_model}")
    print(f"Action: Click 'Analyze Model' button")
    
    print(f"\n📋 TESTING INDIVIDUAL API CALLS:")
    
    # 1. Test main model analysis (what happens when 'Analyze Model' is clicked)
    print(f"\n1. Testing main model analysis: {main_model}")
    endpoints_to_test = [
        f'/api/model_insights_summary/{main_model}',
        f'/api/layer_insights/{main_model}', 
        f'/api/weight_analysis/{main_model}'
    ]
    
    for i, endpoint in enumerate(endpoints_to_test, 1):
        try:
            print(f"   1.{i} Testing: {endpoint}")
            response = requests.get(f'{base_url}{endpoint}')
            print(f"       Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    print(f"       ❌ API Error: {data['error']}")
                    return False
                else:
                    print(f"       ✅ Success - Keys: {list(data.keys())[:3]}...")
            else:
                print(f"       ❌ HTTP Error: {response.status_code}")
                print(f"       Response: {response.text[:200]}...")
                return False
                
        except Exception as e:
            print(f"       ❌ Exception: {e}")
            return False
    
    # 2. Test comparison model endpoints (in case they're called)
    print(f"\n2. Testing comparison model endpoints: {comparison_model}")
    for i, endpoint in enumerate(endpoints_to_test, 1):
        endpoint_comp = endpoint.replace(main_model, comparison_model)
        try:
            print(f"   2.{i} Testing: {endpoint_comp}")
            response = requests.get(f'{base_url}{endpoint_comp}')
            print(f"       Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    print(f"       ❌ API Error: {data['error']}")
                else:
                    print(f"       ✅ Success - Keys: {list(data.keys())[:3]}...")
            else:
                print(f"       ❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"       ❌ Exception: {e}")
    
    # 3. Test model comparison API (what happens when 'Compare Models' is clicked)
    print(f"\n3. Testing model comparison API:")
    try:
        comparison_url = f'{base_url}/api/model_comparison?models={main_model}&models={comparison_model}'
        print(f"   URL: {comparison_url}")
        response = requests.get(comparison_url)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'error' in data:
                print(f"   ❌ Comparison API Error: {data['error']}")
            else:
                print(f"   ✅ Comparison Success - Models: {list(data.keys())}")
        else:
            print(f"   ❌ Comparison HTTP Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"   ❌ Comparison Exception: {e}")
    
    print(f"\n🔍 ANALYSIS:")
    print("The 'Cannot set properties of null (setting innerHTML)' error suggests:")
    print("1. A JavaScript element with a specific ID doesn't exist in the DOM")
    print("2. The element is being referenced before the page fully loads")
    print("3. There's a timing issue between API calls and DOM manipulation")
    
    print(f"\n💡 POTENTIAL CAUSES:")
    print("- Missing HTML element IDs in the analytics dashboard")
    print("- JavaScript trying to update elements before they're created")
    print("- Race condition between model selection and DOM updates")
    print("- Async loading issues when both models are selected")
    
    print(f"\n📋 MANUAL TESTING STEPS:")
    print("1. Open: http://127.0.0.1:5000/analytics_dashboard")
    print("2. Open browser developer tools (F12)")
    print("3. Go to Console tab to see JavaScript errors")
    print(f"4. Select '{main_model}' in 'Select Model for Analysis'")
    print(f"5. Select '{comparison_model}' in 'Compare With (Optional)'")
    print("6. Click 'Analyze Model' button")
    print("7. Check console for the exact error and line number")
    
    return True

if __name__ == "__main__":
    test_model_comparison_scenario()
