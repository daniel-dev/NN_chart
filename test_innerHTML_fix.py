#!/usr/bin/env python3
"""
Test script to verify that the innerHTML null reference error has been fixed
"""

import requests
import time
import json

def test_analytics_dashboard_scenario():
    """
    Test the exact scenario that was causing the innerHTML error:
    1. Select main model: final_amharic_religious_model.pt
    2. Select comparison model: simple_amharic_model.pt  
    3. Click "Analyze Model" button
    4. Verify all API calls work without errors
    """
    
    print("=" * 70)
    print("TESTING: innerHTML Error Fix Verification")
    print("Testing scenario: Both models selected + Analyze Model clicked")
    print("=" * 70)
    
    base_url = "http://127.0.0.1:5000"
    
    # Test models
    main_model = "final_amharic_religious_model.pt"
    comparison_model = "simple_amharic_model.pt"
    
    print(f"🎯 SCENARIO:")
    print(f"Main Model: {main_model}")
    print(f"Compare With: {comparison_model}")
    print(f"Action: Click 'Analyze Model' button")
    print()
    
    success_count = 0
    total_tests = 7
    
    # Test 1: Analytics Dashboard Page Load
    print("1. Testing analytics dashboard page load...")
    try:
        response = requests.get(f"{base_url}/analytics_dashboard")
        if response.status_code == 200:
            print("   ✅ Analytics dashboard loads successfully")
            success_count += 1
            
            # Check if all required DOM elements exist in the HTML
            html_content = response.text
            required_elements = [
                'id="modelOverview"',
                'id="layerAnalysis"', 
                'id="weightLayer"',
                'id="optimizationSuggestions"',
                'id="performanceMetrics"',
                'id="comparisonResults"',
                'id="healthCheck"',
                'id="trainingInsights"'
            ]
            
            missing_elements = []
            for element in required_elements:
                if element not in html_content:
                    missing_elements.append(element)
            
            if missing_elements:
                print(f"   ⚠️ Missing DOM elements: {missing_elements}")
            else:
                print("   ✅ All required DOM elements present")
        else:
            print(f"   ❌ Page load failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error loading page: {e}")
    
    # Test 2-4: Main model API endpoints (exactly as triggered by loadAnalysis())
    print(f"2. Testing main model API endpoints: {main_model}")
    
    # Test 2: Model insights summary
    try:
        response = requests.get(f"{base_url}/api/model_insights_summary/{main_model}")
        if response.status_code == 200:
            data = response.json()
            if 'error' not in data:
                print("   ✅ Model insights summary API working")
                success_count += 1
            else:
                print(f"   ❌ API error: {data['error']}")
        else:
            print(f"   ❌ Model insights API failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error calling model insights API: {e}")
    
    # Test 3: Layer insights
    try:
        response = requests.get(f"{base_url}/api/layer_insights/{main_model}")
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, (list, dict)) and 'error' not in data:
                print("   ✅ Layer insights API working")
                success_count += 1
            else:
                print(f"   ❌ Layer insights API error: {data.get('error', 'Invalid response')}")
        else:
            print(f"   ❌ Layer insights API failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error calling layer insights API: {e}")
    
    # Test 4: Weight analysis
    try:
        response = requests.get(f"{base_url}/api/weight_analysis/{main_model}")
        if response.status_code == 200:
            data = response.json()
            if 'error' not in data:
                print("   ✅ Weight analysis API working")
                success_count += 1
            else:
                print(f"   ❌ Weight analysis API error: {data['error']}")
        else:
            print(f"   ❌ Weight analysis API failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error calling weight analysis API: {e}")
    
    # Test 5: Model comparison (when both models are selected)
    print(f"3. Testing model comparison with both models...")
    try:
        comparison_url = f"{base_url}/api/model_comparison?models={main_model}&models={comparison_model}"
        response = requests.get(comparison_url)
        if response.status_code == 200:
            data = response.json()
            if 'error' not in data and main_model in data and comparison_model in data:
                print("   ✅ Model comparison API working")
                success_count += 1
            else:
                print(f"   ❌ Model comparison API error: {data.get('error', 'Invalid response')}")
        else:
            print(f"   ❌ Model comparison API failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error calling model comparison API: {e}")
    
    # Test 6: Performance metrics (optional, called by benchmarkModel())
    print(f"4. Testing performance metrics API...")
    try:
        response = requests.get(f"{base_url}/api/performance_metrics/{main_model}")
        if response.status_code == 200:
            data = response.json()
            if 'error' not in data:
                print("   ✅ Performance metrics API working")
                success_count += 1
            else:
                print(f"   ❌ Performance metrics API error: {data['error']}")
        else:
            print(f"   ❌ Performance metrics API failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error calling performance metrics API: {e}")
    
    # Test 7: Full page accessibility test
    print("5. Testing page accessibility after fixes...")
    try:
        response = requests.get(f"{base_url}/analytics_dashboard")
        if response.status_code == 200:
            html_content = response.text
            # Check if the JavaScript null checks are present
            null_checks = [
                'if (!modelOverviewElement)',
                'if (!layerSelect)',
                'if (!layerAnalysisElement)',
                'if (!optimizationElement)',
                'if (!performanceElement)',
                'if (!comparisonElement)',
                'if (!healthElement)',
                'if (!trainingElement)'
            ]
            
            checks_found = 0
            for check in null_checks:
                if check in html_content:
                    checks_found += 1
            
            if checks_found >= 6:  # Most null checks should be present
                print(f"   ✅ Null safety checks implemented ({checks_found}/{len(null_checks)} found)")
                success_count += 1
            else:
                print(f"   ⚠️ Limited null safety checks ({checks_found}/{len(null_checks)} found)")
        else:
            print(f"   ❌ Cannot verify null checks: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error verifying null checks: {e}")
    
    print()
    print("=" * 70)
    print("🔍 INNERHTML ERROR FIX SUMMARY:")
    print(f"✅ Tests Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED! The innerHTML error should be fixed.")
        print()
        print("💡 WHAT WAS FIXED:")
        print("1. Added null checks before all innerHTML operations")
        print("2. Protected against DOM elements not being found")
        print("3. Added proper error logging for debugging")
        print("4. Ensured elements exist before JavaScript manipulation")
        print()
        print("🧪 MANUAL VERIFICATION STEPS:")
        print("1. Open: http://127.0.0.1:5000/analytics_dashboard")
        print("2. Open browser dev tools (F12) → Console tab")
        print("3. Select 'final_amharic_religious_model.pt' as main model")
        print("4. Select 'simple_amharic_model.pt' as comparison model")
        print("5. Click 'Analyze Model' button")
        print("6. Verify NO 'Cannot set properties of null' errors in console")
        print("7. Check that data loads properly in all dashboard cards")
    else:
        print("⚠️ SOME TESTS FAILED - Further investigation needed")
        print("Check the specific failed tests above for details")
    
    print("=" * 70)
    return success_count == total_tests

if __name__ == "__main__":
    test_analytics_dashboard_scenario()
