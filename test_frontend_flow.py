import requests
import json

def test_frontend_flow():
    print("=== Testing Frontend Flow ===")
    
    # Step 1: Test model_insights_summary (this loads first)
    print("1. Testing model_insights_summary...")
    try:
        resp1 = requests.get('http://127.0.0.1:5000/api/model_insights_summary/simple_amharic_model.pt')
        data1 = resp1.json()
        print(f"Status: {resp1.status_code}")
        if 'error' in data1:
            print(f"ERROR in step 1: {data1['error']}")
            return
        print("✅ Step 1 success")
    except Exception as e:
        print(f"ERROR in step 1: {e}")
        return
    
    # Step 2: Test layer_insights (this loads second)
    print("\n2. Testing layer_insights...")
    try:
        resp2 = requests.get('http://127.0.0.1:5000/api/layer_insights/simple_amharic_model.pt')
        data2 = resp2.json()
        print(f"Status: {resp2.status_code}")
        if 'error' in data2:
            print(f"ERROR in step 2: {data2['error']}")
            return
        print("✅ Step 2 success")
        
        # Simulate frontend restructuring
        layer_insights = {
            'layer_details': data2.get('layer_details', []) if isinstance(data2, dict) else data2 if isinstance(data2, list) else [],
            'bottlenecks': data2.get('bottlenecks', []),
            'redundancies': data2.get('redundancies', []),
            'recommendations': data2.get('recommendations', [])
        }
        
        print(f"Restructured layer_insights keys: {list(layer_insights.keys())}")
        print(f"bottlenecks type: {type(layer_insights['bottlenecks'])}, value: {layer_insights['bottlenecks']}")
        print(f"recommendations type: {type(layer_insights['recommendations'])}, value: {layer_insights['recommendations']}")
        
    except Exception as e:
        print(f"ERROR in step 2: {e}")
        return
    
    # Step 3: Test weight_analysis 
    print("\n3. Testing weight_analysis...")
    try:
        resp3 = requests.get('http://127.0.0.1:5000/api/weight_analysis/simple_amharic_model.pt')
        data3 = resp3.json()
        print(f"Status: {resp3.status_code}")
        if 'error' in data3:
            print(f"ERROR in step 3: {data3['error']}")
            return
        print("✅ Step 3 success")
    except Exception as e:
        print(f"ERROR in step 3: {e}")
        return
    
    print("\n🎉 All API calls successful - the issue must be in JavaScript execution")

if __name__ == "__main__":
    test_frontend_flow()
