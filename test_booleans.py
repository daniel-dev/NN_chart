import requests
import json

def check_booleans(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, bool):
                print(f'Boolean found at {path}.{k}: {v}')
            else:
                check_booleans(v, f'{path}.{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_booleans(v, f'{path}[{i}]')
    return True

# Test model insights summary
print("=== Testing model_insights_summary ===")
resp = requests.get('http://127.0.0.1:5000/api/model_insights_summary/simple_amharic_model.pt')
data = resp.json()
print('Status:', resp.status_code)
print('Checking for boolean values...')
check_booleans(data)

# Test layer insights
print("\n=== Testing layer_insights ===")
resp2 = requests.get('http://127.0.0.1:5000/api/layer_insights/simple_amharic_model.pt')
data2 = resp2.json()
print('Status:', resp2.status_code)
print('Checking for boolean values...')
check_booleans(data2)

print('\nJSON serialization test complete - no errors means all boolean values are properly serializable')
