import requests
import json

# Test layer insights endpoint
print("=== Testing layer_insights API ===")
resp = requests.get('http://127.0.0.1:5000/api/layer_insights/simple_amharic_model.pt')
data = resp.json()

print(f'Status: {resp.status_code}')
print('\nResponse structure:')
print(json.dumps(data, indent=2))

# Check if bottlenecks and recommendations exist
print('\n=== Checking for problematic arrays ===')
if 'bottlenecks' in data:
    print(f'bottlenecks exists: {data["bottlenecks"]}')
else:
    print('bottlenecks does NOT exist in response')

if 'recommendations' in data:
    print(f'recommendations exists: {data["recommendations"]}')
else:
    print('recommendations does NOT exist in response')

# Check the actual structure we're trying to access
print('\n=== Checking currentAnalysis.layer_insights structure ===')
print('Available keys in response:', list(data.keys()))
