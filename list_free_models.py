import requests
import json

response = requests.get("https://openrouter.ai/api/v1/models")
if response.status_code == 200:
    models = response.json().get('data', [])
    free_models = []
    for m in models:
        # Check if both prompt and completion prices are 0
        pricing = m.get('pricing', {})
        if pricing.get('prompt') == "0" and pricing.get('completion') == "0":
            free_models.append(m.get('id'))
    
    print("Currently available FREE models on OpenRouter:")
    for fm in sorted(free_models):
        print(f"- {fm}")
else:
    print(f"Failed to fetch models: {response.status_code}")
