import os, json, requests

# 加载 .env
with open('.env', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip().strip("'\"")

key = os.environ.get('GPT55_API_KEY', '')
base_url = os.environ.get('GPT55_BASE_URL', 'https://cdn.coderelay.cn/v1').rstrip('/')
model = os.environ.get('GPT55_MODEL', 'gpt-5.5')
endpoint = f"{base_url}/chat/completions"

print(f'Testing: {endpoint}')
print(f'Model: {model}')
print(f'Key prefix: {key[:12]}...')

try:
    resp = requests.post(
        endpoint,
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        json={
            'model': model,
            'messages': [{'role': 'user', 'content': 'Reply with this JSON exactly: {"status": "ok"}'}],
            'max_tokens': 50,
            'response_format': {'type': 'json_object'}
        },
        timeout=30
    )
    print(f'HTTP Status: {resp.status_code}')
    data = resp.json()
    if resp.status_code == 200:
        print('SUCCESS: GPT-5.5 connection OK')
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        print('Response:', content[:300])
    else:
        print('ERROR:', json.dumps(data, ensure_ascii=False)[:500])
except Exception as e:
    print(f'Exception: {e}')
