import json, os

# Check corpus
with open('data/raw/raw_judgements.jsonl', encoding='utf-8') as f:
    lines = f.readlines()

sources = {}
for l in lines:
    d = json.loads(l)
    prefix = d['id'].split('_')[0]
    sources[prefix] = sources.get(prefix, 0) + 1

print(f'Total docs: {len(lines)}')
for k, v in sources.items():
    print(f'  {k}: {v} docs')

# Check API key
def load_env():
    for p in ['.env', '../.env']:
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        k, v = line.split('=', 1)
                        os.environ[k.strip()] = v.strip()
        except:
            pass

load_env()
key = os.environ.get('INDIANKANOON_API_KEY', '')
print(f'\nKey found: {bool(key)}')
print(f'Key preview: {key[:8]}...' if key else 'Key: NONE')