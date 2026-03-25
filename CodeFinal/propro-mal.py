
import pandas as pd, json

# Load vocab and build lowercase lookup
with open('vocab_mappings.json') as f: v = json.load(f)
low2api = {k.lower(): k for k in v['api_to_int']}
a2i, b2i, t2i = v['api_to_int'], v['bigram_to_int'], v['trigram_to_int']

out = []
with open('all_analysis_data.txt', encoding='utf-8') as f:
    content = f.read()
    lines = content.strip().split('\n')
    print(f"Total lines: {len(lines)}")

    for i, line in enumerate(lines):
        if i % 5000 == 0: print(f"Processed {i} lines") # takes time so this shows the script is running and processing

        apis = [low2api.get(a) for a in line.strip().split()]
        apis = [a for a in apis if a]
        if len(apis) < 2: continue

        raw = [a2i[a] for a in apis]
        bi = [b2i[f"{apis[j]}-{apis[j + 1]}"] for j in range(len(apis) - 1) if f"{apis[j]}-{apis[j + 1]}" in b2i]
        tri = [t2i[f"{apis[j]}-{apis[j + 1]}-{apis[j + 2]}"] for j in range(len(apis) - 2) if
               f"{apis[j]}-{apis[j + 1]}-{apis[j + 2]}" in t2i]

        out.append({'labels': 1, 'raw_api_sequence': ' '.join(map(str, raw)),
                    'bigram_sequence': ' '.join(map(str, bi)), 'trigram_sequence': ' '.join(map(str, tri)),
                    'split': 'test'})

pd.DataFrame(out).to_csv('MalAPI2019-processed.csv', index=False)
print(f"Done: {len(out)} samples")