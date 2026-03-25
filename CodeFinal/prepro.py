import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import json

warnings.filterwarnings('ignore')

# file paths
INPUT = "MalBehavD-V1-dataset.csv"
OUTPUT = "MalBehavD-V1-processed-clean.csv"


df = pd.read_csv(INPUT)
print(f"{len(df)} samples")

# remove duplicates by sha256 hash
df_clean = df.drop_duplicates(subset=['sha256'], keep='first').copy()
print(f"after removing duplicates: {len(df_clean)}")


def extract_api_sequence(row):
    # the dataset has api calls in columns named 0, 1, 2, ... 152
    # weird format but ok
    apis = []
    for i in range(153):
        col = str(i)
        if col in row.index and pd.notna(row[col]):
            apis.append(row[col])
    return apis



df_clean['api_list'] = df_clean.apply(extract_api_sequence, axis=1)

# build vocabulary
# need non-overlapping ranges for raw apis bigrams  trigrams
# 0 = padding token, 1 = CLS token so raw apis start at 2
all_apis = set()
for api_list in df_clean['api_list']:
    all_apis.update(api_list)

api_to_int = {api: i + 2 for i, api in enumerate(sorted(all_apis))}
RAW_MAX = len(api_to_int) + 1
print(f"  {len(api_to_int)} unique apis")

# bigrams
all_bigrams = set()
for api_list in df_clean['api_list']:
    for i in range(len(api_list) - 1):
        bg = f"{api_list[i]}-{api_list[i + 1]}"
        all_bigrams.add(bg)

BIGRAM_OFFSET = RAW_MAX + 1
bigram_to_int = {bg: i + BIGRAM_OFFSET for i, bg in enumerate(sorted(all_bigrams))}
print(f"  {len(bigram_to_int)} unique bigrams")

# trigrams
all_trigrams = set()
for api_list in df_clean['api_list']:
    for i in range(len(api_list) - 2):
        tg = f"{api_list[i]}-{api_list[i + 1]}-{api_list[i + 2]}"
        all_trigrams.add(tg)

TRIGRAM_OFFSET = BIGRAM_OFFSET + len(bigram_to_int)
trigram_to_int = {tg: i + TRIGRAM_OFFSET for i, tg in enumerate(sorted(all_trigrams))}
print(f"  {len(trigram_to_int)} unique trigrams")


# convert to integers
def to_ints(api_list):
    return [api_to_int[api] for api in api_list]


def to_bigrams(api_list):
    if len(api_list) < 2:
        return []
    result = []
    for i in range(len(api_list) - 1):
        bg = f"{api_list[i]}-{api_list[i + 1]}"
        result.append(bigram_to_int[bg])
    return result


def to_trigrams(api_list):
    if len(api_list) < 3:
        return []
    result = []
    for i in range(len(api_list) - 2):
        tg = f"{api_list[i]}-{api_list[i + 1]}-{api_list[i + 2]}"
        result.append(trigram_to_int[tg])
    return result



df_clean['raw_ints'] = df_clean['api_list'].apply(to_ints)
df_clean['bigram_ints'] = df_clean['api_list'].apply(to_bigrams)
df_clean['trigram_ints'] = df_clean['api_list'].apply(to_trigrams)

# convert to space separated strings for csv
df_clean['raw_api_sequence'] = df_clean['raw_ints'].apply(lambda x: ' '.join(map(str, x)))
df_clean['bigram_sequence'] = df_clean['bigram_ints'].apply(lambda x: ' '.join(map(str, x)) if x else '')
df_clean['trigram_sequence'] = df_clean['trigram_ints'].apply(lambda x: ' '.join(map(str, x)) if x else '')

# combined sequence for splitting
df_clean['combined'] = df_clean['raw_api_sequence'] + ' ' + df_clean['bigram_sequence'] + ' ' + df_clean[
    'trigram_sequence']


# splitting
# if same sequence appears multiple times all copies need to be in same split
# otherwise model can just memorize

def split_data(df):
    # group by unique sequences
    groups = df.groupby('combined')

    seq_data = []
    for seq, group in groups:
        seq_data.append({
            'seq': seq,
            'label': group['labels'].iloc[0],
            'idxs': group.index.tolist()
        })

    seq_df = pd.DataFrame(seq_data)

    # split each class seperately to keep class balance
    mal = seq_df[seq_df['label'] == 1]
    ben = seq_df[seq_df['label'] == 0]

    # 80/10/10 split
    mal_train, mal_tmp = train_test_split(mal, test_size=0.2, random_state=42)
    mal_val, mal_test = train_test_split(mal_tmp, test_size=0.5, random_state=42)

    ben_train, ben_tmp = train_test_split(ben, test_size=0.2, random_state=42)
    ben_val, ben_test = train_test_split(ben_tmp, test_size=0.5, random_state=42)

    # get indices
    train_idx = []
    val_idx = []
    test_idx = []

    for idxs in pd.concat([mal_train, ben_train])['idxs']:
        train_idx.extend(idxs)
    for idxs in pd.concat([mal_val, ben_val])['idxs']:
        val_idx.extend(idxs)
    for idxs in pd.concat([mal_test, ben_test])['idxs']:
        test_idx.extend(idxs)

    return train_idx, val_idx, test_idx



train_idx, val_idx, test_idx = split_data(df_clean)

df_clean['split'] = 'train'  # default
df_clean.loc[val_idx, 'split'] = 'val'
df_clean.loc[test_idx, 'split'] = 'test'

print(f"  train: {len(train_idx)}")
print(f"  val: {len(val_idx)}")
print(f"  test: {len(test_idx)}")

# save
print(f"\nsaved to {OUTPUT}")
out_df = df_clean[['sha256', 'labels', 'raw_api_sequence', 'bigram_sequence', 'trigram_sequence', 'split']]
out_df.to_csv(OUTPUT, index=False)

# save vocab mappings need these for cross-dataset evaluation later
print("vocab mappings saved")
vocab = {
    'api_to_int': api_to_int,
    'bigram_to_int': bigram_to_int,
    'trigram_to_int': trigram_to_int,
    'RAW_MAX': RAW_MAX,
    'BIGRAM_OFFSET': BIGRAM_OFFSET,
    'TRIGRAM_OFFSET': TRIGRAM_OFFSET
}
with open('vocab_mappings.json', 'w') as f:
    json.dump(vocab, f)

print("\ndone")