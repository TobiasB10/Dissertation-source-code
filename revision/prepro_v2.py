import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# harmonised preprocessing for the revision (applsci-4641827)
# changes from the original pipeline, each one asked for by reviewer 1
# 1. MAL-API-2019 traces are reduced to first-occurrence order (each api kept once, in the order it first appears)
#    which matches how MalbehavD-V1 is published
# 2. the vocab is built on the MalbehavD-V1 training split only
# 3. oov tokens are dropped (ids 2-4 are reserved but not used)
#    mapping them to an UNK token was tried and rejected, UNK never occurs in training so it becomes a spurious cue
# 4. identical MAL-API sequences are grouped (group_id) so S3 can exchange whole groups
#    and no copy of an exchanged sequence stays in the evaluation pool
# 5. MAL-API family labels are attached when the labels file is given

# run it like this:
# python prepro_v2.py --malbehav MalBehavD-V1-dataset.csv --malapi all_analysis_data.txt --out data_v2
# add --malapi-labels labels.csv to attach the family labels

# token ids 0 = PAD, 1 = CLS, 2 = UNK_UNI, 3 = UNK_BI, 4 = UNK_TRI, then unigrams, bigrams, trigrams
PAD, CLS, UNK_UNI, UNK_BI, UNK_TRI = 0, 1, 2, 3, 4
FIRST_ID = 5  # first unigram id


def first_occurrence(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def grams(seq, n):
    return ['-'.join(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def encode(seq, vocab):
    # oov tokens are dropped, not mapped to UNK
    # the vocab is built on the training split so an UNK token would never occur in training and its embedding
    # would stay untrained, then in S3 it would only appear in the exchanged (malicious) target samples
    # and turn into a spurious malware cue
    # dropping is also what the original pipeline did
    u = [vocab['uni'][x] for x in seq if x in vocab['uni']]
    b = [vocab['bi'][x] for x in grams(seq, 2) if x in vocab['bi']]
    t = [vocab['tri'][x] for x in grams(seq, 3) if x in vocab['tri']]
    return ' '.join(map(str, u)), ' '.join(map(str, b)), ' '.join(map(str, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--malbehav', required=True)
    ap.add_argument('--malapi', required=True)
    ap.add_argument('--malapi-labels', default=None,
                    help='optional: one family label per line, same order as the trace file')
    ap.add_argument('--out', default='data_v2')
    ap.add_argument('--split-seed', type=int, default=42)
    ap.add_argument('--split-from', default=None,
                    help='MalBehavD-V1-processed-clean.csv: reuse the published split by sha256 so S1 stays comparable')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # MalbehavD-V1
    df = pd.read_csv(a.malbehav).drop_duplicates('sha256', keep='first').reset_index(drop=True)
    cols = [str(i) for i in range(153) if str(i) in df.columns]
    df['seq'] = df[cols].apply(lambda r: [str(x).lower() for x in r if pd.notna(x)], axis=1)
    # MalbehavD is already in first-occurrence order but run it anyway so both datasets go through the same function
    df['seq'] = df['seq'].apply(first_occurrence)
    df['key'] = df['seq'].apply(lambda s: ' '.join(s))

    # 80/10/10 split over unique sequences, grouped and done per class like before
    # so all copies of a sequence end up in the same split
    g = df.groupby('key').agg(label=('labels', 'first'), n_labels=('labels', 'nunique'),
                              idx=('labels', lambda s: list(s.index))).reset_index()
    conflicting = int((g.n_labels > 1).sum())
    parts = {'train': [], 'val': [], 'test': []}
    for lab in (0, 1):
        sub = g[g.label == lab]
        tr, tmp = train_test_split(sub, test_size=0.2, random_state=a.split_seed)
        va, te = train_test_split(tmp, test_size=0.5, random_state=a.split_seed)
        for name, part in (('train', tr), ('val', va), ('test', te)):
            for idx in part.idx:
                parts[name].extend(idx)
    df['split'] = 'train'
    df.loc[parts['val'], 'split'] = 'val'
    df.loc[parts['test'], 'split'] = 'test'
    # or reuse the published split if one is given
    if a.split_from:
        old = pd.read_csv(a.split_from, usecols=['sha256', 'split']).set_index('sha256')['split']
        assert set(old.index) == set(df.sha256), 'split file does not cover the same samples'
        df['split'] = df.sha256.map(old)

    # vocab from the training split only
    train_seqs = df.loc[df.split == 'train', 'seq']
    uni = sorted({x for s in train_seqs for x in s})
    bi = sorted({x for s in train_seqs for x in grams(s, 2)})
    tri = sorted({x for s in train_seqs for x in grams(s, 3)})
    vocab = {'uni': {x: FIRST_ID + i for i, x in enumerate(uni)}}
    vocab['bi'] = {x: FIRST_ID + len(uni) + i for i, x in enumerate(bi)}
    vocab['tri'] = {x: FIRST_ID + len(uni) + len(bi) + i for i, x in enumerate(tri)}
    vocab_size = FIRST_ID + len(uni) + len(bi) + len(tri)

    enc = df['seq'].apply(lambda s: encode(s, vocab))
    df['uni'], df['bi'], df['tri'] = zip(*enc)
    df['length'] = df['seq'].apply(len)
    df[['sha256', 'labels', 'split', 'length', 'uni', 'bi', 'tri']].to_csv(
        os.path.join(a.out, 'malbehav_v2.csv'), index=False)

    # n-gram counts (reviewer asked how many only occur once)
    from collections import Counter
    bc = Counter(x for s in train_seqs for x in grams(s, 2))
    tc = Counter(x for s in train_seqs for x in grams(s, 3))

    # MAL-API-2019
    # family label is the last field on each line
    labels = None
    if a.malapi_labels:
        with open(a.malapi_labels, encoding='utf-8') as f:
            labels = [l.strip().split(',')[-1] for l in f if l.strip()]
    rows, raw_len, red_len, oov_tok, tot_tok, consec, pairs = [], [], [], 0, 0, 0, 0
    with open(a.malapi, encoding='utf-8') as f:
        for i, line in enumerate(f):
            toks = line.lower().split()
            if not toks:
                continue
            raw_len.append(len(toks))
            # consecutive repeats are counted on the raw trace before it gets reduced
            consec += sum(1 for x, y in zip(toks, toks[1:]) if x == y)
            pairs += max(len(toks) - 1, 0)
            s = first_occurrence(toks)
            red_len.append(len(s))
            tot_tok += len(s)
            oov_tok += sum(1 for x in s if x not in vocab['uni'])
            u, b, t = encode(s, vocab)
            rows.append({'row_id': i, 'length': len(s), 'key': ' '.join(s), 'uni': u, 'bi': b, 'tri': t})
    m = pd.DataFrame(rows)
    if labels is not None:
        if len(labels) == len(m) + 1:  # labels file has a header line
            labels = labels[1:]
        assert len(labels) == len(m), f'labels ({len(labels)}) do not match traces ({len(m)})'
        m['family'] = labels
    else:
        m['family'] = 'unknown'
    m['group_id'] = m.groupby('key').ngroup()
    m['labels'] = 1  # every MAL-API-2019 trace is malware
    m[['row_id', 'group_id', 'family', 'labels', 'length', 'uni', 'bi', 'tri']].to_csv(
        os.path.join(a.out, 'malapi_v2.csv'), index=False)

    # % of the MAL-API n-grams that are in the training vocab
    def cov(n, key):
        tot = hit = 0
        for k in m['key']:
            g = k.split() if n == 1 else grams(k.split(), n)
            tot += len(g)
            hit += sum(1 for x in g if x in vocab[key])
        return 100 * hit / max(tot, 1)

    stats = {
        'malbehav_rows': len(df), 'malbehav_unique_sequences': int(g.shape[0]),
        'malbehav_sequences_with_both_labels': conflicting,
        'split_rows': df.split.value_counts().to_dict(),
        'split_benign_share': {k: float((df[df.split == k].labels == 0).mean()) for k in ('train', 'val', 'test')},
        'vocab': {'uni': len(uni), 'bi': len(bi), 'tri': len(tri), 'total_incl_special': vocab_size},
        'bigrams_seen_once_in_train': sum(1 for v in bc.values() if v == 1),
        'trigrams_seen_once_in_train': sum(1 for v in tc.values() if v == 1),
        'malbehav_length': {'mean': float(df.length.mean()), 'median': float(df.length.median()), 'max': int(df.length.max())},
        'malapi_traces': len(m), 'malapi_unique_groups': int(m.group_id.nunique()),
        'malapi_raw_length': {'mean': float(np.mean(raw_len)), 'median': float(np.median(raw_len)), 'max': int(np.max(raw_len))},
        'malapi_reduced_length': {'mean': float(np.mean(red_len)), 'median': float(np.median(red_len)), 'max': int(np.max(red_len))},
        'malapi_consecutive_repeat_share_raw': consec / max(pairs, 1),
        'malapi_unigram_oov_share': oov_tok / max(tot_tok, 1),
        'malapi_coverage_pct': {'uni': cov(1, 'uni'), 'bi': cov(2, 'bi'), 'tri': cov(3, 'tri')},
        'oov_handling': 'dropped',
        'malapi_max_input_positions': int(max(1 + len(u.split()) + len(b.split()) + len(t.split()) for u, b, t in zip(m.uni, m.bi, m.tri))),
    }
    # save
    with open(os.path.join(a.out, 'vocab_v2.json'), 'w', encoding='utf-8') as f:
        json.dump({'special': {'PAD': PAD, 'CLS': CLS, 'UNK_UNI': UNK_UNI, 'UNK_BI': UNK_BI, 'UNK_TRI': UNK_TRI},
                   'vocab_size': vocab_size, **vocab}, f)
    with open(os.path.join(a.out, 'prepro_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


#   MAIN
if __name__ == '__main__':
    main()
