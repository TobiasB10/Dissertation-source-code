import argparse
import json
import os
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import f1_score

# dataset checks for applsci-4641827, cpu only, a few minutes
# works out the numbers in table 4 (sequence lengths and adjacent repeats), table 10 (vocab coverage)
# the share of uncovered raw MAL-API-2019 bigrams that are the same api twice (section 5.2)
# and the near-duplicate check in section 5.1 (test rows within two edits of a training sequence)
# needs the raw datasets, the data_v2 folder from prepro_v2.py and results_v3 from experiments_v3.py

# run it like this:
# python data_checks_v3.py --malbehav "MalBehavD-V1-dataset.csv" --malapi "all_analysis_data.txt" --data data_v2 --results results_v3


def length_stats(lengths):
    lengths = np.asarray(lengths)
    return {'mean': round(float(lengths.mean()), 1), 'median': float(np.median(lengths)),
            'p95': float(np.percentile(lengths, 95)), 'max': int(lengths.max())}


def grams(seq, n):
    return ['-'.join(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--malbehav', required=True)
    ap.add_argument('--malapi', required=True)
    ap.add_argument('--data', default='data_v2')
    ap.add_argument('--results', default='results_v3')
    ap.add_argument('--out', default='results_v3/data_checks.json')
    a = ap.parse_args()

    vocab = json.load(open(os.path.join(a.data, 'vocab_v2.json')))
    U, B, T = set(vocab['uni']), set(vocab['bi']), set(vocab['tri'])
    mb = pd.read_csv(os.path.join(a.data, 'malbehav_v2.csv'), keep_default_na=False)
    ma = pd.read_csv(os.path.join(a.data, 'malapi_v2.csv'), keep_default_na=False)
    out = {}

    # raw MalbehavD-V1, lower case, same rows as data_v2 (hash deduplicated)
    raw = pd.read_csv(a.malbehav, low_memory=False).drop_duplicates('sha256').set_index('sha256')
    cols = [str(i) for i in range(153)]
    mb_seqs = {h: [str(x).lower() for x in raw.loc[h, cols] if pd.notna(x)] for h in mb.sha256}

    # table 4, MalbehavD-V1 as published
    pairs = sum(max(len(s) - 1, 0) for s in mb_seqs.values())
    repeats = sum(sum(1 for x, y in zip(s, s[1:]) if x == y) for s in mb_seqs.values())
    out['table4_malbehav'] = {**length_stats([len(s) for s in mb_seqs.values()]), 'adjacent_repeats': repeats / pairs}

    # table 4 and 10, raw MAL-API-2019 traces
    lens = []
    pairs = repeats = 0
    tot = {1: 0, 2: 0, 3: 0}
    hit = {1: 0, 2: 0, 3: 0}
    oov_bi = oov_bi_self = 0
    with open(a.malapi, encoding='utf-8') as f:
        for line in f:
            s = line.lower().split()
            if not s:
                continue
            lens.append(len(s))
            for x, y in zip(s, s[1:]):
                pairs += 1
                repeats += x == y
                if x + '-' + y not in B:
                    oov_bi += 1
                    oov_bi_self += x == y
            for n, voc in ((1, U), (2, B), (3, T)):
                g = s if n == 1 else grams(s, n)
                tot[n] += len(g)
                hit[n] += sum(1 for x in g if x in voc)
    out['table4_malapi_raw'] = {**length_stats(lens), 'adjacent_repeats': repeats / pairs}
    out['table10_malapi_raw_coverage'] = {n: hit[n] / tot[n] for n in (1, 2, 3)}
    # section 5.2, most of the uncovered raw bigrams are the same api called twice in a row
    out['malapi_raw_uncovered_bigrams_that_repeat'] = oov_bi_self / oov_bi

    # table 4, MAL-API-2019 after first-occurrence reduction
    out['table4_malapi_reduced'] = {**length_stats(ma.length), 'adjacent_repeats': 0.0}

    # table 10, harmonised MAL-API-2019 coverage
    ntok = lambda col: ma[col].str.split().str.len().fillna(0)
    out['table10_malapi_harmonised_coverage'] = {
        1: ntok('uni').sum() / ma.length.sum(),
        2: ntok('bi').sum() / (ma.length - 1).clip(lower=0).sum(),
        3: ntok('tri').sum() / (ma.length - 2).clip(lower=0).sum()}

    # table 10, MalbehavD-V1 test split coverage by the training vocab
    tot = {1: 0, 2: 0, 3: 0}
    hit = {1: 0, 2: 0, 3: 0}
    for h in mb[mb.split == 'test'].sha256:
        s = mb_seqs[h]
        for n, voc in ((1, U), (2, B), (3, T)):
            g = s if n == 1 else grams(s, n)
            tot[n] += len(g)
            hit[n] += sum(1 for x in g if x in voc)
    out['table10_malbehav_test_coverage'] = {n: hit[n] / tot[n] for n in (1, 2, 3)}

    # table 10, median in-vocab tokens per training sample and per MAL-API-2019 trace
    tr = mb[mb.split == 'train']
    mtok = lambda df, col: float(df[col].str.split().str.len().fillna(0).median())
    out['table10_median_tokens_train'] = {c: mtok(tr, c) for c in ('uni', 'bi', 'tri')}
    out['table10_median_tokens_malapi'] = {c: mtok(ma, c) for c in ('uni', 'bi', 'tri')}
    out['table10_malapi_traces_with_no_trigram'] = float((ntok('tri') == 0).mean())

    # section 5.1 near-duplicate check
    # no test sequence is identical to a training one but some are only a couple of edits away
    seq = lambda s: [int(x) for x in s.split()]
    train_seqs = list({u: seq(u) for u in tr.uni}.values())
    te = mb[mb.split == 'test'].reset_index(drop=True)
    te['min_edits'] = [min(Levenshtein.distance(seq(u), t) for t in train_seqs) for u in te.uni]
    y = te.labels.values
    near = (te.min_edits <= 2).values
    out['neardup_within_two_edits'] = {'rows': int(near.sum()), 'malicious': int(y[near].sum()),
                                       'identical': int((te.min_edits == 0).sum())}
    keep = ~near

    # f1 with the near duplicates left out, from the saved s2 source predictions
    res = [json.loads(l) for l in open(os.path.join(a.results, 'results_v3.jsonl'))]
    seeds = sorted({r['seed'] for r in res if r['mode'] == 's2' and r['seed'] != -1})
    out['neardup_f1'] = {}
    for m in ['uni', 'uni_bi', 'uni_bi_tri', 'bilstm', 'lr_count_uni', 'lr_count_all']:
        files = [f's2src_{m}.npy'] if m.startswith('lr') else [f's2src_{m}_{s}.npy' for s in seeds]
        f_all = []
        f_keep = []
        for fn in files:
            p = np.load(os.path.join(a.results, 'preds', fn)).astype(int)
            f_all.append(f1_score(y, p))
            f_keep.append(f1_score(y[keep], p[keep]))
        out['neardup_f1'][m] = {'all': float(np.mean(f_all)), 'without_near_duplicates': float(np.mean(f_keep))}

    for k, v in out.items():
        print(k, v)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print('saved', a.out)



#   MAIN
if __name__ == '__main__':
    main()
