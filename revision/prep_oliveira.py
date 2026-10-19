import argparse, json, os
import numpy as np
import pandas as pd

# prepares Oliveira's "Malware Analysis Datasets: API Call Sequences" (IEEE DataPort, 2019, 42,797 malware and
# 1,079 goodware traces) as a third external evaluation set for applsci-4641827

# each row of the published csv has the first 100 non-repeated consecutive api calls of the parent process
# (columns t_0 to t_99, integers 0-306), an md5 hash and a malware label (1 malware, 0 goodware)
# the dataset page lists the 307 api names but does not say how the integers map to them
# so this tests candidate mappings against the MalbehavD-V1 training vocab and stops unless page order is clearly the right one
# a correct mapping gives real Cuckoo call sequences where most adjacent pairs are known bigrams
# a wrong mapping gives pairs that are close to random

# then the chosen mapping is applied, names are lower cased, each sequence is reduced to first-occurrence order
# (same as MAL-API-2019) and encoded with the MalbehavD-V1 training vocab (oov tokens dropped)
# identical encoded sequences share a group_id

# run it like this:
# python prep_oliveira.py --csv <dataset csv> --api-list api_list_page_order.txt --vocab data_v2/vocab_v2.json --out external/oliveira_v2.csv


def first_occurrence(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def grams(seq, n):
    return ['-'.join(seq[i:i + n]) for i in range(len(seq) - n + 1)]


# share of adjacent pairs that are known MalbehavD-V1 bigrams (pairs of the same call are skipped)
def bigram_hit_rate(rows, names, vocab_bi):
    hit = tot = 0
    for r in rows:
        s = [names[i].lower() for i in r]
        for a, b in zip(s, s[1:]):
            if a != b:
                tot += 1
                hit += (a + '-' + b) in vocab_bi
    return hit / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--api-list', required=True)
    ap.add_argument('--vocab', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    # the 307 api names in the order the dataset page lists them
    names = [x.strip() for x in open(a.api_list, encoding='utf-8') if x.strip()]
    assert len(names) == 307 and len(set(names)) == 307, 'API list must have 307 distinct names'
    vocab = json.load(open(a.vocab))
    d = pd.read_csv(a.csv)
    tcols = [c for c in d.columns if c.startswith('t_')]
    tcols = sorted(tcols, key=lambda c: int(c[2:]))
    # label column should be malware but falls back to label/labels/class
    lab = 'malware' if 'malware' in d.columns else [c for c in d.columns if c.lower() in ('label', 'labels', 'class')][0]
    X = d[tcols].values.astype(int)
    # every integer has to be a valid index into the 307 names
    assert X.min() >= 0 and X.max() <= 306, (X.min(), X.max())
    print(f'rows {len(d)}; malware {int((d[lab] == 1).sum())}; goodware {int((d[lab] == 0).sum())}; '
          f'call columns {len(tcols)}; distinct hashes {d["hash"].nunique() if "hash" in d.columns else "n/a"}')

    # mapping check
    # score each candidate mapping on up to 3000 random rows and compare with 10 random mappings
    rng = np.random.RandomState(0)
    sample = X[rng.choice(len(X), size=min(3000, len(X)), replace=False)]
    vb = set(vocab['bi'])
    cands = {'page order': names, 'alphabetical': sorted(names), 'alphabetical, case-insensitive': sorted(names, key=str.lower),
             'page order reversed': names[::-1], 'page order shifted by +1': names[1:] + names[:1],
             'page order shifted by -1': names[-1:] + names[:-1], 'page order shifted by +2': names[2:] + names[:2]}
    res = {k: bigram_hit_rate(sample, v, vb) for k, v in cands.items()}
    rand = [bigram_hit_rate(sample, list(rng.permutation(names)), vb) for _ in range(10)]
    for k, v in res.items():
        print(f'  mapping "{k}": {100 * v:.1f}% of adjacent pairs are known MalbehavD-V1 bigrams')
    print(f'  random mappings: {100 * np.mean(rand):.1f}% on average (max {100 * np.max(rand):.1f}%)')
    best = max(res, key=res.get)
    others = max(v for k, v in res.items() if k != 'page order')
    # page order has to come out best, score over 5x the random average
    # and over 2x both the best random mapping and the best other candidate
    assert best == 'page order' and res['page order'] > 5 * np.mean(rand) and res['page order'] > 2 * max(max(rand), others), \
        'page order is not clearly the right mapping: STOP'
    print('mapping check passed: integers index the page list (0 = first name)')

    # encode
    # n_pairs and n_triples are how many bigrams and trigrams the reduced sequence has, used for coverage below
    U, B, T = vocab['uni'], vocab['bi'], vocab['tri']
    out = []
    for i, r in enumerate(X):
        s = first_occurrence([names[j].lower() for j in r])
        u = [U[x] for x in s if x in U]
        b = [B[x] for x in grams(s, 2) if x in B]
        t = [T[x] for x in grams(s, 3) if x in T]
        out.append({'row_id': i, 'hash': d['hash'].iloc[i] if 'hash' in d.columns else '', 'labels': int(d[lab].iloc[i]),
                    'length': len(s), 'uni': ' '.join(map(str, u)), 'bi': ' '.join(map(str, b)), 'tri': ' '.join(map(str, t)),
                    'n_calls': len(s), 'n_pairs': max(len(s) - 1, 0), 'n_triples': max(len(s) - 2, 0)})
    o = pd.DataFrame(out)
    o['group_id'] = o.groupby('uni').ngroup()
    o['family'] = np.where(o.labels == 1, 'malware', 'goodware')
    # sequences that show up with both labels
    n_ld = o.groupby('uni').labels.nunique()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    o.to_csv(a.out, index=False)
    # coverage = share of the bigrams or trigrams that are in the training vocab
    cov = lambda col, n: o[col].apply(lambda s: len(s.split())).sum() / o[n].sum()
    print(f'written {a.out}: {len(o)} rows, {o.group_id.nunique()} distinct sequences '
          f'({int((n_ld > 1).sum())} carry both labels)')
    for lb, g in o.groupby('labels'):
        print(f'  label {lb}: rows {len(g)}, distinct {g.group_id.nunique()}, median reduced length {g.length.median():.0f}, '
              f'max {g.length.max()}')
    print(f'  coverage: bigrams {100 * cov("bi", "n_pairs"):.1f}%, trigrams {100 * cov("tri", "n_triples"):.1f}%; '
          f'longest input {int((1 + o.uni.str.split().str.len() + o.bi.str.split().str.len().fillna(0) + o.tri.str.split().str.len().fillna(0)).max())} positions')


#   MAIN
if __name__ == '__main__':
    main()
