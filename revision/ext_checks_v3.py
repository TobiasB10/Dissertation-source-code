import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupKFold

# so experiments_v3 can be imported from the same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments_v3 as E

# supporting checks for the external corpus test (section 5.6 of applsci-4641827), cpu only and takes a few minutes
# (a) logistic regression trained and tested within the external corpus, grouped five-fold cv so identical sequences stay in the same fold
# (b) auc of the MalbehavD-V1 logistic regressions on the external corpus
# (c) MalbehavD-V1 test sequences cut to the external length distribution and classified by the same regressions
# (d) regressions trained on cut MalbehavD-V1 training sequences and scored on the external corpus
# (c) and (d) are averaged over 20 random length draws

# run it like this:
# python ext_checks_v3.py --data data_v2 --external external/oliveira_v2.csv --out results_extra/ext_checks.json

warnings.filterwarnings('ignore')

# random length draws for (c) and (d)
DRAWS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_v2')
    ap.add_argument('--external', default='external/oliveira_v2.csv')
    ap.add_argument('--out', default='results_extra/ext_checks.json')
    a = ap.parse_args()
    vocab = json.load(open(os.path.join(a.data, 'vocab_v2.json')))
    mb = pd.read_csv(os.path.join(a.data, 'malbehav_v2.csv'), keep_default_na=False)
    ext = pd.read_csv(a.external, keep_default_na=False)
    tr, te = mb[mb.split == 'train'], mb[mb.split == 'test']
    inv = {i: n for n, i in vocab['uni'].items()}  # token id back to api name

    # cut each sequence to a length drawn from the external corpus and rebuild its bigrams and trigrams
    # n-grams that are not in the vocab get dropped
    def cut(df, seed):
        rng = np.random.RandomState(seed)
        L = rng.choice(ext.length.values, size=len(df))
        rows = []
        for (_, r), n in zip(df.iterrows(), L):
            names = [inv[int(t)] for t in str(r.uni).split()][:n]
            bi = ['-'.join(names[i:i + 2]) for i in range(len(names) - 1)]
            tri = ['-'.join(names[i:i + 3]) for i in range(len(names) - 2)]
            rows.append({'labels': r.labels, 'uni': ' '.join(str(vocab['uni'][x]) for x in names),
                         'bi': ' '.join(str(vocab['bi'][x]) for x in bi if x in vocab['bi']),
                         'tri': ' '.join(str(vocab['tri'][x]) for x in tri if x in vocab['tri'])})
        return pd.DataFrame(rows)

    out = {}
    for kind in E.LR_KINDS:
        X, y, g = E.lr_features(ext, kind, vocab), ext.labels.values, ext.group_id.values

        # (a) grouped 5 fold cv within the external corpus
        sc = np.zeros(len(ext))
        for i, j in GroupKFold(n_splits=5).split(X, y, g):
            sc[j] = E.fit_lr(ext.iloc[i], kind, vocab).decision_function(X[j])
        out[kind] = {'within_auc': roc_auc_score(y, sc)}

        # (b) MalbehavD-V1 regression scored on the external corpus, plus its f1 on the uncut test set
        clf = E.fit_lr(tr, kind, vocab)
        out[kind]['mb_lr_ext_auc'] = roc_auc_score(y, clf.decision_function(X))
        out[kind]['mb_lr_test_f1'] = f1_score(te.labels.values, clf.predict(E.lr_features(te, kind, vocab)))

        # (c) and (d) over the random length draws, saved as (mean, min, max)
        f1c, aucd = [], []
        for d in range(DRAWS):
            tc = cut(te, 1000 + d)
            f1c.append(f1_score(tc.labels.values, clf.predict(E.lr_features(tc, kind, vocab))))
            aucd.append(roc_auc_score(y, E.fit_lr(cut(tr, 2000 + d), kind, vocab).decision_function(X)))
        out[kind]['cut_test_f1'] = (float(np.mean(f1c)), float(np.min(f1c)), float(np.max(f1c)))
        out[kind]['cut_train_ext_auc'] = (float(np.mean(aucd)), float(np.min(aucd)), float(np.max(aucd)))
        print(kind, out[kind], flush=True)

    # save
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({'external_median_length': float(np.median(ext.length)), 'test_median_length': float(np.median(te.length)),
               'draws': DRAWS, **out}, open(a.out, 'w'), indent=1)
    print('written', a.out)


#   MAIN
if __name__ == '__main__':
    main()
