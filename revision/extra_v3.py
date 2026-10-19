# four extra checks for applsci-4641827 that came up in the internal review of the revision
# uses experiments_v3.py unchanged so the data, training code, seeds and exchange protocol match the main experiments

# --mode shortcut
# checks if the few-shot gain could just be a sparsity shortcut
# every exchanged MAL-API-2019 trace is malware and has sparse n-gram blocks (38.8% bigram, 1.2% trigram coverage)
# so an adapted model might learn sparse n-gram blocks = malware
# the n-gram blocks of the MalbehavD-V1 test inputs get thinned to the target density, then the false positive rate
# on the thinned benign rows is compared for models trained without target data (S1) and with n = 50 target traces (S3)
# if adaptation teaches the shortcut the S3 models fpr on thinned benign rows goes well above the S1 models
# retraining is deterministic so these are the same models as in the paper

# --mode lrsens
# the transformers use lr 5e-5 (a fine-tuning default) and the bilstm uses 1e-3
# retrains the three transformer configs at a higher lr (default 3e-4) over the same 20 seeds and records S1, S2 and S3 (n = 50)
# to see if the bilstm zero-shot lead depends on the transformer learning rate

# --mode minfreq
# reviewer 1 asked for a minimum frequency cut-off on n-gram tokens
# bigram and trigram tokens that appear in fewer than --minfreq (default 2) MalbehavD-V1 training samples are removed
# from every input (train, val, test and target) which is the same as deleting their embedding rows
# uni+bi and the full model are retrained over the same 20 seeds and evaluated in S1, S2 and S3 (n = 50)

# --mode external
# both reviewers asked about MAL-API-2019 having no benign traces
# every model is trained as in S1 (no target data) and as in S3 (n = 50 MAL-API-2019 traces) over the same 20 seeds
# then tested on an external corpus that has goodware (Oliveira's api call sequences, made by prep_oliveira.py)
# reports the false positive rate on its goodware and recall on its malware, per row and per distinct sequence
# if adapting to MAL-API-2019 taught it unfamiliar corpus = malware the fpr on the external goodware goes up after adaptation

# run it like this:
# python extra_v3.py --mode shortcut --models lr_count_all  (cpu, only takes minutes)
# python extra_v3.py --mode shortcut --models uni_bi uni_bi_tri  (gpu)
# python extra_v3.py --mode lrsens  (gpu)
# python extra_v3.py --mode minfreq --models uni_bi uni_bi_tri  (gpu)
# python extra_v3.py --mode external --external external/oliveira_v2.csv  (gpu, the lr models run on cpu)
# python extra_v3.py --mode summary
# results get appended to extra_results.jsonl in --outdir and finished runs are skipped on restart

import os
# for deterministic cuda, has to be set before torch gets imported
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, time, argparse, zlib
import numpy as np
import pandas as pd

# so experiments_v3 can be imported from the same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments_v3 as E

# random thinning draws for the shortcut test
DRAWS = 5


# keep probabilities that bring MalbehavD-V1 test n-gram density down to the MAL-API-2019 level
# density = in-vocab n-gram tokens per adjacent pair (or triple) in the sequence
def density_keep(mb, vocab):
    te = mb[mb.split == 'test']
    L = te.length.astype(int).values

    def dens(col, n):
        k = te[col].apply(lambda s: len(str(s).split())).values
        return k.sum() / np.clip(L - (n - 1), 0, None).sum()

    d_src = {'bi': dens('bi', 2), 'tri': dens('tri', 3)}
    d_tgt = {'bi': 0.3884, 'tri': 0.0121}  # table 10, harmonised MAL-API-2019
    return {c: min(1.0, d_tgt[c] / d_src[c]) for c in ('bi', 'tri')}, d_src


# drop each n-gram token with prob 1 - keep, seeded per draw so reruns get the same thinned copies
def thinned(te, keep, draw):
    rng = np.random.RandomState(zlib.crc32(f'thin_{draw}'.encode()))
    out = te.copy()
    for col in ('bi', 'tri'):
        out[col] = [' '.join(t for t in str(s).split() if rng.rand() < keep[col]) for s in te[col]]
    return out


# predictions are 0/1 so the mean on benign rows is the fpr and on malware rows the recall
def rates(p, y):
    p, y = np.asarray(p), np.asarray(y)
    return {'fpr': float(p[y == 0].mean()), 'recall': float(p[y == 1].mean())}


# runs already in the results file so they can be skipped on restart
def done_keys(path):
    if not os.path.exists(path):
        return set()
    return {(r['mode'], r['model'], r['seed'], r['budget'], r.get('lr')) for r in map(json.loads, open(path))}


def run_shortcut(mb, ma, vocab, a, res):
    is_adapt, is_test = E.target_pools(ma)
    tr, va, te = mb[mb.split == 'train'], mb[mb.split == 'val'], mb[mb.split == 'test']
    keep, d_src = density_keep(mb, vocab)
    print('source density', {k: round(v, 4) for k, v in d_src.items()}, 'keep', {k: round(v, 4) for k, v in keep.items()}, flush=True)

    # clean test set plus DRAWS thinned copies
    evals = {'clean': te}
    for d in range(DRAWS):
        evals[f'thin{d}'] = thinned(te, keep, d)
    y = te.labels.values

    done = done_keys(res)
    for model in a.models:
        for budget in (0, 50):
            for seed in a.seeds:
                # lr with no target data does not use the seed so it only runs once, saved as seed -1
                s = -1 if (model in E.LR_KINDS and budget == 0) else seed
                if ('shortcut', model, s, budget, None) in done:
                    continue
                t0 = time.time()

                # n=0 is S1 (no target data), n=50 is S3 with the target traces exchanged in
                if budget == 0:
                    trx, vax = tr, va
                else:
                    trx, vax, _ = E.exchange(mb, ma, is_adapt, seed, budget)
                if model in E.LR_KINDS:
                    clf = E.fit_lr(trx, model, vocab)
                    preds = {k: clf.predict(E.lr_features(df, model, vocab)) for k, df in evals.items()}
                else:
                    preds, _ = E.train_eval(model, trx, vax, evals, vocab['vocab_size'], seed, a)

                # thinned fpr and recall are averaged over the draws, the fpr of each draw is kept too
                clean = rates(preds['clean'], y)
                th = [rates(preds[f'thin{d}'], y) for d in range(DRAWS)]
                row = {'mode': 'shortcut', 'model': model, 'seed': s, 'budget': budget,
                       'fpr_clean': clean['fpr'], 'recall_clean': clean['recall'],
                       'fpr_thin': float(np.mean([r['fpr'] for r in th])),
                       'recall_thin': float(np.mean([r['recall'] for r in th])),
                       'fpr_thin_by_draw': [r['fpr'] for r in th], 'keep': keep,
                       'minutes': (time.time() - t0) / 60}
                with open(res, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f"shortcut {model} n={budget} seed {s}: FPR clean {row['fpr_clean']:.3f} thinned "
                      f"{row['fpr_thin']:.3f} ({row['minutes']:.1f} min)", flush=True)
                if s == -1:
                    break


def run_lrsens(mb, ma, vocab, a, res):
    is_adapt, is_test = E.target_pools(ma)
    tr, va, te = mb[mb.split == 'train'], mb[mb.split == 'val'], mb[mb.split == 'test']
    tp = ma[is_test]  # MAL-API-2019 test pool for S3
    done = done_keys(res)
    for model in a.models:
        # only the transformers are retrained here
        if model not in E.TRANSFORMERS:
            continue
        for seed in a.seeds:
            for budget in (0, 50):
                if ('lrsens', model, seed, budget, a.lr) in done:
                    continue
                t0 = time.time()

                # n=0 gives S1 on the source test set and S2 on all of MAL-API-2019
                if budget == 0:
                    preds, info = E.train_eval(model, tr, va, {'src': te, 'ma': ma}, vocab['vocab_size'], seed, a)
                    row = {'recall_all': float(preds['ma'].mean())}
                else:
                    # S3, recall on the test pool
                    trx, vax, _ = E.exchange(mb, ma, is_adapt, seed, budget)
                    preds, info = E.train_eval(model, trx, vax, {'src': te, 'tp': tp}, vocab['vocab_size'], seed, a)
                    row = {'recall_testpool': float(preds['tp'].mean())}

                # source f1 and fpr get recorded for every run
                sm = E.source_metrics(te, preds['src'])
                row = {'mode': 'lrsens', 'model': model, 'seed': seed, 'budget': budget, 'lr': a.lr, **info,
                       'src_f1': sm['src_f1'], 'src_fpr': sm['src_fpr'], **row,
                       'minutes': (time.time() - t0) / 60}
                with open(res, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f"lrsens {model} lr={a.lr} n={budget} seed {seed}: "
                      f"{ {k: round(v, 4) for k, v in row.items() if k.startswith(('recall', 'src_'))} } "
                      f"epochs {info['epochs']} ({row['minutes']:.1f} min)", flush=True)


def run_minfreq(mb, ma, vocab, a, res):
    from collections import Counter
    is_adapt, is_test = E.target_pools(ma)
    tr = mb[mb.split == 'train']

    # how many training samples each bigram and trigram token appears in
    df = {c: Counter(t for s in tr[c] for t in set(str(s).split())) for c in ('bi', 'tri')}
    keep = {c: {t for t, k in df[c].items() if k >= a.minfreq} for c in ('bi', 'tri')}
    print('n-gram tokens kept', {c: f'{len(keep[c])} of {len(df[c])}' for c in keep}, flush=True)

    # remove the rare tokens from every input, same as deleting their embedding rows
    def filt(d):
        d = d.copy()
        for c in ('bi', 'tri'):
            d[c] = [' '.join(t for t in str(s).split() if t in keep[c]) for s in d[c]]
        return d

    mbf, maf = filt(mb), filt(ma)
    trf, vaf, tef = mbf[mbf.split == 'train'], mbf[mbf.split == 'val'], mbf[mbf.split == 'test']
    tpf = maf[is_test]

    # same loop as lrsens but on the filtered data
    done = done_keys(res)
    for model in a.models:
        # only uni+bi and the full model are retrained for this one
        if model not in ('uni_bi', 'uni_bi_tri'):
            continue
        for seed in a.seeds:
            for budget in (0, 50):
                if ('minfreq', model, seed, budget, None) in done:
                    continue
                t0 = time.time()
                if budget == 0:
                    preds, info = E.train_eval(model, trf, vaf, {'src': tef, 'ma': maf}, vocab['vocab_size'], seed, a)
                    row = {'recall_all': float(preds['ma'].mean())}
                else:
                    trx, vax, _ = E.exchange(mbf, maf, is_adapt, seed, budget)
                    preds, info = E.train_eval(model, trx, vax, {'src': tef, 'tp': tpf}, vocab['vocab_size'], seed, a)
                    row = {'recall_testpool': float(preds['tp'].mean())}
                sm = E.source_metrics(tef, preds['src'])
                row = {'mode': 'minfreq', 'model': model, 'seed': seed, 'budget': budget, 'minfreq': a.minfreq,
                       'n_bi_kept': len(keep['bi']), 'n_tri_kept': len(keep['tri']), **info,
                       'src_f1': sm['src_f1'], 'src_fpr': sm['src_fpr'], **row, 'minutes': (time.time() - t0) / 60}
                with open(res, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f"minfreq {model} n={budget} seed {seed}: "
                      f"{ {k: round(v, 4) for k, v in row.items() if k.startswith(('recall', 'src_'))} } "
                      f"({row['minutes']:.1f} min)", flush=True)


def run_external(mb, ma, vocab, a, res):
    is_adapt, is_test = E.target_pools(ma)
    tr, va, te = mb[mb.split == 'train'], mb[mb.split == 'val'], mb[mb.split == 'test']
    ext = pd.read_csv(a.external, keep_default_na=False)
    y = ext.labels.values
    print(f'external corpus: {len(ext)} rows, goodware {int((y == 0).sum())}, malware {int((y == 1).sum())}', flush=True)

    # if --models is left as the default run every model (sequence models and lr)
    models = a.models if a.models != E.TRANSFORMERS else E.SEQ_MODELS + E.LR_KINDS
    done = done_keys(res)
    for model in models:
        for budget in (0, 50):
            for seed in a.seeds:
                # lr with no target data only runs once, same as the shortcut test
                s = -1 if (model in E.LR_KINDS and budget == 0) else seed
                if ('external', model, s, budget, None) in done:
                    if s == -1:
                        break
                    continue
                t0 = time.time()
                if budget == 0:
                    trx, vax = tr, va
                else:
                    trx, vax, _ = E.exchange(mb, ma, is_adapt, seed, budget)
                if model in E.LR_KINDS:
                    clf = E.fit_lr(trx, model, vocab)
                    p_ext = clf.predict(E.lr_features(ext, model, vocab))
                    p_src = clf.predict(E.lr_features(te, model, vocab))
                else:
                    preds, _ = E.train_eval(model, trx, vax, {'src': te, 'ext': ext}, vocab['vocab_size'], seed, a)
                    p_ext, p_src = preds['ext'], preds['src']
                p_ext = np.asarray(p_ext)

                # per distinct sequence, rows with the same group_id are averaged first
                g = pd.DataFrame({'g': ext.group_id.values, 'y': y, 'p': p_ext}).groupby('g').agg(y=('y', 'first'), p=('p', 'mean'))
                sm = E.source_metrics(te, p_src)
                row = {'mode': 'external', 'model': model, 'seed': s, 'budget': budget,
                       'ext_fpr': float(p_ext[y == 0].mean()), 'ext_recall': float(p_ext[y == 1].mean()),
                       'ext_fpr_groups': float(g[g.y == 0].p.mean()), 'ext_recall_groups': float(g[g.y == 1].p.mean()),
                       'src_f1': sm['src_f1'], 'src_fpr': sm['src_fpr'], 'minutes': (time.time() - t0) / 60}

                # keep the external predictions as well
                os.makedirs(os.path.join(a.outdir, 'preds'), exist_ok=True)
                np.save(os.path.join(a.outdir, 'preds', f'external_{model}_{s}_{budget}.npy'), p_ext.astype(np.uint8))
                with open(res, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f"external {model} n={budget} seed {s}: goodware FPR {row['ext_fpr']:.3f} malware recall "
                      f"{row['ext_recall']:.3f} ({row['minutes']:.1f} min)", flush=True)
                if s == -1:
                    break


# print a table for each mode that has results in the file
def summary(res):
    d = pd.DataFrame([json.loads(l) for l in open(res)])

    sc = d[d['mode'] == 'shortcut']
    if len(sc):
        print('\nSHORTCUT TEST: FPR (%) on MalbehavD-V1 benign test rows, clean vs n-gram blocks thinned to target density')
        g = sc.groupby(['model', 'budget'])[['fpr_clean', 'fpr_thin', 'recall_clean', 'recall_thin']].agg(['mean', 'std'])
        g = (g * 100).round(1)
        g['runs'] = sc.groupby(['model', 'budget']).size()
        print(g.to_string())

    ex = d[d['mode'] == 'external']
    if len(ex):
        print('\nEXTERNAL CORPUS WITH GOODWARE (%): FPR on its goodware, recall on its malware; per row and per distinct sequence')
        g = ex.groupby(['model', 'budget'])[['ext_fpr', 'ext_fpr_groups', 'ext_recall', 'ext_recall_groups', 'src_fpr']].agg(['mean', 'std'])
        g = (g * 100).round(1)
        g['runs'] = ex.groupby(['model', 'budget']).size()
        print(g.to_string())

    mf = d[d['mode'] == 'minfreq']
    if len(mf):
        print('\nMINIMUM-FREQUENCY CUT-OFF (%), mean ± sd over seeds')
        for (m, b), g in mf.groupby(['model', 'budget']):
            key = 'recall_all' if b == 0 else 'recall_testpool'
            print(f'{m:11s} n={b:3d} minfreq={g.minfreq.iloc[0]}: recall {100 * g[key].mean():.1f} ± {100 * g[key].std():.1f} | '
                  f'src F1 {100 * g.src_f1.mean():.1f} | FPR {100 * g.src_fpr.mean():.1f} | '
                  f'bigrams kept {g.n_bi_kept.iloc[0]} | trigrams kept {g.n_tri_kept.iloc[0]} | seeds {len(g)}')

    ls = d[d['mode'] == 'lrsens']
    if len(ls):
        print('\nLEARNING-RATE SENSITIVITY (%), mean ± sd over seeds')
        for (m, b, lr), g in ls.groupby(['model', 'budget', 'lr']):
            key = 'recall_all' if b == 0 else 'recall_testpool'
            print(f'{m:11s} n={b:3d} lr={lr:g}: recall {100 * g[key].mean():.1f} ± {100 * g[key].std():.1f} | '
                  f'src F1 {100 * g.src_f1.mean():.1f} | FPR {100 * g.src_fpr.mean():.1f} | '
                  f'epochs {g.epochs.min()}-{g.epochs.max()} | seeds {len(g)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['shortcut', 'lrsens', 'minfreq', 'external', 'summary'])
    ap.add_argument('--data', default='data_v2')
    ap.add_argument('--outdir', default='results_extra')
    ap.add_argument('--models', nargs='+', default=E.TRANSFORMERS)
    ap.add_argument('--seeds', type=int, nargs='+', default=E.SEEDS)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=None, help='transformer lr (shortcut: 5e-5; lrsens: 3e-4)')
    ap.add_argument('--lstm-lr', type=float, default=1e-3)
    ap.add_argument('--minfreq', type=int, default=2)
    ap.add_argument('--external', default='external/oliveira_v2.csv')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    res = os.path.join(a.outdir, 'extra_results.jsonl')
    if a.mode == 'summary':
        return summary(res)

    # transformer lr defaults to the 5e-5 from the paper so the retrained models match, lrsens defaults to 3e-4
    if a.lr is None:
        a.lr = 3e-4 if a.mode == 'lrsens' else 5e-5
    mb, ma, vocab = E.load(a.data)
    if a.mode == 'shortcut':
        run_shortcut(mb, ma, vocab, a, res)
    elif a.mode == 'minfreq':
        run_minfreq(mb, ma, vocab, a, res)
    elif a.mode == 'external':
        run_external(mb, ma, vocab, a, res)
    else:
        run_lrsens(mb, ma, vocab, a, res)
    summary(res)


#   MAIN
if __name__ == '__main__':
    main()
