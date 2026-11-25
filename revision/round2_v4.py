"""
Second-round analyses for applsci-4641827 (Reviewer 1, second report).

Modes (run in this order):
  perturb   S6 over all 20 seeds. Same perturbed inputs as perturb_v3.py (asserted), plus a deletion-only
            control for the benign preamble: the malicious part loses exactly the calls the preamble would
            delete, but no preamble is prepended. Saves hard predictions AND malicious-class scores, so
            threshold-free metrics (AUC, balanced accuracy) can be computed. Also records, for every perturbed
            row, how many calls of the original sequence survive first-occurrence reduction.
  ngram     Zero-shot (S2) recall on MAL-API-2019 against each trace's share of known n-grams, from the
            saved S2 predictions in results_v3/preds (no training).
  mapping   Sanity check of the inferred integer-to-name mapping of the external corpus: rank correlation of
            API frequency profiles against MalbehavD-V1 training samples, under the inferred mapping and the
            alternatives (no training).
  summary   Tables and the pre-registered paired comparisons -> round2_v4.md

Usage:
  python round2_v4.py --mode perturb --data data_v2 --malbehav <MalBehavD-V1-dataset.csv>
                      --malapi <all_analysis_data.txt> --outdir results_perturb20
  python round2_v4.py --mode ngram   --data data_v2 --s2preds results_v3/preds --outdir results_perturb20
  python round2_v4.py --mode mapping --data data_v2 --malbehav <MalBehavD-V1-dataset.csv>
                      --ext-csv <Oliveira csv> --api-list <api_list_page_order.txt> --outdir results_perturb20
  python round2_v4.py --mode summary --outdir results_perturb20 [--old-preds results_perturb/preds]
"""
import os, sys, json, time, argparse, copy, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments_v3 as E
import perturb_v3 as P

METRICS = ['recall', 'fpr', 'bal_acc', 'auc', 'recall_groups', 'fpr_groups', 'bal_acc_groups', 'auc_groups']


# ----------------------------------------------------------------- shared
def load_names(malbehav_csv, mb):
    raw = pd.read_csv(malbehav_csv, low_memory=False).drop_duplicates('sha256', keep='first').set_index('sha256')
    cols = [str(i) for i in range(153) if str(i) in raw.columns]
    return {h: P.first_occurrence([str(x).lower() for x in raw.loc[h, cols] if pd.notna(x)]) for h in mb.sha256}


def check_names(names, mb, vocab):
    re = [P.encode(names[h], vocab) for h in mb.sha256]
    assert all((u, b, t) == (r.uni, r.bi, r.tri) for (u, b, t), (_, r) in zip(re, mb.iterrows())), 'name/ID mismatch'


def preamble_parts(seq, level, rng, benign_seqs):
    """Identical random draws to perturb_v3.perturb(kind='preamble'); returns the preamble too."""
    pre = []
    while len(pre) < level:
        for x in benign_seqs[rng.randint(len(benign_seqs))]:
            if x not in pre:
                pre.append(x)
            if len(pre) == level:
                break
    full = P.first_occurrence(pre + list(seq))
    pset = set(pre)
    delonly = [x for x in seq if x not in pset]          # seq is already in first-occurrence form
    return full, delonly


def v3_rng(kind, level, draw):
    # exactly as in the perturb_v3.py that produced results_perturb (line 148)
    return np.random.RandomState(zlib.crc32(f'{kind}_{level}_{draw}'.encode()))


def build_conditions(mb, names, vocab, malapi_txt):
    known = set(x for s in names.values() for x in s)
    with open(malapi_txt, encoding='utf-8') as f:
        for line in f:
            known.update(line.lower().split())
    variants = P.variant_map(known)
    tr, te = mb[mb.split == 'train'], mb[mb.split == 'test']
    benign_tr = [names[h] for h in tr[tr.labels == 0].sha256]
    from collections import Counter
    fc = Counter(x for s in benign_tr for x in s)
    filler = sorted(fc); fprob = np.array([fc[x] for x in filler], float); fprob /= fprob.sum()

    # distinct clean test sequences (label included), for per-sequence metrics
    keys = [(tuple(names[h]), int(l)) for h, l in zip(te.sha256, te.labels)]
    gid = {k: i for i, k in enumerate(dict.fromkeys(keys))}

    def row(cond, kind, level, draw, i, r, s, n_clean):
        u, b, t = P.encode(s, vocab)
        return {'cond': cond, 'kind': kind, 'level': level, 'draw': draw, 'idx': i, 'group': gid[keys[i]],
                'labels': int(r.labels), 'n_clean': n_clean, 'n_after': len(s), 'uni': u, 'bi': b, 'tri': t}

    rows = [row('clean', 'clean', 0, 0, i, r, names[r.sha256], len(names[r.sha256]))
            for i, (_, r) in enumerate(te.iterrows())]
    # 1) exactly the perturb_v3 conditions, in the same order
    for kind, levels in P.LEVELS.items():
        for level in levels:
            for draw in range(P.DRAWS):
                rng = v3_rng(kind, level, draw)
                for i, (_, r) in enumerate(te.iterrows()):
                    s = P.perturb(names[r.sha256], kind, level, rng, filler, fprob, variants, benign_tr)
                    rows.append(row(f'{kind}_{level}_{draw}', kind, level, draw, i, r, s, len(names[r.sha256])))
    n_v3 = len(rows)
    # 2) deletion-only control, same preamble draws (asserted against the v3 preamble rows)
    v3_pre = {(x['level'], x['draw'], x['idx']): (x['uni'], x['bi'], x['tri']) for x in rows if x['kind'] == 'preamble'}
    survive = {}
    for level in P.LEVELS['preamble']:
        for draw in range(P.DRAWS):
            rng = v3_rng('preamble', level, draw)
            for i, (_, r) in enumerate(te.iterrows()):
                full, delonly = preamble_parts(names[r.sha256], level, rng, benign_tr)
                assert P.encode(full, vocab) == v3_pre[(level, draw, i)], 'preamble draws differ from perturb_v3'
                rows.append(row(f'preamble_delonly_{level}_{draw}', 'preamble_delonly', level, draw, i, r, delonly,
                                len(names[r.sha256])))
    return pd.DataFrame(rows), n_v3, variants


def train_eval_scores(model_name, tr, va, eval_sets, vocab_size, seed, a):
    """experiments_v3.train_eval, unchanged except that it also returns the malicious-class probability."""
    torch, nn, DirIntBERT, BiLSTM = E.torch_bits()
    from transformers import get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    E.set_seed(seed, torch)
    gen = torch.Generator().manual_seed(seed)
    model = (BiLSTM if model_name == 'bilstm' else DirIntBERT)(vocab_size).to(dev)
    lr = a.lstm_lr if model_name == 'bilstm' else a.lr
    xtr, ytr = E.seqs(tr, model_name), tr.labels.tolist()
    xva, yva = E.seqs(va, model_name), va.labels.tolist()
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], 'weight_decay': 0.0}], lr=lr)
    steps = ((len(xtr) + a.batch - 1) // a.batch) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossf = nn.CrossEntropyLoss()

    def predict(xs, scores=False):
        model.eval(); out, sc = [], []
        with torch.no_grad():
            for ids, mask, _ in E.batches(xs, [0] * len(xs), 64, False, None, torch):
                logits = model(ids.to(dev), mask.to(dev))
                out.append(logits.argmax(1).cpu().numpy())
                sc.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        p = np.concatenate(out) if out else np.zeros(0, dtype=int)
        return (p, np.concatenate(sc) if sc else np.zeros(0)) if scores else p

    best, best_state, patience, epochs_run = -1.0, None, 0, 0
    for ep in range(a.epochs):
        model.train(); epochs_run = ep + 1
        for ids, mask, y in E.batches(xtr, ytr, a.batch, True, gen, torch):
            opt.zero_grad()
            loss = lossf(model(ids.to(dev), mask.to(dev)), y.to(dev))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        f1 = f1_score(yva, predict(xva), zero_division=0)
        if f1 > best:
            best, patience = f1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= a.patience:
                break
    model.load_state_dict(best_state)
    out = {k: predict(E.seqs(df, model_name), scores=True) for k, df in eval_sets.items()}
    return out, {'val_f1': best, 'epochs': epochs_run}


def condition_metrics(g, p, s):
    """g: rows of one (kind, level) condition; p hard predictions; s scores (same order)."""
    from sklearn.metrics import roc_auc_score
    g = g.assign(p=p, s=s)
    out = {}
    mal, ben = g[g.labels == 1], g[g.labels == 0]
    out['recall'], out['fpr'] = float(mal.p.mean()), float(ben.p.mean())
    out['bal_acc'] = (out['recall'] + 1 - out['fpr']) / 2
    out['auc'] = float(np.mean([roc_auc_score(d.labels, d.s) for _, d in g.groupby('draw')]))
    gg = g.groupby(['group', 'labels'], as_index=False)[['p', 's']].mean()   # each distinct sequence counts once
    out['recall_groups'] = float(gg[gg.labels == 1].p.mean())
    out['fpr_groups'] = float(gg[gg.labels == 0].p.mean())
    out['bal_acc_groups'] = (out['recall_groups'] + 1 - out['fpr_groups']) / 2
    out['auc_groups'] = float(roc_auc_score(gg.labels, gg.s))
    out['recall_by_draw'] = [float(x) for x in mal.groupby('draw').p.mean()]
    return out


# ----------------------------------------------------------------- modes
def run_perturb(a, train_fn=None):
    train_fn = train_fn or train_eval_scores
    os.makedirs(os.path.join(a.outdir, 'preds'), exist_ok=True)
    mb, ma, vocab = E.load(a.data)
    names = load_names(a.malbehav, mb)
    check_names(names, mb, vocab)
    ev, n_v3, variants = build_conditions(mb, names, vocab, a.malapi)
    ev.drop(columns=['uni', 'bi', 'tri']).to_csv(os.path.join(a.outdir, 'conditions.csv'), index=False)
    json.dump({'n_eval_rows': len(ev), 'n_rows_as_perturb_v3': n_v3, 'n_test_rows': int((ev.kind == 'clean').sum()),
               'n_distinct_test_sequences': int(ev[ev.kind == 'clean'].group.nunique())},
              open(os.path.join(a.outdir, 'meta.json'), 'w'), indent=1)
    print(f'eval rows {len(ev)} (first {n_v3} identical to perturb_v3); '
          f'distinct test sequences {ev[ev.kind == "clean"].group.nunique()}', flush=True)

    res = os.path.join(a.outdir, 'perturb20_results.jsonl')
    done = set()
    if os.path.exists(res):
        done = {(r['model'], r['seed']) for r in map(json.loads, open(res))}

    def record(model, seed, p, s, extra):
        np.save(os.path.join(a.outdir, 'preds', f'{model}_{seed}.npy'), np.asarray(p).astype(np.uint8))
        np.save(os.path.join(a.outdir, 'preds', f'{model}_{seed}_score.npy'), np.asarray(s).astype(np.float32))
        with open(res, 'a', encoding='utf-8') as f:
            for (kind, level), g in ev.groupby(['kind', 'level']):
                m = condition_metrics(g, np.asarray(p)[g.index], np.asarray(s)[g.index])
                f.write(json.dumps({'model': model, 'seed': seed, 'kind': kind, 'level': level, **m, **extra}) + '\n')

    tr, va = mb[mb.split == 'train'], mb[mb.split == 'val']
    for kind in E.LR_KINDS:
        if (kind, -1) in done: continue
        clf = E.fit_lr(tr, kind, vocab)
        X = E.lr_features(ev, kind, vocab)
        record(kind, -1, clf.predict(X), clf.predict_proba(X)[:, 1], {})
        print(kind, 'done', flush=True)
    for seed in a.seeds:
        for model in a.models:
            if (model, seed) in done: continue
            t0 = time.time()
            out, info = train_fn(model, tr, va, {'ev': ev}, vocab['vocab_size'], seed, a)
            p, s = out['ev']
            record(model, seed, p, s, {'epochs': info['epochs'], 'minutes': (time.time() - t0) / 60})
            print(f'{model} seed {seed} done ({(time.time() - t0) / 60:.1f} min)', flush=True)


def run_ngram(a):
    mb, ma, vocab = E.load(a.data)
    unk = {int(v) for k, v in vocab.get('special', {}).items() if k.upper().startswith('UNK')}
    ntok = lambda s: sum(1 for t in str(s).split() if int(t) not in unk)      # known n-grams only
    L = ma.length.astype(int)
    nb, nt = ma.bi.apply(ntok), ma.tri.apply(ntok)
    share_bi = np.where(L > 1, nb / (L - 1).clip(lower=1), np.nan)
    share_tri = np.where(L > 2, nt / (L - 2).clip(lower=1), np.nan)
    bins = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.0]
    lab = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    bbin = pd.cut(pd.Series(share_bi), bins, labels=lab)
    has_tri = nt > 0
    out = {'n_traces': int(len(ma)), 'bigram_bins': {k: int(v) for k, v in bbin.value_counts().sort_index().items()},
           'with_known_trigram': int(has_tri.sum()), 'without_known_trigram': int((~has_tri).sum()), 'models': {}}
    for model in E.TRANSFORMERS + ['bilstm'] + E.LR_KINDS:
        files = [os.path.join(a.s2preds, f's2_{model}.npy')] if model in E.LR_KINDS else \
                [os.path.join(a.s2preds, f's2_{model}_{s}.npy') for s in E.SEEDS]
        files = [f for f in files if os.path.exists(f)]
        if not files: continue
        P_ = np.stack([np.load(f).astype(float) for f in files])        # seeds x traces
        assert P_.shape[1] == len(ma), 'prediction length does not match malapi_v2.csv'
        per_seed = lambda mask: [float(x[mask].mean()) for x in P_]
        m = {'n_seeds': len(files), 'recall_all': float(P_.mean()),
             'recall_no_known_trigram': float(np.mean(per_seed(~has_tri.values))),
             'recall_with_known_trigram': float(np.mean(per_seed(has_tri.values))),
             'recall_by_bigram_share': {k: float(np.mean(per_seed((bbin == k).values))) for k in lab if (bbin == k).sum()}}
        from scipy.stats import spearmanr
        rate = P_.mean(0); ok = ~np.isnan(share_bi)
        m['spearman_rate_vs_bigram_share'] = float(spearmanr(share_bi[ok], rate[ok]).correlation)
        out['models'][model] = m
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(out, open(os.path.join(a.outdir, 'ngram_recall.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


def run_mapping(a):
    from scipy.stats import spearmanr
    mb, ma, vocab = E.load(a.data)
    names_mb = load_names(a.malbehav, mb)
    tr = mb[mb.split == 'train']
    page = [x.strip() for x in open(a.api_list, encoding='utf-8') if x.strip()]
    assert len(page) == 307 and len(set(page)) == 307
    d = pd.read_csv(a.ext_csv)
    tcols = sorted([c for c in d.columns if c.startswith('t_')], key=lambda c: int(c[2:]))
    lab = 'malware' if 'malware' in d.columns else [c for c in d.columns if c.lower() in ('label', 'labels', 'class')][0]
    X = d[tcols].values.astype(int)
    rng = np.random.RandomState(0)
    cands = {'page order (used)': page, 'alphabetical': sorted(page), 'alphabetical, case-insensitive': sorted(page, key=str.lower),
             'page order reversed': page[::-1]}
    for k in range(20):
        cands[f'random {k}'] = list(rng.permutation(page))

    def doc_freq(seqs):
        from collections import Counter
        c = Counter(x for s in seqs for x in set(s)); n = len(seqs)
        return {k: v / n for k, v in c.items()}

    ref = {lb: doc_freq([names_mb[h] for h in tr[tr.labels == lb].sha256]) for lb in (0, 1)}
    shared = sorted(set(x.lower() for x in page) & set(x for s in names_mb.values() for x in s))
    res = {}
    for cname, mapping in cands.items():
        low = [x.lower() for x in mapping]
        row = {}
        for lb, tag in ((0, 'goodware'), (1, 'malware')):
            df_ = doc_freq([[low[j] for j in r] for r in X[d[lab].values == lb]])
            x = [df_.get(n, 0.0) for n in shared]; y = [ref[lb].get(n, 0.0) for n in shared]
            row[tag] = float(spearmanr(x, y).correlation)
        res[cname] = row
    rnd = pd.DataFrame([v for k, v in res.items() if k.startswith('random')])
    out = {'n_shared_names': len(shared), 'named': {k: v for k, v in res.items() if not k.startswith('random')},
           'random_mean': rnd.mean().to_dict(), 'random_max': rnd.max().to_dict()}
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(out, open(os.path.join(a.outdir, 'mapping_check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


def run_summary(a):
    from scipy import stats
    d = pd.read_json(os.path.join(a.outdir, 'perturb20_results.jsonl'), lines=True)
    cond = pd.read_csv(os.path.join(a.outdir, 'conditions.csv'))
    L = ['# Second-round analyses (round2_v4.py)\n',
         'p (Holm) is corrected over the six comparisons fixed in advance (experiments_v3.COMPARISONS), separately for '
         'each condition and metric. Seeds are matched; the logistic regressions are deterministic and enter as '
         'constants. "+ seeds" counts seeds where the difference is above zero. Group metrics count each distinct '
         'clean test sequence once.\n']
    L.append('\n## Runs recorded\n')
    L.append(d.drop_duplicates(['model', 'seed']).groupby('model').seed.nunique().rename('seeds').to_frame().to_markdown())

    # determinism check against the five-seed perturb_v3 run
    if a.old_preds and os.path.isdir(a.old_preds):
        meta = json.load(open(os.path.join(a.outdir, 'meta.json'))); n3 = meta['n_rows_as_perturb_v3']
        L.append('\n## Reproduction of the five-seed perturb_v3 predictions\n')
        L.append('| run | rows compared | rows differing |'); L.append('|---|---|---|')
        for f in sorted(os.listdir(a.old_preds)):
            new = os.path.join(a.outdir, 'preds', f)
            if f.endswith('.npy') and os.path.exists(new):
                o, n = np.load(os.path.join(a.old_preds, f)), np.load(new)[:n3]
                if len(o) != len(n):
                    L.append(f'| {f[:-4]} | length mismatch: old {len(o)}, new {len(n)} | n/a |'); continue
                L.append(f'| {f[:-4]} | {len(o)} | {int((o != n).sum())} |')

    # preamble: how much of the original sequence survives first-occurrence reduction
    L.append('\n## Benign preamble: calls of the original sequence that survive first-occurrence reduction\n')
    L.append('| level | label | rows | mean original length | mean surviving | mean share surviving | median share | '
             'distinct sequences: mean share |')
    L.append('|---|---|---|---|---|---|---|---|')
    pre = cond[cond.kind == 'preamble_delonly']
    for (lv, lb), g in pre.groupby(['level', 'labels']):
        sh = g.n_after / g.n_clean.clip(lower=1)
        gs = g.assign(sh=sh).groupby('group').sh.mean()
        L.append(f'| {lv:g} | {"malicious" if lb else "benign"} | {len(g)} | {g.n_clean.mean():.1f} | {g.n_after.mean():.1f} | '
                 f'{100 * sh.mean():.1f}% | {100 * sh.median():.1f}% | {100 * gs.mean():.1f}% |')

    # means per model and condition
    order = ['clean'] + [k for k in P.LEVELS] + ['preamble_delonly']
    for metric in METRICS:
        L.append(f'\n## {metric}: mean (sd) over seeds\n')
        piv = d.groupby(['kind', 'level', 'model'])[metric].agg(['mean', 'std']).reset_index()
        models = [m for m in E.TRANSFORMERS + ['bilstm'] + E.LR_KINDS if m in set(d.model)]
        L.append('| condition | ' + ' | '.join(models) + ' |'); L.append('|---' * (len(models) + 1) + '|')
        for kind in order:
            for lv in sorted(piv[piv.kind == kind].level.unique()):
                cells = []
                for m in models:
                    r = piv[(piv.kind == kind) & (piv.level == lv) & (piv.model == m)]
                    if r.empty: cells.append(''); continue
                    mu, sd = 100 * r['mean'].iloc[0], 100 * r['std'].iloc[0]
                    cells.append(f'{mu:.1f}' + ('' if np.isnan(sd) else f' ({sd:.1f})'))
                L.append(f'| {kind} {lv:g} | ' + ' | '.join(cells) + ' |')

    # pre-registered paired comparisons
    for metric in ['recall', 'bal_acc', 'auc', 'recall_groups', 'bal_acc_groups', 'auc_groups']:
        L.append(f'\n## Paired comparisons, {metric} (percentage points)\n')
        L.append('| condition | comparison | n | mean diff | 95% CI | + seeds | p (t) | p (Holm) | p (Wilcoxon) |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for kind in order:
            for lv in sorted(d[d.kind == kind].level.unique()):
                sub = d[(d.kind == kind) & (d.level == lv)]
                piv = sub.pivot_table(index='seed', columns='model', values=metric, aggfunc='first')
                rows = []
                for a_, b_ in E.COMPARISONS:
                    if a_ not in piv or b_ not in piv: continue
                    if -1 in piv.index and piv[b_].notna().sum() == 1:
                        diff = (piv[a_].drop(index=-1) - piv.loc[-1, b_]).dropna()
                    else:
                        pair = piv[[a_, b_]].dropna(); diff = pair[a_] - pair[b_]
                    if len(diff) < 3: continue
                    sd = diff.std(ddof=1)
                    t = stats.ttest_1samp(diff, 0).pvalue if sd > 0 else (1.0 if diff.mean() == 0 else 0.0)
                    half = stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff)) if sd > 0 else 0.0
                    try: w = stats.wilcoxon(diff).pvalue
                    except ValueError: w = float('nan')
                    rows.append([f'{a_} minus {b_}', len(diff), diff.mean(), half, int((diff > 0).sum()), t, w])
                if not rows: continue
                ps = np.array([r[5] for r in rows]); order_ = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
                for k, i in enumerate(order_):
                    run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
                for r, h in zip(rows, adj):
                    L.append(f'| {kind} {lv:g} | {r[0]} | {r[1]} | {100 * r[2]:+.1f} | {100 * (r[2] - r[3]):+.1f} to '
                             f'{100 * (r[2] + r[3]):+.1f} | {r[4]}/{r[1]} | {r[5]:.4f} | {h:.4f} | {r[6]:.4f} |')
    open(os.path.join(a.outdir, 'round2_v4.md'), 'w', encoding='utf-8').write('\n'.join(L))
    print('\n'.join(L[:60])); print(f'... full report in {os.path.join(a.outdir, "round2_v4.md")}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['perturb', 'ngram', 'mapping', 'summary'])
    ap.add_argument('--data', default='data_v2')
    ap.add_argument('--malbehav'); ap.add_argument('--malapi')
    ap.add_argument('--s2preds', default='results_v3/preds')
    ap.add_argument('--ext-csv'); ap.add_argument('--api-list')
    ap.add_argument('--old-preds', default='results_perturb/preds')
    ap.add_argument('--outdir', default='results_perturb20')
    ap.add_argument('--seeds', type=int, nargs='+', default=E.SEEDS)
    ap.add_argument('--models', nargs='+', default=E.SEQ_MODELS)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--lstm-lr', type=float, default=1e-3)
    a = ap.parse_args()
    {'perturb': run_perturb, 'ngram': run_ngram, 'mapping': run_mapping, 'summary': run_summary}[a.mode](a)


if __name__ == '__main__':
    main()
