import os, sys, json, time, argparse, zlib
import numpy as np
import pandas as pd

# experiments_v3.py sits in the same folder as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments_v3 as E

# controlled perturbation experiment for applsci-4641827 (reviewer 1, sixth major comment)
# trains each config on the MalbehavD-V1 train split with the run 3 code and settings (experiments_v3.train_eval)
# then tests the same trained model on perturbed copies of the MalbehavD-V1 test split
# perturbations are done on the api name sequence, then it goes through the same first-occurrence reduction
# and train-only vocab (oov dropped) as every other input, exactly like a perturbed trace would in deployment

# run it like this:
# python perturb_v3.py --data data_v2 --malbehav MalBehavD-V1-dataset.csv --malapi all_analysis_data.txt --outdir results_perturb
# --seeds is optional

# perturbation kinds
# insert = benign filler calls inserted at random positions, k = round(rate * L)
# swap = adjacent calls swapped with probability p (one left to right pass, no overlaps)
# subst = each call replaced with probability q by an equivalent variant seen in either corpus
#   (A/W suffix, Ex suffix, Nt/Zw prefix) if one exists
# repeat = random calls and short segments repeated (tests that the input format really is invariant to this)
# preamble = k distinct benign calls added to the start (the evasion discussed in section 3.4)
LEVELS = {'insert': [0.1, 0.25, 0.5, 1.0], 'swap': [0.05, 0.1, 0.2, 0.4], 'subst': [0.25, 0.5, 1.0],
          'repeat': [0.5], 'preamble': [10, 25, 50, 100]}
DRAWS = 5  # random draws per level


def first_occurrence(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def grams(seq, n):
    return ['-'.join(seq[i:i + n]) for i in range(len(seq) - n + 1)]


# encode with the train vocab, oov tokens just get dropped
def encode(seq, vocab):
    u = [vocab['uni'][x] for x in seq if x in vocab['uni']]
    b = [vocab['bi'][x] for x in grams(seq, 2) if x in vocab['bi']]
    t = [vocab['tri'][x] for x in grams(seq, 3) if x in vocab['tri']]
    return ' '.join(map(str, u)), ' '.join(map(str, b)), ' '.join(map(str, t))


# equivalent variants of each api name, only keeps ones seen in either corpus (names are lower case)
def variant_map(names):
    out = {}
    for n in names:
        cands = set()
        # nt/zw prefix
        if n.startswith('nt'):
            cands.add('zw' + n[2:])
        if n.startswith('zw'):
            cands.add('nt' + n[2:])
        # a/w suffix
        stem, suf = (n[:-1], n[-1]) if n[-1:] in ('a', 'w') else (n, '')
        if suf:
            cands.add(stem + ('w' if suf == 'a' else 'a'))
        # with and without ex
        if stem.endswith('ex'):
            base = stem[:-2]
            cands.update({base + suf, base + 'a', base + 'w', base})
        else:
            cands.update({stem + 'ex' + suf, stem + 'exa', stem + 'exw', stem + 'ex'})
        cands = sorted(c for c in cands if c in names and c != n)
        if cands:
            out[n] = cands
    return out


# perturb one trace then reduce it like any other input
def perturb(seq, kind, level, rng, filler, fprob, variants, benign_seqs):
    s = list(seq)
    if kind == 'insert':
        k = int(round(level * len(s)))
        for _ in range(k):
            s.insert(rng.randint(0, len(s) + 1), filler[rng.choice(len(filler), p=fprob)])
    elif kind == 'swap':
        # skip past a swapped pair so swaps never overlap
        i = 0
        while i < len(s) - 1:
            if rng.rand() < level:
                s[i], s[i + 1] = s[i + 1], s[i]
                i += 2
            else:
                i += 1
    elif kind == 'subst':
        s = [variants[x][rng.randint(len(variants[x]))] if x in variants and rng.rand() < level else x for x in s]
    elif kind == 'repeat':
        out = []
        for x in s:
            out.append(x)
            if rng.rand() < level:
                out.extend([x] * rng.randint(1, 4))
        j = rng.randint(0, max(1, len(out) - 3))
        out = out[:j + 3] + out[j:j + 3] * rng.randint(1, 4) + out[j + 3:]  # one repeated segment
        s = out
    elif kind == 'preamble':
        pre = []
        while len(pre) < level:  # distinct benign calls taken in order from real benign training traces
            for x in benign_seqs[rng.randint(len(benign_seqs))]:
                if x not in pre:
                    pre.append(x)
                if len(pre) == level:
                    break
        s = pre + s
    return first_occurrence(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_v2')
    ap.add_argument('--malbehav', required=True)
    ap.add_argument('--malapi', required=True)
    ap.add_argument('--outdir', default='results_perturb')
    ap.add_argument('--seeds', type=int, nargs='+', default=E.SEEDS[:5])
    ap.add_argument('--models', nargs='+', default=E.SEQ_MODELS)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--lstm-lr', type=float, default=1e-3)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.outdir, 'preds'), exist_ok=True)
    mb, ma, vocab = E.load(a.data)

    # api names for each MalbehavD-V1 sample, lower case and in first-occurrence order
    raw = pd.read_csv(a.malbehav, low_memory=False).drop_duplicates('sha256', keep='first').set_index('sha256')
    cols = [str(i) for i in range(153) if str(i) in raw.columns]
    names = {h: first_occurrence([str(x).lower() for x in raw.loc[h, cols] if pd.notna(x)]) for h in mb.sha256}
    # sanity check, re-encoding the unperturbed name sequences has to give back the processed data exactly
    reenc = [encode(names[h], vocab) for h in mb.sha256]
    assert all((u, b, t) == (r.uni, r.bi, r.tri) for (u, b, t), (_, r) in zip(reenc, mb.iterrows())), 'name/ID mismatch'

    # every api name seen in either corpus, used to find the variants
    known = set(x for s in names.values() for x in s)
    with open(a.malapi, encoding='utf-8') as f:
        for line in f:
            known.update(line.lower().split())
    variants = variant_map(known)

    tr, te = mb[mb.split == 'train'], mb[mb.split == 'test']
    benign_tr = [names[h] for h in tr[tr.labels == 0].sha256]
    # filler calls are picked by how often they appear in benign training traces
    from collections import Counter
    fc = Counter(x for s in benign_tr for x in s)
    filler = sorted(fc)
    fprob = np.array([fc[x] for x in filler], float)
    fprob /= fprob.sum()

    # clean test set first, then a perturbed copy of the test set for every draw of every level
    rows = [{'cond': 'clean', 'kind': 'clean', 'level': 0, 'draw': 0, 'idx': i, 'labels': int(r.labels),
             'uni': r.uni, 'bi': r.bi, 'tri': r.tri} for i, (_, r) in enumerate(te.iterrows())]
    for kind, levels in LEVELS.items():
        for level in levels:
            for draw in range(DRAWS):
                # seed comes from the condition name so the perturbed sets are the same every run
                rng = np.random.RandomState(zlib.crc32(f'{kind}_{level}_{draw}'.encode()))
                for i, (_, r) in enumerate(te.iterrows()):
                    s = perturb(names[r.sha256], kind, level, rng, filler, fprob, variants, benign_tr)
                    u, b, t = encode(s, vocab)
                    rows.append({'cond': f'{kind}_{level}_{draw}', 'kind': kind, 'level': level, 'draw': draw,
                                 'idx': i, 'labels': int(r.labels), 'uni': u, 'bi': b, 'tri': t})
    ev = pd.DataFrame(rows)
    ev.drop(columns=['uni', 'bi', 'tri']).to_csv(os.path.join(a.outdir, 'conditions.csv'), index=False)
    # longest input counts the CLS position too
    longest = int(max(1 + len(u.split()) + len(b.split()) + len(t.split()) for u, b, t in zip(ev.uni, ev.bi, ev.tri)))
    # test samples with at least one call that has a variant
    nsub = sum(1 for h in te.sha256 if any(x in variants for x in names[h]))
    meta = {'n_eval_rows': len(ev), 'longest_input': longest, 'test_samples_with_a_variant': nsub,
            'variant_pairs': {k: v for k, v in sorted(variants.items())}}
    json.dump(meta, open(os.path.join(a.outdir, 'meta.json'), 'w'), indent=1)
    print('eval rows', len(ev), 'longest input', longest, 'test samples with a substitutable call', nsub, flush=True)

    # so a rerun skips model/seed pairs that are already in the results file
    res = os.path.join(a.outdir, 'perturb_results.jsonl')
    done = set()
    if os.path.exists(res):
        done = {(r['model'], r['seed']) for r in map(json.loads, open(res))}

    # save the predictions then write one results line per kind and level
    # recall = share of malware flagged, fpr = share of benign flagged, the _by_draw lists give the same per draw
    def record(model, seed, p, extra):
        ev['p'] = p
        np.save(os.path.join(a.outdir, 'preds', f'{model}_{seed}.npy'), p.astype(np.uint8))
        with open(res, 'a', encoding='utf-8') as f:
            for (kind, level), g in ev.groupby(['kind', 'level']):
                mal, ben = g[g.labels == 1], g[g.labels == 0]
                f.write(json.dumps({'model': model, 'seed': seed, 'kind': kind, 'level': level,
                                    'recall': float(mal.p.mean()), 'fpr': float(ben.p.mean()),
                                    'recall_by_draw': [float(x) for x in mal.groupby('draw').p.mean()],
                                    'fpr_by_draw': [float(x) for x in ben.groupby('draw').p.mean()], **extra}) + '\n')

    # logistic regression baselines, no seed so they get stored as -1
    for kind in E.LR_KINDS:
        if (kind, -1) in done:
            continue
        clf = E.fit_lr(tr, kind, vocab)
        record(kind, -1, clf.predict(E.lr_features(ev, kind, vocab)), {})
        print(kind, 'done', flush=True)
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    va = mb[mb.split == 'val']
    # sequence models, trained on the clean train split and tested on every condition
    for seed in a.seeds:
        for model in a.models:
            if (model, seed) in done:
                continue
            t0 = time.time()
            preds, info = E.train_eval(model, tr, va, {'ev': ev}, vocab['vocab_size'], seed, a)
            record(model, seed, preds['ev'], {'epochs': info['epochs'], 'minutes': (time.time() - t0) / 60})
            print(f'{model} seed {seed} done ({(time.time() - t0) / 60:.1f} min)', flush=True)


#   MAIN
if __name__ == '__main__':
    main()
