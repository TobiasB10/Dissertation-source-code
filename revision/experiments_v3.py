# revision experiments for applsci-4641827, run 3 (the final run) on the harmonised data from prepro_v2.py
# the data files are the same as run 2, only this script changed

# changes from experiments_v2.py, each one came out of the independent audit of run 2
# 1 - bilstm now packs its input (pack_padded_sequence)
#     in run 2 the backward direction read the trailing PAD positions, so a sample's prediction depended on how
#     much padding its batch needed and all bilstm numbers from run 2 are void
#     the transformers were not affected because the attention mask already hides PAD
# 2 - 20 seeds for every model, every budget and both scenarios (run 2 had 5 seeds at n = 10-100)
# 3 - lr_bin_uni removed, every api occurs once per sample so it was identical to lr_count_uni
# 4 - per-sample predictions are saved for every run and every model, on the source test split and the target set,
#     so every reported number can be recomputed without retraining
# 5 - extra metrics per run
#     recall over unique target sequences (groups) as well as over traces
#     recall on the target test pool leaving out traces whose api set also occurs in the adapt pool
#     source metrics over unique test sequences as well as over rows
# 6 - stats comparisons are fixed in advance (COMPARISONS) and always use matched seeds
#     the logistic regression reference is included, paired by seed in s3 where it sees the same exchanged samples
#     p values are holm corrected within each table
#     each bootstrap interval uses its own fixed random state
# 7 - adversarial validation de-duplicates MalbehavD malware by sequence, like MAL-API already was
# 8 - s2 channel-removal diagnostic with no extra training
#     each trained uni_bi_tri model is also evaluated with its trigram block blanked and with both n-gram blocks
#     blanked, and uni_bi with its bigram block blanked
#     tests which channels drive the zero-shot behaviour, descriptive only

# modes, each one appends to OUTDIR/results_v3.jsonl and skips runs that are already recorded
# baselines = logistic regression references on cpu, s2 and also s3 at every seed and budget
# s2 = zero-shot transfer, every model x every seed
# s3 = few-shot adaptation, every model x every seed x every budget
# advval = adversarial validation, can a simple classifier tell MalbehavD malware from MAL-API malware
# stats = tables, paired tests, bootstrap CIs and effect sizes, writes stats_v3.md and results_v3.csv

# models
# uni, uni_bi and uni_bi_tri are bert tiny with random initialisation, same as the paper
# bilstm is the unigram bilstm sequence baseline

# run it like this:
# python experiments_v3.py --mode baselines
# python experiments_v3.py --mode s2
# python experiments_v3.py --mode s3 --models uni_bi_tri --budgets 10 20
# python experiments_v3.py --mode advval
# python experiments_v3.py --mode stats
# data is read from --data (default data_v2) and results go to --outdir (default results_v3)

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # has to be set before cuda starts for determinism
import argparse, json, time, copy, random
import numpy as np
import pandas as pd

TRANSFORMERS = ['uni', 'uni_bi', 'uni_bi_tri']
SEQ_MODELS = TRANSFORMERS + ['bilstm']
# token columns each model reads
COLS = {'uni': ['uni'], 'uni_bi': ['uni', 'bi'], 'uni_bi_tri': ['uni', 'bi', 'tri'], 'bilstm': ['uni']}
CLS, PAD = 1, 0
SEEDS = [72, 42, 123, 7, 2024, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]
BUDGETS = [10, 20, 50, 100, 200]
LR_KINDS = ['lr_count_uni', 'lr_count_all']

# s2 channel-removal diagnostic, the trained model is evaluated again with these blocks blanked (descriptive only)
ABLATIONS = {'uni_bi_tri': [('notri', ['tri']), ('unionly', ['bi', 'tri'])],
             'uni_bi': [('unionly', ['bi'])]}

# comparisons fixed before any run 3 results existed, (a, b) reports a minus b
COMPARISONS = [('uni_bi', 'uni'), ('uni_bi_tri', 'uni'), ('uni_bi_tri', 'uni_bi'),
               ('uni_bi_tri', 'bilstm'), ('uni_bi_tri', 'lr_count_all'), ('uni_bi', 'lr_count_all')]


def load(data):
    mb = pd.read_csv(os.path.join(data, 'malbehav_v2.csv'), keep_default_na=False)
    ma = pd.read_csv(os.path.join(data, 'malapi_v2.csv'), keep_default_na=False)
    vocab = json.load(open(os.path.join(data, 'vocab_v2.json')))
    return mb, ma, vocab


# split MAL-API sequence groups 50/50 into an adapt pool and a test pool, always the same split
def target_pools(ma, pool_seed=0):
    groups = np.array(sorted(ma.group_id.unique()))
    rng = np.random.RandomState(pool_seed)
    rng.shuffle(groups)
    adapt_groups = set(groups[: len(groups) // 2])
    is_adapt = ma.group_id.isin(adapt_groups).values
    return is_adapt, ~is_adapt


# within the test pool, true for traces whose set of apis does not occur anywhere in the adapt pool
def strict_mask(ma, is_adapt, is_test):
    sets = ma.uni.apply(lambda s: frozenset(s.split()))
    adapt_sets = set(sets[is_adapt])
    return ~sets[is_test].isin(adapt_sets).values


# join the columns the model uses into one list of token ids per row
def seqs(df, model):
    out = []
    for _, r in df.iterrows():
        parts = [str(r[c]) for c in COLS[model] if str(r[c]).strip()]
        out.append([int(t) for t in ' '.join(parts).split()])
    return out


# s3 protocol, draw as many distinct groups as the budget from the adapt pool (one trace each)
# 10% of them (at least one) go to validation and the rest to training
# the same number of malicious MalbehavD training rows are removed so the training set size stays comparable
def exchange(mb, ma, is_adapt, seed, budget):
    rng = np.random.RandomState(seed * 1000 + budget)
    pool = ma[is_adapt].drop_duplicates('group_id')
    pick = pool.iloc[rng.choice(len(pool), size=budget, replace=False)].copy()
    n_val = max(1, int(round(0.1 * budget)))
    pick['split'] = 'train'
    pick.iloc[:n_val, pick.columns.get_loc('split')] = 'val'
    mal_train = mb[(mb.labels == 1) & (mb.split == 'train')]
    drop = mal_train.sample(n=budget, random_state=seed * 1000 + budget).index
    src = mb.drop(index=drop)
    tr = pd.concat([src[src.split == 'train'], pick[pick.split == 'train']], ignore_index=True)
    va = pd.concat([src[src.split == 'val'], pick[pick.split == 'val']], ignore_index=True)
    return tr, va, pick.row_id.tolist()


# recall over traces and over unique sequences (groups)
# optionally also leaving out traces whose api set is shared with the adapt pool
def target_metrics(p, df, prefix, strict=None):
    out = {f'{prefix}': float(p.mean()),
           f'{prefix}_groups': float(pd.Series(p).groupby(df.group_id.values).mean().mean())}
    if strict is not None:
        out[f'{prefix}_strict'] = float(p[strict].mean())
        out['n_strict'] = int(strict.sum())
    return out


def read_results(path):
    # parse floats exactly, pd.read_json uses a low precision parser which flips ties in the rank tests
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, encoding='utf-8') as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])


def done_keys(path):
    d = read_results(path)
    if d.empty:
        return set()
    return set(zip(d['mode'], d['model'], d['seed'], d['budget']))


def append(path, row):
    clean = {k: (v.item() if hasattr(v, 'item') else v) for k, v in row.items()}
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(clean) + '\n')


# recall per malware family
def family_recall(pred, families):
    out = {}
    for f in sorted(set(families)):
        m = families == f
        out[f'fam_{f}'] = float(pred[m].mean())
    return out


def save_preds(outdir, name, p):
    np.save(os.path.join(outdir, 'preds', name + '.npy'), np.asarray(p).astype(np.uint8))


# torch only gets imported in here, the baselines, advval and stats modes never load it
def torch_bits():
    import torch, torch.nn as nn
    from transformers import BertConfig, BertModel

    class DirIntBERT(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            # bert tiny, same config as the paper
            cfg = BertConfig(vocab_size=vocab_size, hidden_size=128, num_hidden_layers=2,
                             num_attention_heads=2, intermediate_size=512,
                             max_position_embeddings=512, pad_token_id=PAD,
                             attn_implementation='eager')  # deterministic attention on gpu
            self.bert = BertModel(cfg, add_pooling_layer=False)
            self.drop = nn.Dropout(0.1)
            self.head = nn.Linear(128, 2)

        def forward(self, ids, mask):
            h = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
            # mean pooling over the non PAD tokens
            m = mask.unsqueeze(-1).float()
            return self.head(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1e-9)))

    class BiLSTM(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, 128, padding_idx=PAD)
            self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.1)
            self.head = nn.Linear(256, 2)

        def forward(self, ids, mask):
            # packed so neither direction ever reads PAD positions (this fixes the run 2 bug)
            lengths = mask.sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(self.emb(ids), lengths, batch_first=True,
                                                       enforce_sorted=False)
            h, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=ids.size(1))
            # same mean pooling as the bert model
            m = mask.unsqueeze(-1).float()
            return self.head(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1e-9)))

    return torch, nn, DirIntBERT, BiLSTM


def set_seed(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # these lines are for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def batches(seqlist, labels, bs, shuffle, gen, torch, max_len=512):
    order = np.arange(len(seqlist))
    if shuffle:
        order = torch.randperm(len(seqlist), generator=gen).numpy()
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        rows = [([CLS] + seqlist[j])[:max_len] for j in idx]
        # pad to the longest row in this batch
        L = max(len(r) for r in rows)
        ids = torch.tensor([r + [PAD] * (L - len(r)) for r in rows])
        mask = torch.tensor([[1] * len(r) + [0] * (L - len(r)) for r in rows])
        yield ids, mask, torch.tensor([labels[j] for j in idx])


# train with the paper's settings, gives back predictions for each eval set plus some run info
def train_eval(model_name, tr, va, eval_sets, vocab_size, seed, a):
    torch, nn, DirIntBERT, BiLSTM = torch_bits()
    from transformers import get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(seed, torch)
    gen = torch.Generator().manual_seed(seed)
    model = (BiLSTM if model_name == 'bilstm' else DirIntBERT)(vocab_size).to(dev)
    lr = a.lstm_lr if model_name == 'bilstm' else a.lr
    xtr, ytr = seqs(tr, model_name), tr.labels.tolist()
    xva, yva = seqs(va, model_name), va.labels.tolist()
    # no weight decay on bias and layernorm
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], 'weight_decay': 0.0}], lr=lr)
    steps = ((len(xtr) + a.batch - 1) // a.batch) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)  # 10% warmup
    lossf = nn.CrossEntropyLoss()

    def predict(xs):
        model.eval()
        out = []
        with torch.no_grad():
            for ids, mask, _ in batches(xs, [0] * len(xs), 64, False, None, torch):
                out.append(model(ids.to(dev), mask.to(dev)).argmax(1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=int)

    # early stopping on val f1, keeps a copy of the best weights
    best, best_state, patience, epochs_run = -1.0, None, 0, 0
    for ep in range(a.epochs):
        model.train()
        epochs_run = ep + 1
        for ids, mask, y in batches(xtr, ytr, a.batch, True, gen, torch):
            opt.zero_grad()
            loss = lossf(model(ids.to(dev), mask.to(dev)), y.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            opt.step()
            sched.step()
        # check validation
        f1 = f1_score(yva, predict(xva), zero_division=0)
        if f1 > best:
            best, patience = f1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= a.patience:
                break
    # load the best model back before predicting
    model.load_state_dict(best_state)
    preds = {k: predict(seqs(df, model_name)) for k, df in eval_sets.items()}
    info = {'val_f1': best, 'epochs': epochs_run,
            'params': sum(p.numel() for p in model.parameters())}
    return preds, info


# source test metrics over rows and over unique test sequences (identical encoded inputs)
def source_metrics(te, p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    def m(y, q, suffix):
        pr, rc, f1, _ = precision_recall_fscore_support(y, q, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, q, labels=[0, 1]).ravel()
        return {f'src_acc{suffix}': accuracy_score(y, q), f'src_precision{suffix}': pr,
                f'src_recall{suffix}': rc, f'src_f1{suffix}': f1,
                f'src_fpr{suffix}': fp / max(fp + tn, 1), f'src_fp{suffix}': int(fp),
                f'src_benign{suffix}': int(fp + tn)}
    p = np.asarray(p)
    y = te.labels.values
    out = m(y, p, '')
    # unique sequences, a sequence counts as predicted malware if at least half its rows are
    key = (te.uni + '|' + te.bi + '|' + te.tri).values
    g = pd.DataFrame({'k': key, 'y': y, 'p': p}).groupby('k').agg(y=('y', 'first'), ny=('y', 'nunique'), p=('p', 'mean'))
    assert (g.ny == 1).all(), 'a test sequence carries both labels'
    out.update(m(g.y.values, (g.p.values >= 0.5).astype(int), '_unique'))
    return out


# token count features for the logistic regression references
def lr_features(df, kind, vocab):
    from scipy import sparse
    V = vocab['vocab_size']
    cols = ['uni'] if kind == 'lr_count_uni' else ['uni', 'bi', 'tri']
    r, c, v = [], [], []
    for i, (_, row) in enumerate(df.iterrows()):
        cnt = {}
        for col in cols:
            for t in str(row[col]).split():
                cnt[int(t)] = cnt.get(int(t), 0) + 1
        for k, x in cnt.items():
            r.append(i)
            c.append(k)
            v.append(float(x))
    return sparse.csr_matrix((v, (r, c)), shape=(len(df), V))


def fit_lr(tr, kind, vocab):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=5000, C=1.0).fit(lr_features(tr, kind, vocab), tr.labels.values)


def run_baselines(mb, ma, vocab, is_adapt, is_test, strict, a, res, outdir):
    done = done_keys(res)
    tr, te = mb[mb.split == 'train'], mb[mb.split == 'test']
    tp = ma[is_test]
    for kind in LR_KINDS:
        # s2 only needs one lr fit because it is deterministic, recorded as seed -1
        if ('s2', kind, -1, 0) not in done:
            clf = fit_lr(tr, kind, vocab)
            p_ma = clf.predict(lr_features(ma, kind, vocab))
            p_src = clf.predict(lr_features(te, kind, vocab))
            save_preds(outdir, f's2_{kind}', p_ma)
            save_preds(outdir, f's2src_{kind}', p_src)
            row = {'mode': 's2', 'model': kind, 'seed': -1, 'budget': 0,
                   **source_metrics(te, p_src),
                   **target_metrics(p_ma, ma, 'recall_all'),
                   **target_metrics(p_ma[is_test], tp, 'recall_testpool', strict),
                   **family_recall(p_ma, ma.family.values)}
            append(res, row)
            print(kind, 's2 recall', round(row['recall_all'], 4))
        # s3 at every seed and budget, the seed picks the exchanged samples
        for seed in a.seeds:
            for budget in a.budgets:
                if ('s3', kind, seed, budget) in done:
                    continue
                trx, vax, used = exchange(mb, ma, is_adapt, seed, budget)
                clf = fit_lr(trx, kind, vocab)  # lr has no use for validation so the val rows are just left out
                p_test = clf.predict(lr_features(tp, kind, vocab))
                p_src = clf.predict(lr_features(te, kind, vocab))
                save_preds(outdir, f's3_{kind}_{seed}_{budget}', p_test)
                save_preds(outdir, f's3src_{kind}_{seed}_{budget}', p_src)
                row = {'mode': 's3', 'model': kind, 'seed': seed, 'budget': budget,
                       **source_metrics(te, p_src),
                       **target_metrics(p_test, tp, 'recall_testpool', strict),
                       'misses_testpool': int((1 - p_test).sum()),
                       **family_recall(p_test, tp.family.values)}
                append(res, row)
        print(kind, 's3 done')


def run_s2(mb, ma, vocab, is_adapt, is_test, strict, a, res, outdir):
    done = done_keys(res)
    tr, va, te = mb[mb.split == 'train'], mb[mb.split == 'val'], mb[mb.split == 'test']
    tp = ma[is_test]
    for model in a.models:
        for seed in a.seeds:
            if ('s2', model, seed, 0) in done:
                continue
            t0 = time.time()
            # channel-removal diagnostic (no extra training), the same trained model is evaluated with its
            # n-gram blocks blanked to test which channels the zero-shot behaviour depends on
            evals = {'src': te, 'ma': ma}
            for tag, blank in ABLATIONS.get(model, []):
                evals[f'ma_{tag}'] = ma.assign(**{c: '' for c in blank})
                evals[f'src_{tag}'] = te.assign(**{c: '' for c in blank})
            preds, info = train_eval(model, tr, va, evals, vocab['vocab_size'], seed, a)
            p = preds['ma']
            save_preds(outdir, f's2_{model}_{seed}', p)
            save_preds(outdir, f's2src_{model}_{seed}', preds['src'])
            abl = {}
            for tag, _ in ABLATIONS.get(model, []):
                save_preds(outdir, f's2_{model}_{tag}_{seed}', preds[f'ma_{tag}'])
                save_preds(outdir, f's2src_{model}_{tag}_{seed}', preds[f'src_{tag}'])
                sm = source_metrics(te, preds[f'src_{tag}'])
                abl.update({f'recall_all_{tag}': float(preds[f'ma_{tag}'].mean()),
                            f'src_f1_{tag}': sm['src_f1'], f'src_fpr_{tag}': sm['src_fpr']})
            row = {'mode': 's2', 'model': model, 'seed': seed, 'budget': 0, **info,
                   **source_metrics(te, preds['src']),
                   **target_metrics(p, ma, 'recall_all'),
                   **target_metrics(p[is_test], tp, 'recall_testpool', strict),
                   **abl,
                   **family_recall(p, ma.family.values), 'minutes': (time.time() - t0) / 60}
            append(res, row)
            print(f's2 {model} seed {seed}: src_f1 {row["src_f1"]:.4f} recall {row["recall_all"]:.4f} ({row["minutes"]:.1f} min)')


def run_s3(mb, ma, vocab, is_adapt, is_test, strict, a, res, outdir):
    done = done_keys(res)
    te = mb[mb.split == 'test']
    tp = ma[is_test]
    for budget in a.budgets:
        for model in a.models:
            for seed in a.seeds:
                if ('s3', model, seed, budget) in done:
                    continue
                t0 = time.time()
                trx, vax, used = exchange(mb, ma, is_adapt, seed, budget)
                preds, info = train_eval(model, trx, vax, {'src': te, 'tp': tp}, vocab['vocab_size'], seed, a)
                p = preds['tp']
                save_preds(outdir, f's3_{model}_{seed}_{budget}', p)
                save_preds(outdir, f's3src_{model}_{seed}_{budget}', preds['src'])
                # save which target rows were exchanged into training
                with open(os.path.join(outdir, 'exchanged', f's3_{seed}_{budget}.json'), 'w') as f:
                    json.dump(used, f)
                row = {'mode': 's3', 'model': model, 'seed': seed, 'budget': budget, **info,
                       **source_metrics(te, preds['src']),
                       **target_metrics(p, tp, 'recall_testpool', strict),
                       'misses_testpool': int((1 - p).sum()),
                       **family_recall(p, tp.family.values), 'minutes': (time.time() - t0) / 60}
                append(res, row)
                print(f's3 n={budget} {model} seed {seed}: recall {row["recall_testpool"]:.4f} '
                      f'fpr {row["src_fpr"]:.4f} ({row["minutes"]:.1f} min)')


def run_advval(mb, ma, vocab, outdir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    mal = mb[mb.labels == 1].copy()
    mal = mal.assign(k=mal.uni + '|' + mal.bi + '|' + mal.tri).drop_duplicates('k')  # one row per sequence
    other = ma.drop_duplicates('group_id')
    n = min(len(mal), len(other))
    mal = mal.sample(n=n, random_state=0) if len(mal) > n else mal
    other = other.sample(n=n, random_state=0)
    # 0 = MalbehavD malware, 1 = MAL-API malware, same number of each
    df = pd.concat([mal.assign(y=0), other.assign(y=1)], ignore_index=True)
    out = {'n_per_corpus': int(n)}
    for kind in LR_KINDS:
        X = lr_features(df, kind, vocab)
        auc = cross_val_score(LogisticRegression(max_iter=5000), X, df.y.values,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc')
        out[kind] = {'auc_mean': float(auc.mean()), 'auc_std': float(auc.std())}
    # can trace length on its own separate the two
    out['length_only_auc'] = float(roc_auc_score(df.y.values, df.length.values))
    json.dump(out, open(os.path.join(outdir, 'advval.json'), 'w'), indent=2)
    print('adversarial validation:', json.dumps(out, indent=2))


def run_stats(outdir):
    from scipy import stats
    d = read_results(os.path.join(outdir, 'results_v3.jsonl'))
    d.to_csv(os.path.join(outdir, 'results_v3.csv'), index=False)
    lines = ['# Statistics for the revision (run 3)\n',
             'Seeds are matched in every comparison. In S3 a seed fixes both the initialisation and the exchanged '
             'samples, so the logistic-regression reference is paired with the neural models; in S2 and S1 it is '
             'deterministic and enters as a constant. p (Holm) is corrected within each table over the comparisons '
             'fixed in advance in COMPARISONS. CIs: percentile bootstrap over seeds (10,000 resamples) for means; '
             't-interval for mean differences. Both describe seed-to-seed variation only, not test-set sampling.\n']

    # percentile bootstrap ci of the mean, fixed random state so every interval can be reproduced
    def boot_ci(x, n=10000):
        rng = np.random.RandomState(0)
        x = np.asarray(x)
        b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
        return np.percentile(b, 2.5), np.percentile(b, 97.5)

    # holm correction, nan p values stay nan
    def holm(ps):
        ps = np.asarray(ps, dtype=float)
        adj = np.full(len(ps), np.nan)
        ok = np.where(~np.isnan(ps))[0]
        order = ok[np.argsort(ps[ok])]
        run = 0.0
        for k, i in enumerate(order):
            run = max(run, min(1.0, (len(order) - k) * ps[i]))
            adj[i] = run
        return adj

    # one table per metric, then the fixed comparisons unless tests is False
    def block(sub, metric, title, tests=True):
        if metric not in sub or sub[metric].notna().sum() == 0:
            return
        lines.append(f'\n## {title}\n')
        lines.append('| model | n seeds | mean | sd | 95% bootstrap CI | min | max |')
        lines.append('|---|---|---|---|---|---|---|')
        for m, g in sub.groupby('model'):
            x = g[metric].dropna().values
            if len(x) == 0:
                continue
            if len(x) == 1:
                lines.append(f'| {m} | 1 | {x[0]:.4f} | n/a | deterministic | | |')
                continue
            lo, hi = boot_ci(x)
            lines.append(f'| {m} | {len(x)} | {x.mean():.4f} | {x.std(ddof=1):.4f} | {lo:.4f} to {hi:.4f} | {x.min():.4f} | {x.max():.4f} |')
        if not tests:  # descriptive only, fixed in the analysis plan
            return
        piv = sub.pivot_table(index='seed', columns='model', values=metric, aggfunc='first')
        rows = []
        for a_, b_ in COMPARISONS:
            if a_ not in piv or b_ not in piv:
                continue
            # if b only has the deterministic seed -1 row (lr in s2) every seed of a is compared against it as a constant
            if -1 in piv.index and not np.isnan(piv.loc[-1, b_]) and piv[b_].notna().sum() == 1:
                diff = (piv[a_].drop(index=-1) - piv.loc[-1, b_]).dropna()
                kind = 'vs constant'
            else:
                pair = piv[[a_, b_]].dropna()
                diff = pair[a_] - pair[b_]
                kind = 'paired'
            if len(diff) < 3:
                continue
            sd = diff.std(ddof=1)
            if sd > 1e-12:
                t = stats.ttest_1samp(diff, 0).pvalue
                half = stats.t.ppf(0.975, len(diff) - 1) * sd / np.sqrt(len(diff))
                dz = diff.mean() / sd
            else:
                t, half, dz = (1.0 if abs(diff.mean()) < 1e-12 else float('nan')), 0.0, float('nan')  # constant difference so no test
            # wilcoxon on the differences rounded to 1e-10 so float noise cannot create or break ties
            # tie corrected normal approximation, zeros dropped
            dr = np.round(diff.values.astype(float), 10)
            try:
                w = stats.wilcoxon(dr, zero_method='wilcox', method='approx', correction=False).pvalue
            except ValueError:
                w = float('nan')
            rows.append([f'{a_} minus {b_}', kind, len(diff), diff.mean(), half, t, w, dz])
        if not rows:
            return
        adj = holm([r[5] for r in rows])
        lines.append('\n| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |')
        lines.append('|---|---|---|---|---|---|---|---|---|')
        for r, h in zip(rows, adj):
            md = 0.0 if abs(r[3]) < 5e-13 else r[3]  # show float noise as zero
            lines.append(f'| {r[0]} | {r[1]} | {r[2]} | {md:+.4f} | {md-r[4]:+.4f} to {md+r[4]:+.4f} | '
                         f'{r[5]:.4f} | {h:.4f} | {r[6]:.4f} | {r[7]:.2f} |')

    # s2 tables, the s1 numbers come from the source test set of the same s2 runs
    s2 = d[d['mode'] == 's2']
    block(s2, 'recall_all', 'S2 zero-shot recall, all 7,107 MAL-API traces')
    block(s2, 'recall_all_groups', 'S2 zero-shot recall, unique MAL-API sequences')
    block(s2, 'src_f1', 'S1 in-distribution F1, MalbehavD test rows (same runs)')
    block(s2, 'src_f1_unique', 'S1 in-distribution F1, unique MalbehavD test sequences (same runs)')
    block(s2, 'src_fpr', 'S1 in-distribution false-positive rate, MalbehavD test rows (same runs)')
    block(s2, 'src_fpr_unique', 'S1 in-distribution false-positive rate, unique MalbehavD test sequences (same runs)')
    # channel-removal diagnostic tables, change is the mean with the block removed minus the mean intact
    for col, title in (('recall_all_notri', 'Diagnostic: S2 recall with the trigram block removed at test time (same trained models)'),
                       ('recall_all_unionly', 'Diagnostic: S2 recall with all n-gram blocks removed at test time (same trained models)'),
                       ('src_f1_notri', 'Diagnostic: S1 F1 with the trigram block removed at test time'),
                       ('src_f1_unionly', 'Diagnostic: S1 F1 with all n-gram blocks removed at test time')):
        if col in s2:
            sub = s2[s2[col].notna()]
            lines.append(f'\n## {title}\n')
            lines.append('| model | n seeds | mean with block removed | mean intact | change |')
            lines.append('|---|---|---|---|---|')
            base = col.replace('_notri', '').replace('_unionly', '')
            for m, g in sub.groupby('model'):
                lines.append(f'| {m} | {len(g)} | {g[col].mean():.4f} | {g[base].mean():.4f} | {g[col].mean() - g[base].mean():+.4f} |')
    # s3 tables, one set per budget
    for b in sorted(d[d['mode'] == 's3'].budget.unique()):
        s3 = d[(d['mode'] == 's3') & (d.budget == b)]
        block(s3, 'recall_testpool', f'S3 recall on target test pool, budget n = {b}')
        block(s3, 'recall_testpool_strict', f'S3 recall on target test pool excluding API sets seen in the adapt pool, n = {b} (descriptive)', tests=False)
        block(s3, 'src_fpr', f'S3 source false-positive rate, budget n = {b}')
    # per family recall and how many seeds got recorded for each run
    fam = [c for c in d.columns if c.startswith('fam_')]
    if fam:
        lines.append('\n## S2 per-family recall (mean over seeds)\n')
        lines.append(s2.groupby('model')[fam].mean().round(4).to_markdown())
    lines.append('\n## Runs recorded\n')
    lines.append(d.groupby(['mode', 'budget', 'model']).seed.nunique().rename('seeds').to_frame().to_markdown())
    open(os.path.join(outdir, 'stats_v3.md'), 'w', encoding='utf-8').write('\n'.join(lines))
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['baselines', 's2', 's3', 'advval', 'stats'])
    ap.add_argument('--data', default='data_v2')
    ap.add_argument('--outdir', default='results_v3')
    ap.add_argument('--models', nargs='+', default=SEQ_MODELS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--budgets', type=int, nargs='+', default=BUDGETS)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--lstm-lr', type=float, default=1e-3)
    a = ap.parse_args()
    for sub in ('', 'preds', 'exchanged'):
        os.makedirs(os.path.join(a.outdir, sub), exist_ok=True)
    res = os.path.join(a.outdir, 'results_v3.jsonl')
    # stats only reads the results file so it does not need the data
    if a.mode == 'stats':
        return run_stats(a.outdir)
    mb, ma, vocab = load(a.data)
    is_adapt, is_test = target_pools(ma)
    strict = strict_mask(ma, is_adapt, is_test)
    if a.mode in ('s2', 's3'):
        import torch
        print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
    # bit of an odd way to pick the mode but it works
    {'baselines': lambda: run_baselines(mb, ma, vocab, is_adapt, is_test, strict, a, res, a.outdir),
     's2': lambda: run_s2(mb, ma, vocab, is_adapt, is_test, strict, a, res, a.outdir),
     's3': lambda: run_s3(mb, ma, vocab, is_adapt, is_test, strict, a, res, a.outdir),
     'advval': lambda: run_advval(mb, ma, vocab, a.outdir)}[a.mode]()


#   MAIN
if __name__ == '__main__':
    main()
