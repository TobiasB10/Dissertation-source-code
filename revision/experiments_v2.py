# earlier version of the revision experiments (run 2), kept in the release only for --mode resources
# which measured model size and cpu latency for section 5.7, every other result comes from experiments_v3.py
# revision experiments for applsci-4641827 on the harmonised data from prepro_v2.py

# modes, run them in this order, each one appends to OUTDIR/results_v2.jsonl and skips runs that are already recorded
# baselines = logistic regression references on cpu (only takes minutes), s2 and also s3 at every seed and budget
# s2 = zero-shot transfer, every model x every seed
#      gives in-distribution test metrics and MAL-API recall overall, per family and on the target test pool
# s3 = few-shot adaptation, every model x every seed x every budget
#      exchanged samples only ever come from the adapt pool
#      recall is measured on the untouched test pool (disjoint by sequence group)
#      and source fpr on the MalbehavD test split
# advval = adversarial validation, can a simple classifier tell MalbehavD malware from MAL-API malware
# resources = model size, single thread cpu latency and peak memory for the full model
# stats = paired tests, bootstrap CIs and effect sizes from results_v2.jsonl, writes stats_v2.md and results_v2.csv

# models
# uni, uni_bi and uni_bi_tri are bert tiny with random initialisation, same as the paper
# bilstm is the unigram bilstm sequence baseline

# run it like this:
# python experiments_v2.py --mode resources
# data is read from --data (default data_v2) and results go to --outdir (default results_v2)

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


# join the columns the model uses into one list of token ids per row
def seqs(df, model):
    out = []
    for _, r in df.iterrows():
        parts = [str(r[c]) for c in COLS[model] if str(r[c]).strip()]
        out.append([int(t) for t in ' '.join(parts).split()])
    return out


# clean s3 protocol, draw as many distinct groups as the budget from the adapt pool (one row each)
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


def read_results(path):
    return pd.read_json(path, lines=True) if os.path.exists(path) else pd.DataFrame()


# (mode, model, seed, budget) of every run already in the results file so they get skipped
def done_keys(path):
    d = read_results(path)
    if d.empty:
        return set()
    return set(zip(d['mode'], d['model'], d['seed'], d['budget']))


def append(path, row):
    clean = {k: (v.item() if hasattr(v, 'item') else v) for k, v in row.items()}  # numpy values to plain python for json
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(clean) + '\n')


# recall per malware family
def family_recall(pred, families):
    out = {}
    for f in sorted(set(families)):
        m = families == f
        out[f'fam_{f}'] = float(pred[m].mean())
    return out


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
            # not packed here so the backward direction also reads the PAD at the end (the run 2 bug, fixed in v3)
            h, _ = self.lstm(self.emb(ids))
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


# metrics on the MalbehavD test split, src_fpr is false positives over all benign test rows
def source_metrics(y, p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {'src_acc': accuracy_score(y, p), 'src_precision': pr, 'src_recall': rc, 'src_f1': f1,
            'src_fpr': fp / max(fp + tn, 1), 'src_fp': int(fp), 'src_benign': int(fp + tn)}


# features for the logistic regression references
# lr_count_uni = unigram counts, lr_bin_uni = unigram there or not, lr_count_all = counts over uni, bi and tri
def lr_features(df, kind, vocab):
    from scipy import sparse
    V = vocab['vocab_size']
    cols = ['uni'] if kind != 'lr_count_all' else ['uni', 'bi', 'tri']
    r, c, v = [], [], []
    for i, (_, row) in enumerate(df.iterrows()):
        cnt = {}
        for col in cols:
            for t in str(row[col]).split():
                cnt[int(t)] = cnt.get(int(t), 0) + 1
        for k, x in cnt.items():
            r.append(i)
            c.append(k)
            v.append(1.0 if kind == 'lr_bin_uni' else float(x))
    return sparse.csr_matrix((v, (r, c)), shape=(len(df), V))


def fit_lr(tr, kind, vocab):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=5000, C=1.0).fit(lr_features(tr, kind, vocab), tr.labels.values)


LR_KINDS = ['lr_count_uni', 'lr_bin_uni', 'lr_count_all']


def run_baselines(mb, ma, vocab, is_adapt, is_test, a, res):
    done = done_keys(res)
    tr, te = mb[mb.split == 'train'], mb[mb.split == 'test']
    for kind in LR_KINDS:
        # s2 only needs one lr fit because it is deterministic, recorded as seed -1
        if ('s2', kind, -1, 0) not in done:
            clf = fit_lr(tr, kind, vocab)
            p_ma = clf.predict(lr_features(ma, kind, vocab))
            row = {'mode': 's2', 'model': kind, 'seed': -1, 'budget': 0,
                   **source_metrics(te.labels.values, clf.predict(lr_features(te, kind, vocab))),
                   'recall_all': p_ma.mean(), 'recall_testpool': p_ma[is_test].mean(),
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
                p_test = clf.predict(lr_features(ma[is_test], kind, vocab))
                row = {'mode': 's3', 'model': kind, 'seed': seed, 'budget': budget,
                       **source_metrics(te.labels.values, clf.predict(lr_features(te, kind, vocab))),
                       'recall_testpool': p_test.mean(), 'misses_testpool': int((1 - p_test).sum()),
                       **family_recall(p_test, ma[is_test].family.values)}
                append(res, row)
        print(kind, 's3 done')


def run_s2(mb, ma, vocab, is_adapt, is_test, a, res, outdir):
    done = done_keys(res)
    tr, va, te = mb[mb.split == 'train'], mb[mb.split == 'val'], mb[mb.split == 'test']
    for model in a.models:
        for seed in a.seeds:
            if ('s2', model, seed, 0) in done:
                continue
            t0 = time.time()
            preds, info = train_eval(model, tr, va, {'src': te, 'ma': ma}, vocab['vocab_size'], seed, a)
            # save the per-sample MAL-API predictions for this run
            p = preds['ma']
            np.save(os.path.join(outdir, 'preds', f's2_{model}_{seed}.npy'), p.astype(np.uint8))
            row = {'mode': 's2', 'model': model, 'seed': seed, 'budget': 0, **info,
                   **source_metrics(te.labels.values, preds['src']),
                   'recall_all': p.mean(), 'recall_testpool': p[is_test].mean(),
                   **family_recall(p, ma.family.values), 'minutes': (time.time() - t0) / 60}
            append(res, row)
            print(f's2 {model} seed {seed}: src_f1 {row["src_f1"]:.4f} recall {row["recall_all"]:.4f} ({row["minutes"]:.1f} min)')


def run_s3(mb, ma, vocab, is_adapt, is_test, a, res, outdir):
    done = done_keys(res)
    te = mb[mb.split == 'test']
    for budget in a.budgets:
        for model in a.models:
            for seed in a.seeds:
                if ('s3', model, seed, budget) in done:
                    continue
                t0 = time.time()
                trx, vax, used = exchange(mb, ma, is_adapt, seed, budget)
                preds, info = train_eval(model, trx, vax, {'src': te, 'tp': ma[is_test]},
                                         vocab['vocab_size'], seed, a)
                # predictions on the target test pool
                p = preds['tp']
                np.save(os.path.join(outdir, 'preds', f's3_{model}_{seed}_{budget}.npy'), p.astype(np.uint8))
                # save which target rows were exchanged into training
                with open(os.path.join(outdir, 'exchanged', f's3_{seed}_{budget}.json'), 'w') as f:
                    json.dump(used, f)
                row = {'mode': 's3', 'model': model, 'seed': seed, 'budget': budget, **info,
                       **source_metrics(te.labels.values, preds['src']),
                       'recall_testpool': p.mean(), 'misses_testpool': int((1 - p).sum()),
                       **family_recall(p, ma[is_test].family.values), 'minutes': (time.time() - t0) / 60}
                append(res, row)
                print(f's3 n={budget} {model} seed {seed}: recall {row["recall_testpool"]:.4f} '
                      f'fpr {row["src_fpr"]:.4f} ({row["minutes"]:.1f} min)')


def run_advval(mb, ma, vocab, outdir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    mal = mb[mb.labels == 1]
    other = ma.drop_duplicates('group_id').sample(n=len(mal), random_state=0)
    # 0 = MalbehavD malware, 1 = MAL-API malware, same number of each
    df = pd.concat([mal.assign(y=0), other.assign(y=1)], ignore_index=True)
    out = {}
    for kind in ['lr_bin_uni', 'lr_count_all']:
        X = lr_features(df, kind, vocab)
        auc = cross_val_score(LogisticRegression(max_iter=5000), X, df.y.values,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc')
        out[kind] = {'auc_mean': float(auc.mean()), 'auc_std': float(auc.std())}
    # length as the only feature, can sequence length on its own tell the two corpora apart
    from sklearn.metrics import roc_auc_score
    out['length_only_auc'] = float(roc_auc_score(df.y.values, df.length.values))
    json.dump(out, open(os.path.join(outdir, 'advval.json'), 'w'), indent=2)
    print('adversarial validation:', json.dumps(out, indent=2))


# model size, cpu latency and peak memory for section 5.7
# the model is freshly initialised and not trained, size and latency come out the same either way
def run_resources(mb, vocab, outdir, a):
    torch, nn, DirIntBERT, BiLSTM = torch_bits()
    torch.set_num_threads(1)  # single cpu thread
    model = DirIntBERT(vocab['vocab_size']).eval()
    # fp32 size on disk, the file gets deleted straight after
    path = os.path.join(outdir, 'size_check.pt')
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / 1e6
    os.remove(path)
    # full model so the inputs are the uni_bi_tri sequences from the MalbehavD test split
    te = mb[mb.split == 'test']
    xs = seqs(te, 'uni_bi_tri')
    try:
        import resource  # unix only
    except ImportError:
        resource = None
    times = []
    with torch.no_grad():
        for x in xs[:20]:  # warm up
            ids = torch.tensor([[CLS] + x])
            model(ids, torch.ones_like(ids))
        # one sample at a time, latency in ms
        for x in xs:
            ids = torch.tensor([[CLS] + x])
            t = time.perf_counter()
            model(ids, torch.ones_like(ids))
            times.append((time.perf_counter() - t) * 1000)
    # peak memory of the whole process, ru_maxrss is in kB on linux so /1024 gives MB
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None
    out = {'params': sum(p.numel() for p in model.parameters()), 'fp32_size_mb': size_mb,
           'cpu_threads': 1, 'latency_ms_median': float(np.median(times)),
           'latency_ms_p95': float(np.percentile(times, 95)),
           'peak_rss_mb_process': (rss1 / 1024) if rss1 else None,
           'peak_rss_note': 'whole Python process incl. PyTorch runtime; not reported on Windows',
           'note': 'single-sample batch, one CPU thread, MalbehavD test inputs, full model'}
    json.dump(out, open(os.path.join(outdir, 'resources.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2))


def run_stats(outdir):
    from scipy import stats
    d = read_results(os.path.join(outdir, 'results_v2.jsonl'))
    d.to_csv(os.path.join(outdir, 'results_v2.csv'), index=False)
    lines = ['# Statistics for the revision\n']

    # percentile bootstrap ci of the mean
    # the rng default is only made once so every call carries on the same random stream
    def boot_ci(x, n=10000, rng=np.random.RandomState(0)):
        x = np.asarray(x)
        b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
        return np.percentile(b, 2.5), np.percentile(b, 97.5)

    # one table per metric, then paired comparisons between the sequence models
    def block(sub, metric, title):
        lines.append(f'\n## {title}\n')
        lines.append('| model | n seeds | mean | sd | 95% bootstrap CI | min | max |')
        lines.append('|---|---|---|---|---|---|---|')
        for m, g in sub.groupby('model'):
            x = g[metric].dropna().values
            if len(x) == 0:
                continue
            # lr in s2 is deterministic so it only has the one row
            if len(x) == 1:
                lines.append(f'| {m} | 1 | {x[0]:.4f} | n/a | deterministic | | |')
                continue
            lo, hi = boot_ci(x)
            lines.append(f'| {m} | {len(x)} | {x.mean():.4f} | {x.std(ddof=1):.4f} | {lo:.4f} to {hi:.4f} | {x.min():.4f} | {x.max():.4f} |')
        # matched by seed, the lr references are left out of the paired tests
        seq = [m for m in SEQ_MODELS if m in set(sub.model)]
        piv = sub[sub.model.isin(seq)].pivot_table(index='seed', columns='model', values=metric)
        lines.append('\nPaired comparisons across seeds (same seed = same initialisation stream and, in S3, same exchanged samples):\n')
        lines.append('| comparison | n | mean diff | paired t p | Wilcoxon p | Cohen d (paired) | Levene p (variance) |')
        lines.append('|---|---|---|---|---|---|---|')
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                pair = piv[[seq[i], seq[j]]].dropna()
                if len(pair) < 3:  # need at least 3 matched seeds
                    continue
                diff = pair[seq[j]] - pair[seq[i]]
                t = stats.ttest_rel(pair[seq[j]], pair[seq[i]]).pvalue
                try:
                    w = stats.wilcoxon(diff).pvalue
                except ValueError:
                    w = float('nan')
                dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float('nan')
                # levene checks if the two models differ in seed to seed variance
                lev = stats.levene(pair[seq[i]], pair[seq[j]]).pvalue
                lines.append(f'| {seq[j]} minus {seq[i]} | {len(pair)} | {diff.mean():+.4f} | {t:.4f} | {w:.4f} | {dz:.2f} | {lev:.4f} |')

    # s2 tables, the s1 numbers come from the source test set of the same s2 runs
    s2 = d[d['mode'] == 's2']
    block(s2, 'recall_all', 'S2 zero-shot recall, all 7,107 MAL-API samples')
    block(s2, 'src_f1', 'S1 in-distribution F1 (same runs)')
    # s3 tables, one set per budget
    for b in sorted(d[d['mode'] == 's3'].budget.unique()):
        s3 = d[(d['mode'] == 's3') & (d.budget == b)]
        block(s3, 'recall_testpool', f'S3 recall on target test pool, budget n = {b}')
        block(s3, 'src_fpr', f'S3 source false-positive rate, budget n = {b}')
    # per family recall
    fam = [c for c in d.columns if c.startswith('fam_')]
    if fam:
        lines.append('\n## S2 per-family recall (mean over seeds)\n')
        lines.append(s2.groupby('model')[fam].mean().round(4).to_markdown())
    open(os.path.join(outdir, 'stats_v2.md'), 'w', encoding='utf-8').write('\n'.join(lines))
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['baselines', 's2', 's3', 'advval', 'resources', 'stats'])
    ap.add_argument('--data', default='data_v2')
    ap.add_argument('--outdir', default='results_v2')
    ap.add_argument('--models', nargs='+', default=SEQ_MODELS)
    ap.add_argument('--seeds', type=int, nargs='+', default=[72, 42, 123, 7, 2024, 1, 2, 3, 4, 5])
    ap.add_argument('--budgets', type=int, nargs='+', default=[200])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--lstm-lr', type=float, default=1e-3)
    a = ap.parse_args()
    for sub in ('', 'preds', 'exchanged'):
        os.makedirs(os.path.join(a.outdir, sub), exist_ok=True)
    res = os.path.join(a.outdir, 'results_v2.jsonl')
    # stats only reads the results file so it does not need the data
    if a.mode == 'stats':
        return run_stats(a.outdir)
    # resources only loads what it needs so the peak memory reflects the model and not the datasets
    if a.mode == 'resources':
        mb = pd.read_csv(os.path.join(a.data, 'malbehav_v2.csv'), keep_default_na=False)
        return run_resources(mb, json.load(open(os.path.join(a.data, 'vocab_v2.json'))), a.outdir, a)
    mb, ma, vocab = load(a.data)
    is_adapt, is_test = target_pools(ma)
    if a.mode in ('s2', 's3'):
        import torch
        print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
    # bit of an odd way to pick the mode but it works
    {'baselines': lambda: run_baselines(mb, ma, vocab, is_adapt, is_test, a, res),
     's2': lambda: run_s2(mb, ma, vocab, is_adapt, is_test, a, res, a.outdir),
     's3': lambda: run_s3(mb, ma, vocab, is_adapt, is_test, a, res, a.outdir),
     'advval': lambda: run_advval(mb, ma, vocab, a.outdir)}[a.mode]()


#   MAIN
if __name__ == '__main__':
    main()
