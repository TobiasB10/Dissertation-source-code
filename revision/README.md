# Revision code and results

Code and per-run outputs for the revised version of "Multi-Granular DirIntBERT: Explicit API Call Order and the Limits of Cross-Corpus Malware Detection" (Burgess and Shone, Applied Sciences, manuscript applsci-4641827). The code in the folder above this one is the original dissertation version; everything in the revised paper comes from this folder.

Every table and figure in the paper can be recomputed from the files in `results_v3`, `results_extra` and `results_perturb` without retraining. Retraining reproduces them too: training is deterministic, and on the GPU used for the paper (NVIDIA RTX 2060 Super, Windows 11) re-running gives identical predictions. Other GPUs can differ slightly.

## Setup

```
pip install -r requirements.txt
```

The GPU runs used the CUDA 12.1 build of PyTorch listed in `requirements.txt`.

## Data

- MalbehavD-V1 and MAL-API-2019: use the dataset packs in the folder above (`MalBehavD-V1-dataset.csv`, `MalBehavD-V1-processed-clean.csv` and `all_analysis_data.txt`), or download them from https://github.com/mpasco/MalbehavD-V1 and https://github.com/ocatak/malware_api_class
- MAL-API-2019 family labels: `labels.csv` from https://github.com/ocatak/malware_api_class (one family per line, in the same order as `all_analysis_data.txt`). Needed for the per-category recall in Table 14.
- External corpus with goodware (S4): Oliveira, "Malware Analysis Datasets: API Call Sequences", IEEE DataPort, https://doi.org/10.21227/tqqm-aq14 (the CSV with columns `t_0` to `t_99`, `hash` and `malware`). It is not included here; download it from IEEE DataPort.

## Run order

| Step | Command | Produces | Used for |
|---|---|---|---|
| 1 | `python prepro_v2.py --malbehav MalBehavD-V1-dataset.csv --malapi all_analysis_data.txt --malapi-labels labels.csv --split-from MalBehavD-V1-processed-clean.csv --out data_v2` | `data_v2/` (harmonised data, vocabulary, splits) | Sections 3.3, 3.4, 4.2 |
| 2 | `python experiments_v3.py --mode baselines` then `--mode s2`, `--mode s3`, `--mode advval`, `--mode stats` | `results_v3/` | S1 to S3, Tables 8 to 15, Figures 2 and 3, Section 5.4 |
| 3 | `python data_checks_v3.py --malbehav MalBehavD-V1-dataset.csv --malapi all_analysis_data.txt` | `results_v3/data_checks.json` | Tables 4 and 10, near-duplicate check in Section 5.1 |
| 4 | `python perturb_v3.py --malbehav MalBehavD-V1-dataset.csv --malapi all_analysis_data.txt` | `results_perturb/` | Section 5.5, Table 16, Figure 4 |
| 5 | `python prep_oliveira.py --csv <Oliveira csv> --api-list api_list_page_order.txt --vocab data_v2/vocab_v2.json --out external/oliveira_v2.csv` | `external/oliveira_v2.csv` | Section 4.1 (external corpus and its API mapping) |
| 6 | `python extra_v3.py --mode shortcut`, `--mode lrsens`, `--mode minfreq`, `--mode external --external external/oliveira_v2.csv`, then `--mode summary` | `results_extra/` | Sections 4.5, 5.2, 5.4, 5.6, Table 17, Supplementary Table S4 |
| 7 | `python ext_checks_v3.py --external external/oliveira_v2.csv` | `results_extra/ext_checks.json` | Supporting checks in Section 5.6 |
| 8 | `python experiments_v2.py --mode resources` | model size and CPU latency | Section 5.7 |

`experiments_v2.py` is the earlier version of the experiments and is included only for `--mode resources`. All other results from it were replaced by `experiments_v3.py`. Each script explains its options in the comments at the top. Runs append to a results file and finished runs are skipped, so a script can be restarted after an interruption.

## What is in the results folders

- `results_v3/results_v3.jsonl`: one line per run (model, seed, budget and every metric). `results_v3.csv` and `stats_v3.md` are written by `--mode stats` (paired tests, Holm correction, confidence intervals, effect sizes).
- `results_v3/preds/`: per-sample predictions (0 benign, 1 malicious) for every run. `s2_<model>_<seed>.npy` is on all 7,107 MAL-API-2019 traces and `s2src_...` on the MalbehavD-V1 test split. `s3_<model>_<seed>_<n>.npy` is on the MAL-API-2019 test pool and `s3src_...` on the MalbehavD-V1 test split. The logistic regressions have no seed in S1 and S2.
- `results_v3/exchanged/s3_<seed>_<n>.json`: the `row_id`s of the MAL-API-2019 traces exchanged into training at each seed and budget.
- `results_extra/`: the extra checks and the external test (S4), with predictions on the external corpus in `preds/external_<model>_<seed>_<n>.npy` (seed -1 is the deterministic logistic regression with no target data).
- `results_perturb/`: perturbation results, the list of perturbed conditions and predictions per model and seed.
- `supplementary/`: Supplementary Tables S1 to S4 of the paper.

## Splits

- `splits/malbehav_split.csv`: the train, validation or test split of every MalbehavD-V1 sample (by `sha256`) and its sequence group. Identical sequences always share a split.
- `splits/malapi_pools.csv`: whether each MAL-API-2019 trace is in the adaptation pool or the test pool used in S3. `row_id` is the 0-based line number in `all_analysis_data.txt`.
- `splits/oliveira_groups.csv`: the sequence group of every row of the external corpus, in the row order of the published CSV.
- `splits/vocab_v2.json`: the unigram, bigram and trigram vocabulary built on the MalbehavD-V1 training split, and `prepro_stats.json` the statistics printed by `prepro_v2.py`.

The seeds are 72, 42, 123, 7, 2024 and the integers 1 to 16 other than 7. The perturbation tests use the first five.
