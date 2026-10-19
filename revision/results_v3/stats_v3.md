# Statistics for the revision (run 3)

Seeds are matched in every comparison. In S3 a seed fixes both the initialisation and the exchanged samples, so the logistic-regression reference is paired with the neural models; in S2 and S1 it is deterministic and enters as a constant. p (Holm) is corrected within each table over the comparisons fixed in advance in COMPARISONS. CIs: percentile bootstrap over seeds (10,000 resamples) for means; t-interval for mean differences. Both describe seed-to-seed variation only, not test-set sampling.


## S2 zero-shot recall, all 7,107 MAL-API traces

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.7338 | 0.0981 | 0.6907 to 0.7750 | 0.5421 | 0.9116 |
| lr_count_all | 1 | 0.6350 | n/a | deterministic | | |
| lr_count_uni | 1 | 0.6242 | n/a | deterministic | | |
| uni | 20 | 0.6391 | 0.0446 | 0.6195 to 0.6576 | 0.5397 | 0.7168 |
| uni_bi | 20 | 0.6819 | 0.0661 | 0.6526 to 0.7093 | 0.5433 | 0.7737 |
| uni_bi_tri | 20 | 0.5988 | 0.0749 | 0.5654 to 0.6296 | 0.4086 | 0.7017 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.0428 | +0.0046 to +0.0810 | 0.0300 | 0.0900 | 0.0400 | 0.52 |
| uni_bi_tri minus uni | paired | 20 | -0.0402 | -0.0800 to -0.0004 | 0.0478 | 0.0900 | 0.0522 | -0.47 |
| uni_bi_tri minus uni_bi | paired | 20 | -0.0831 | -0.1146 to -0.0516 | 0.0000 | 0.0002 | 0.0003 | -1.23 |
| uni_bi_tri minus bilstm | paired | 20 | -0.1350 | -0.1887 to -0.0812 | 0.0000 | 0.0002 | 0.0004 | -1.17 |
| uni_bi_tri minus lr_count_all | vs constant | 20 | -0.0362 | -0.0712 to -0.0011 | 0.0438 | 0.0900 | 0.0859 | -0.48 |
| uni_bi minus lr_count_all | vs constant | 20 | +0.0469 | +0.0160 to +0.0778 | 0.0050 | 0.0200 | 0.0080 | 0.71 |

## S2 zero-shot recall, unique MAL-API sequences

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.7518 | 0.0952 | 0.7095 to 0.7916 | 0.5690 | 0.9368 |
| lr_count_all | 1 | 0.6568 | n/a | deterministic | | |
| lr_count_uni | 1 | 0.6457 | n/a | deterministic | | |
| uni | 20 | 0.6659 | 0.0454 | 0.6461 to 0.6848 | 0.5702 | 0.7474 |
| uni_bi | 20 | 0.6963 | 0.0642 | 0.6678 to 0.7226 | 0.5529 | 0.7756 |
| uni_bi_tri | 20 | 0.6210 | 0.0763 | 0.5868 to 0.6524 | 0.4260 | 0.7289 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.0304 | -0.0073 to +0.0682 | 0.1078 | 0.1078 | 0.0930 | 0.38 |
| uni_bi_tri minus uni | paired | 20 | -0.0449 | -0.0855 to -0.0043 | 0.0322 | 0.0965 | 0.0400 | -0.52 |
| uni_bi_tri minus uni_bi | paired | 20 | -0.0753 | -0.1070 to -0.0436 | 0.0001 | 0.0004 | 0.0005 | -1.11 |
| uni_bi_tri minus bilstm | paired | 20 | -0.1308 | -0.1836 to -0.0780 | 0.0001 | 0.0003 | 0.0004 | -1.16 |
| uni_bi_tri minus lr_count_all | vs constant | 20 | -0.0359 | -0.0715 to -0.0002 | 0.0490 | 0.0981 | 0.1005 | -0.47 |
| uni_bi minus lr_count_all | vs constant | 20 | +0.0395 | +0.0094 to +0.0695 | 0.0128 | 0.0512 | 0.0169 | 0.61 |

## S1 in-distribution F1, MalbehavD test rows (same runs)

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.9639 | 0.0192 | 0.9548 to 0.9709 | 0.9048 | 0.9806 |
| lr_count_all | 1 | 0.9809 | n/a | deterministic | | |
| lr_count_uni | 1 | 0.9809 | n/a | deterministic | | |
| uni | 20 | 0.9719 | 0.0077 | 0.9686 to 0.9751 | 0.9545 | 0.9838 |
| uni_bi | 20 | 0.9778 | 0.0064 | 0.9750 to 0.9805 | 0.9623 | 0.9905 |
| uni_bi_tri | 20 | 0.9762 | 0.0044 | 0.9742 to 0.9780 | 0.9653 | 0.9840 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.0059 | +0.0019 to +0.0099 | 0.0061 | 0.0307 | 0.0064 | 0.69 |
| uni_bi_tri minus uni | paired | 20 | +0.0043 | +0.0001 to +0.0085 | 0.0461 | 0.1303 | 0.0364 | 0.48 |
| uni_bi_tri minus uni_bi | paired | 20 | -0.0016 | -0.0055 to +0.0023 | 0.4099 | 0.4099 | 0.2772 | -0.19 |
| uni_bi_tri minus bilstm | paired | 20 | +0.0123 | +0.0032 to +0.0215 | 0.0111 | 0.0442 | 0.0022 | 0.63 |
| uni_bi_tri minus lr_count_all | vs constant | 20 | -0.0047 | -0.0067 to -0.0026 | 0.0001 | 0.0009 | 0.0005 | -1.06 |
| uni_bi minus lr_count_all | vs constant | 20 | -0.0031 | -0.0061 to -0.0001 | 0.0434 | 0.1303 | 0.0140 | -0.48 |

## S1 in-distribution F1, unique MalbehavD test sequences (same runs)

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.9021 | 0.0199 | 0.8931 to 0.9103 | 0.8627 | 0.9348 |
| lr_count_all | 1 | 0.9375 | n/a | deterministic | | |
| lr_count_uni | 1 | 0.9375 | n/a | deterministic | | |
| uni | 20 | 0.9078 | 0.0226 | 0.8982 to 0.9174 | 0.8696 | 0.9451 |
| uni_bi | 20 | 0.9287 | 0.0189 | 0.9205 to 0.9365 | 0.8800 | 0.9691 |
| uni_bi_tri | 20 | 0.9220 | 0.0143 | 0.9155 to 0.9278 | 0.8889 | 0.9474 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.0208 | +0.0090 to +0.0326 | 0.0015 | 0.0061 | 0.0032 | 0.83 |
| uni_bi_tri minus uni | paired | 20 | +0.0141 | +0.0008 to +0.0274 | 0.0386 | 0.1159 | 0.0442 | 0.50 |
| uni_bi_tri minus uni_bi | paired | 20 | -0.0067 | -0.0176 to +0.0043 | 0.2171 | 0.2171 | 0.2273 | -0.29 |
| uni_bi_tri minus bilstm | paired | 20 | +0.0199 | +0.0089 to +0.0308 | 0.0012 | 0.0061 | 0.0040 | 0.85 |
| uni_bi_tri minus lr_count_all | vs constant | 20 | -0.0155 | -0.0222 to -0.0088 | 0.0001 | 0.0007 | 0.0005 | -1.09 |
| uni_bi minus lr_count_all | vs constant | 20 | -0.0088 | -0.0177 to -0.0000 | 0.0497 | 0.1159 | 0.0175 | -0.47 |

## S1 in-distribution false-positive rate, MalbehavD test rows (same runs)

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.0361 | 0.0177 | 0.0291 to 0.0439 | 0.0082 | 0.0820 |
| lr_count_all | 1 | 0.0246 | n/a | deterministic | | |
| lr_count_uni | 1 | 0.0246 | n/a | deterministic | | |
| uni | 20 | 0.0201 | 0.0129 | 0.0148 to 0.0258 | 0.0000 | 0.0574 |
| uni_bi | 20 | 0.0246 | 0.0138 | 0.0189 to 0.0311 | 0.0082 | 0.0656 |
| uni_bi_tri | 20 | 0.0283 | 0.0105 | 0.0242 to 0.0332 | 0.0164 | 0.0574 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.0045 | -0.0035 to +0.0125 | 0.2537 | 0.7610 | 0.2818 | 0.26 |
| uni_bi_tri minus uni | paired | 20 | +0.0082 | +0.0016 to +0.0148 | 0.0174 | 0.1044 | 0.0242 | 0.58 |
| uni_bi_tri minus uni_bi | paired | 20 | +0.0037 | -0.0048 to +0.0122 | 0.3743 | 0.7610 | 0.2486 | 0.20 |
| uni_bi_tri minus bilstm | paired | 20 | -0.0078 | -0.0179 to +0.0023 | 0.1221 | 0.6104 | 0.1521 | -0.36 |
| uni_bi_tri minus lr_count_all | vs constant | 20 | +0.0037 | -0.0012 to +0.0086 | 0.1313 | 0.6104 | 0.1191 | 0.35 |
| uni_bi minus lr_count_all | vs constant | 20 | +0.0000 | -0.0065 to +0.0065 | 1.0000 | 1.0000 | 0.6400 | 0.00 |

## S1 in-distribution false-positive rate, unique MalbehavD test sequences (same runs)

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.0388 | 0.0184 | 0.0312 to 0.0469 | 0.0089 | 0.0893 |
| lr_count_all | 1 | 0.0268 | n/a | deterministic | | |
| lr_count_uni | 1 | 0.0268 | n/a | deterministic | | |
| uni | 20 | 0.0219 | 0.0140 | 0.0161 to 0.0281 | 0.0000 | 0.0625 |
| uni_bi | 20 | 0.0268 | 0.0151 | 0.0205 to 0.0339 | 0.0089 | 0.0714 |
| uni_bi_tri | 20 | 0.0308 | 0.0114 | 0.0263 to 0.0362 | 0.0179 | 0.0625 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.0049 | -0.0038 to +0.0136 | 0.2537 | 0.7610 | 0.2818 | 0.26 |
| uni_bi_tri minus uni | paired | 20 | +0.0089 | +0.0018 to +0.0161 | 0.0174 | 0.1044 | 0.0242 | 0.58 |
| uni_bi_tri minus uni_bi | paired | 20 | +0.0040 | -0.0052 to +0.0133 | 0.3743 | 0.7610 | 0.2486 | 0.20 |
| uni_bi_tri minus bilstm | paired | 20 | -0.0080 | -0.0188 to +0.0027 | 0.1343 | 0.6567 | 0.1659 | -0.35 |
| uni_bi_tri minus lr_count_all | vs constant | 20 | +0.0040 | -0.0013 to +0.0094 | 0.1313 | 0.6567 | 0.1191 | 0.35 |
| uni_bi minus lr_count_all | vs constant | 20 | +0.0000 | -0.0070 to +0.0070 | 1.0000 | 1.0000 | 0.6400 | 0.00 |

## Diagnostic: S2 recall with the trigram block removed at test time (same trained models)

| model | n seeds | mean with block removed | mean intact | change |
|---|---|---|---|---|
| uni_bi_tri | 20 | 0.6010 | 0.5988 | +0.0022 |

## Diagnostic: S2 recall with all n-gram blocks removed at test time (same trained models)

| model | n seeds | mean with block removed | mean intact | change |
|---|---|---|---|---|
| uni_bi | 20 | 0.6356 | 0.6819 | -0.0463 |
| uni_bi_tri | 20 | 0.6085 | 0.5988 | +0.0096 |

## Diagnostic: S1 F1 with the trigram block removed at test time

| model | n seeds | mean with block removed | mean intact | change |
|---|---|---|---|---|
| uni_bi_tri | 20 | 0.9778 | 0.9762 | +0.0016 |

## Diagnostic: S1 F1 with all n-gram blocks removed at test time

| model | n seeds | mean with block removed | mean intact | change |
|---|---|---|---|---|
| uni_bi | 20 | 0.9617 | 0.9778 | -0.0161 |
| uni_bi_tri | 20 | 0.9513 | 0.9762 | -0.0249 |

## S3 recall on target test pool, budget n = 10

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.8887 | 0.0442 | 0.8692 to 0.9072 | 0.8085 | 0.9562 |
| lr_count_all | 20 | 0.8678 | 0.0350 | 0.8530 to 0.8825 | 0.8039 | 0.9253 |
| lr_count_uni | 20 | 0.6576 | 0.0138 | 0.6520 to 0.6637 | 0.6326 | 0.6917 |
| uni | 20 | 0.6696 | 0.0443 | 0.6514 to 0.6887 | 0.5970 | 0.7352 |
| uni_bi | 20 | 0.8479 | 0.0719 | 0.8164 to 0.8784 | 0.7104 | 0.9793 |
| uni_bi_tri | 20 | 0.8639 | 0.0662 | 0.8347 to 0.8907 | 0.6888 | 0.9753 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.1784 | +0.1372 to +0.2195 | 0.0000 | 0.0000 | 0.0001 | 2.03 |
| uni_bi_tri minus uni | paired | 20 | +0.1944 | +0.1552 to +0.2336 | 0.0000 | 0.0000 | 0.0001 | 2.32 |
| uni_bi_tri minus uni_bi | paired | 20 | +0.0160 | -0.0065 to +0.0385 | 0.1524 | 0.6095 | 0.2471 | 0.33 |
| uni_bi_tri minus bilstm | paired | 20 | -0.0248 | -0.0647 to +0.0151 | 0.2086 | 0.6095 | 0.2043 | -0.29 |
| uni_bi_tri minus lr_count_all | paired | 20 | -0.0039 | -0.0277 to +0.0200 | 0.7381 | 0.7381 | 0.9679 | -0.08 |
| uni_bi minus lr_count_all | paired | 20 | -0.0199 | -0.0486 to +0.0089 | 0.1644 | 0.6095 | 0.2471 | -0.32 |

## S3 recall on target test pool excluding API sets seen in the adapt pool, n = 10 (descriptive)

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.8839 | 0.0440 | 0.8643 to 0.9022 | 0.8006 | 0.9539 |
| lr_count_all | 20 | 0.8595 | 0.0330 | 0.8457 to 0.8736 | 0.8135 | 0.9169 |
| lr_count_uni | 20 | 0.6771 | 0.0111 | 0.6725 to 0.6819 | 0.6609 | 0.7016 |
| uni | 20 | 0.6988 | 0.0427 | 0.6813 to 0.7173 | 0.6277 | 0.7721 |
| uni_bi | 20 | 0.8425 | 0.0685 | 0.8135 to 0.8721 | 0.7338 | 0.9763 |
| uni_bi_tri | 20 | 0.8596 | 0.0623 | 0.8323 to 0.8852 | 0.7033 | 0.9719 |

## S3 source false-positive rate, budget n = 10

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.0279 | 0.0131 | 0.0225 to 0.0336 | 0.0082 | 0.0492 |
| lr_count_all | 20 | 0.0246 | 0.0000 | 0.0246 to 0.0246 | 0.0246 | 0.0246 |
| lr_count_uni | 20 | 0.0242 | 0.0018 | 0.0234 to 0.0246 | 0.0164 | 0.0246 |
| uni | 20 | 0.0266 | 0.0164 | 0.0201 to 0.0340 | 0.0000 | 0.0738 |
| uni_bi | 20 | 0.0275 | 0.0117 | 0.0230 to 0.0328 | 0.0082 | 0.0574 |
| uni_bi_tri | 20 | 0.0270 | 0.0089 | 0.0238 to 0.0311 | 0.0164 | 0.0574 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.0008 | -0.0065 to +0.0082 | 0.8180 | 1.0000 | 0.8073 | 0.05 |
| uni_bi_tri minus uni | paired | 20 | +0.0004 | -0.0077 to +0.0085 | 0.9169 | 1.0000 | 0.5823 | 0.02 |
| uni_bi_tri minus uni_bi | paired | 20 | -0.0004 | -0.0063 to +0.0055 | 0.8859 | 1.0000 | 0.9635 | -0.03 |
| uni_bi_tri minus bilstm | paired | 20 | -0.0008 | -0.0093 to +0.0076 | 0.8409 | 1.0000 | 0.6278 | -0.05 |
| uni_bi_tri minus lr_count_all | paired | 20 | +0.0025 | -0.0017 to +0.0066 | 0.2296 | 1.0000 | 0.2482 | 0.28 |
| uni_bi minus lr_count_all | paired | 20 | +0.0029 | -0.0026 to +0.0083 | 0.2856 | 1.0000 | 0.2803 | 0.25 |

## S3 recall on target test pool, budget n = 20

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.9129 | 0.0406 | 0.8950 to 0.9296 | 0.8142 | 0.9804 |
| lr_count_all | 20 | 0.9250 | 0.0262 | 0.9138 to 0.9359 | 0.8738 | 0.9722 |
| lr_count_uni | 20 | 0.6795 | 0.0171 | 0.6723 to 0.6868 | 0.6465 | 0.7155 |
| uni | 20 | 0.6936 | 0.0417 | 0.6763 to 0.7119 | 0.6124 | 0.7880 |
| uni_bi | 20 | 0.9040 | 0.0398 | 0.8867 to 0.9211 | 0.8340 | 0.9608 |
| uni_bi_tri | 20 | 0.9244 | 0.0484 | 0.9025 to 0.9434 | 0.8062 | 0.9835 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.2103 | +0.1834 to +0.2373 | 0.0000 | 0.0000 | 0.0001 | 3.65 |
| uni_bi_tri minus uni | paired | 20 | +0.2308 | +0.1995 to +0.2621 | 0.0000 | 0.0000 | 0.0001 | 3.45 |
| uni_bi_tri minus uni_bi | paired | 20 | +0.0204 | +0.0053 to +0.0355 | 0.0106 | 0.0319 | 0.0206 | 0.63 |
| uni_bi_tri minus bilstm | paired | 20 | +0.0115 | -0.0099 to +0.0329 | 0.2730 | 0.5460 | 0.2707 | 0.25 |
| uni_bi_tri minus lr_count_all | paired | 20 | -0.0007 | -0.0160 to +0.0147 | 0.9301 | 0.9301 | 0.8405 | -0.02 |
| uni_bi minus lr_count_all | paired | 20 | -0.0211 | -0.0349 to -0.0072 | 0.0049 | 0.0196 | 0.0057 | -0.71 |

## S3 recall on target test pool excluding API sets seen in the adapt pool, n = 20 (descriptive)

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.9075 | 0.0436 | 0.8881 to 0.9253 | 0.7999 | 0.9773 |
| lr_count_all | 20 | 0.9189 | 0.0260 | 0.9077 to 0.9296 | 0.8671 | 0.9678 |
| lr_count_uni | 20 | 0.6994 | 0.0147 | 0.6931 to 0.7055 | 0.6714 | 0.7277 |
| uni | 20 | 0.7165 | 0.0372 | 0.7008 to 0.7324 | 0.6361 | 0.8009 |
| uni_bi | 20 | 0.8982 | 0.0418 | 0.8800 to 0.9162 | 0.8305 | 0.9590 |
| uni_bi_tri | 20 | 0.9208 | 0.0463 | 0.9000 to 0.9391 | 0.8182 | 0.9820 |

## S3 source false-positive rate, budget n = 20

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.0336 | 0.0170 | 0.0266 to 0.0414 | 0.0082 | 0.0738 |
| lr_count_all | 20 | 0.0230 | 0.0043 | 0.0209 to 0.0246 | 0.0082 | 0.0246 |
| lr_count_uni | 20 | 0.0217 | 0.0040 | 0.0201 to 0.0234 | 0.0164 | 0.0246 |
| uni | 20 | 0.0266 | 0.0164 | 0.0201 to 0.0340 | 0.0000 | 0.0656 |
| uni_bi | 20 | 0.0270 | 0.0136 | 0.0217 to 0.0332 | 0.0082 | 0.0574 |
| uni_bi_tri | 20 | 0.0320 | 0.0124 | 0.0270 to 0.0377 | 0.0164 | 0.0656 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.0004 | -0.0070 to +0.0078 | 0.9091 | 1.0000 | 0.8761 | 0.03 |
| uni_bi_tri minus uni | paired | 20 | +0.0053 | -0.0033 to +0.0140 | 0.2127 | 0.7857 | 0.0894 | 0.29 |
| uni_bi_tri minus uni_bi | paired | 20 | +0.0049 | -0.0018 to +0.0117 | 0.1437 | 0.7183 | 0.1018 | 0.34 |
| uni_bi_tri minus bilstm | paired | 20 | -0.0016 | -0.0109 to +0.0076 | 0.7157 | 1.0000 | 0.8400 | -0.08 |
| uni_bi_tri minus lr_count_all | paired | 20 | +0.0090 | +0.0032 to +0.0148 | 0.0043 | 0.0259 | 0.0062 | 0.72 |
| uni_bi minus lr_count_all | paired | 20 | +0.0041 | -0.0023 to +0.0105 | 0.1964 | 0.7857 | 0.2630 | 0.30 |

## S3 recall on target test pool, budget n = 50

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.9534 | 0.0284 | 0.9405 to 0.9640 | 0.8642 | 0.9875 |
| lr_count_all | 20 | 0.9751 | 0.0094 | 0.9708 to 0.9788 | 0.9525 | 0.9878 |
| lr_count_uni | 20 | 0.7364 | 0.0278 | 0.7247 to 0.7481 | 0.6840 | 0.7835 |
| uni | 20 | 0.7378 | 0.0626 | 0.7121 to 0.7642 | 0.6365 | 0.8744 |
| uni_bi | 20 | 0.9765 | 0.0118 | 0.9713 to 0.9812 | 0.9480 | 0.9929 |
| uni_bi_tri | 20 | 0.9798 | 0.0146 | 0.9732 to 0.9857 | 0.9452 | 0.9989 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.2387 | +0.2112 to +0.2663 | 0.0000 | 0.0000 | 0.0001 | 4.06 |
| uni_bi_tri minus uni | paired | 20 | +0.2421 | +0.2139 to +0.2702 | 0.0000 | 0.0000 | 0.0001 | 4.03 |
| uni_bi_tri minus uni_bi | paired | 20 | +0.0033 | -0.0033 to +0.0100 | 0.3085 | 0.6169 | 0.1850 | 0.23 |
| uni_bi_tri minus bilstm | paired | 20 | +0.0265 | +0.0123 to +0.0407 | 0.0010 | 0.0039 | 0.0007 | 0.87 |
| uni_bi_tri minus lr_count_all | paired | 20 | +0.0048 | -0.0018 to +0.0113 | 0.1429 | 0.4287 | 0.0289 | 0.34 |
| uni_bi minus lr_count_all | paired | 20 | +0.0014 | -0.0031 to +0.0060 | 0.5219 | 0.6169 | 0.5377 | 0.15 |

## S3 recall on target test pool excluding API sets seen in the adapt pool, n = 50 (descriptive)

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.9495 | 0.0287 | 0.9367 to 0.9603 | 0.8596 | 0.9854 |
| lr_count_all | 20 | 0.9715 | 0.0107 | 0.9667 to 0.9757 | 0.9461 | 0.9858 |
| lr_count_uni | 20 | 0.7556 | 0.0243 | 0.7454 to 0.7658 | 0.7050 | 0.8047 |
| uni | 20 | 0.7576 | 0.0532 | 0.7355 to 0.7801 | 0.6663 | 0.8691 |
| uni_bi | 20 | 0.9748 | 0.0125 | 0.9693 to 0.9798 | 0.9437 | 0.9915 |
| uni_bi_tri | 20 | 0.9773 | 0.0160 | 0.9702 to 0.9838 | 0.9427 | 0.9986 |

## S3 source false-positive rate, budget n = 50

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.0361 | 0.0171 | 0.0291 to 0.0439 | 0.0082 | 0.0820 |
| lr_count_all | 20 | 0.0221 | 0.0039 | 0.0205 to 0.0238 | 0.0164 | 0.0246 |
| lr_count_uni | 20 | 0.0230 | 0.0057 | 0.0205 to 0.0254 | 0.0164 | 0.0328 |
| uni | 20 | 0.0275 | 0.0153 | 0.0209 to 0.0340 | 0.0000 | 0.0574 |
| uni_bi | 20 | 0.0270 | 0.0166 | 0.0209 to 0.0348 | 0.0082 | 0.0820 |
| uni_bi_tri | 20 | 0.0242 | 0.0050 | 0.0221 to 0.0262 | 0.0164 | 0.0328 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | -0.0004 | -0.0114 to +0.0106 | 0.9388 | 1.0000 | 0.9301 | -0.02 |
| uni_bi_tri minus uni | paired | 20 | -0.0033 | -0.0104 to +0.0038 | 0.3449 | 1.0000 | 0.3126 | -0.22 |
| uni_bi_tri minus uni_bi | paired | 20 | -0.0029 | -0.0108 to +0.0050 | 0.4565 | 1.0000 | 0.7139 | -0.17 |
| uni_bi_tri minus bilstm | paired | 20 | -0.0119 | -0.0205 to -0.0033 | 0.0092 | 0.0550 | 0.0106 | -0.65 |
| uni_bi_tri minus lr_count_all | paired | 20 | +0.0020 | +0.0003 to +0.0038 | 0.0210 | 0.1050 | 0.0253 | 0.56 |
| uni_bi minus lr_count_all | paired | 20 | +0.0049 | -0.0028 to +0.0126 | 0.1978 | 0.7911 | 0.2968 | 0.30 |

## S3 recall on target test pool, budget n = 100

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.9766 | 0.0157 | 0.9694 to 0.9828 | 0.9261 | 0.9963 |
| lr_count_all | 20 | 0.9890 | 0.0045 | 0.9870 to 0.9908 | 0.9784 | 0.9952 |
| lr_count_uni | 20 | 0.8074 | 0.0244 | 0.7971 to 0.8181 | 0.7712 | 0.8573 |
| uni | 20 | 0.8266 | 0.0721 | 0.7949 to 0.8548 | 0.6468 | 0.9111 |
| uni_bi | 20 | 0.9886 | 0.0066 | 0.9857 to 0.9914 | 0.9756 | 0.9997 |
| uni_bi_tri | 20 | 0.9932 | 0.0036 | 0.9915 to 0.9946 | 0.9861 | 0.9969 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.1620 | +0.1283 to +0.1958 | 0.0000 | 0.0000 | 0.0001 | 2.25 |
| uni_bi_tri minus uni | paired | 20 | +0.1666 | +0.1330 to +0.2002 | 0.0000 | 0.0000 | 0.0001 | 2.32 |
| uni_bi_tri minus uni_bi | paired | 20 | +0.0046 | +0.0013 to +0.0079 | 0.0089 | 0.0179 | 0.0169 | 0.65 |
| uni_bi_tri minus bilstm | paired | 20 | +0.0166 | +0.0093 to +0.0240 | 0.0001 | 0.0006 | 0.0003 | 1.06 |
| uni_bi_tri minus lr_count_all | paired | 20 | +0.0042 | +0.0019 to +0.0065 | 0.0011 | 0.0032 | 0.0038 | 0.86 |
| uni_bi minus lr_count_all | paired | 20 | -0.0004 | -0.0028 to +0.0020 | 0.7345 | 0.7345 | 0.9132 | -0.08 |

## S3 recall on target test pool excluding API sets seen in the adapt pool, n = 100 (descriptive)

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.9733 | 0.0178 | 0.9652 to 0.9803 | 0.9166 | 0.9963 |
| lr_count_all | 20 | 0.9873 | 0.0051 | 0.9850 to 0.9894 | 0.9749 | 0.9942 |
| lr_count_uni | 20 | 0.8143 | 0.0160 | 0.8076 to 0.8214 | 0.7850 | 0.8430 |
| uni | 20 | 0.8292 | 0.0591 | 0.8028 to 0.8525 | 0.6745 | 0.9010 |
| uni_bi | 20 | 0.9872 | 0.0074 | 0.9840 to 0.9903 | 0.9725 | 0.9997 |
| uni_bi_tri | 20 | 0.9921 | 0.0042 | 0.9901 to 0.9937 | 0.9837 | 0.9963 |

## S3 source false-positive rate, budget n = 100

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.0348 | 0.0182 | 0.0275 to 0.0430 | 0.0082 | 0.0820 |
| lr_count_all | 20 | 0.0217 | 0.0048 | 0.0197 to 0.0238 | 0.0082 | 0.0246 |
| lr_count_uni | 20 | 0.0279 | 0.0077 | 0.0246 to 0.0311 | 0.0164 | 0.0410 |
| uni | 20 | 0.0381 | 0.0128 | 0.0328 to 0.0439 | 0.0164 | 0.0656 |
| uni_bi | 20 | 0.0279 | 0.0131 | 0.0225 to 0.0340 | 0.0082 | 0.0656 |
| uni_bi_tri | 20 | 0.0254 | 0.0084 | 0.0221 to 0.0291 | 0.0164 | 0.0410 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | -0.0102 | -0.0172 to -0.0033 | 0.0059 | 0.0294 | 0.0066 | -0.69 |
| uni_bi_tri minus uni | paired | 20 | -0.0127 | -0.0199 to -0.0055 | 0.0015 | 0.0093 | 0.0027 | -0.83 |
| uni_bi_tri minus uni_bi | paired | 20 | -0.0025 | -0.0108 to +0.0059 | 0.5454 | 0.5454 | 0.5852 | -0.14 |
| uni_bi_tri minus bilstm | paired | 20 | -0.0094 | -0.0189 to +0.0001 | 0.0517 | 0.2066 | 0.0815 | -0.46 |
| uni_bi_tri minus lr_count_all | paired | 20 | +0.0037 | -0.0005 to +0.0079 | 0.0828 | 0.2217 | 0.0630 | 0.41 |
| uni_bi minus lr_count_all | paired | 20 | +0.0061 | -0.0007 to +0.0130 | 0.0739 | 0.2217 | 0.0741 | 0.42 |

## S3 recall on target test pool, budget n = 200

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.9879 | 0.0103 | 0.9831 to 0.9918 | 0.9605 | 0.9980 |
| lr_count_all | 20 | 0.9955 | 0.0017 | 0.9947 to 0.9962 | 0.9918 | 0.9989 |
| lr_count_uni | 20 | 0.8868 | 0.0146 | 0.8801 to 0.8926 | 0.8463 | 0.9034 |
| uni | 20 | 0.9124 | 0.0367 | 0.8960 to 0.9271 | 0.8204 | 0.9605 |
| uni_bi | 20 | 0.9963 | 0.0030 | 0.9950 to 0.9975 | 0.9886 | 0.9997 |
| uni_bi_tri | 20 | 0.9971 | 0.0032 | 0.9955 to 0.9982 | 0.9852 | 1.0000 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | +0.0839 | +0.0668 to +0.1011 | 0.0000 | 0.0000 | 0.0001 | 2.29 |
| uni_bi_tri minus uni | paired | 20 | +0.0847 | +0.0673 to +0.1020 | 0.0000 | 0.0000 | 0.0001 | 2.29 |
| uni_bi_tri minus uni_bi | paired | 20 | +0.0007 | -0.0011 to +0.0025 | 0.4163 | 0.5374 | 0.1399 | 0.19 |
| uni_bi_tri minus bilstm | paired | 20 | +0.0092 | +0.0039 to +0.0145 | 0.0018 | 0.0072 | 0.0013 | 0.81 |
| uni_bi_tri minus lr_count_all | paired | 20 | +0.0016 | +0.0001 to +0.0031 | 0.0376 | 0.1128 | 0.0060 | 0.50 |
| uni_bi minus lr_count_all | paired | 20 | +0.0009 | -0.0007 to +0.0025 | 0.2687 | 0.5374 | 0.0440 | 0.25 |

## S3 recall on target test pool excluding API sets seen in the adapt pool, n = 200 (descriptive)

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.9859 | 0.0116 | 0.9806 to 0.9904 | 0.9559 | 0.9976 |
| lr_count_all | 20 | 0.9947 | 0.0019 | 0.9939 to 0.9955 | 0.9908 | 0.9986 |
| lr_count_uni | 20 | 0.8764 | 0.0140 | 0.8704 to 0.8822 | 0.8474 | 0.8973 |
| uni | 20 | 0.9078 | 0.0331 | 0.8932 to 0.9212 | 0.8277 | 0.9552 |
| uni_bi | 20 | 0.9957 | 0.0034 | 0.9942 to 0.9971 | 0.9868 | 0.9997 |
| uni_bi_tri | 20 | 0.9966 | 0.0037 | 0.9947 to 0.9979 | 0.9830 | 1.0000 |

## S3 source false-positive rate, budget n = 200

| model | n seeds | mean | sd | 95% bootstrap CI | min | max |
|---|---|---|---|---|---|---|
| bilstm | 20 | 0.0250 | 0.0129 | 0.0197 to 0.0307 | 0.0000 | 0.0492 |
| lr_count_all | 20 | 0.0172 | 0.0059 | 0.0148 to 0.0197 | 0.0082 | 0.0246 |
| lr_count_uni | 20 | 0.0365 | 0.0094 | 0.0328 to 0.0406 | 0.0246 | 0.0574 |
| uni | 20 | 0.0398 | 0.0149 | 0.0336 to 0.0463 | 0.0246 | 0.0738 |
| uni_bi | 20 | 0.0258 | 0.0128 | 0.0209 to 0.0316 | 0.0082 | 0.0574 |
| uni_bi_tri | 20 | 0.0275 | 0.0153 | 0.0213 to 0.0344 | 0.0082 | 0.0656 |

| comparison | type | n | mean diff | 95% CI of diff | p (t) | p (Holm) | p (Wilcoxon) | Cohen dz |
|---|---|---|---|---|---|---|---|---|
| uni_bi minus uni | paired | 20 | -0.0139 | -0.0227 to -0.0051 | 0.0037 | 0.0221 | 0.0079 | -0.74 |
| uni_bi_tri minus uni | paired | 20 | -0.0123 | -0.0211 to -0.0035 | 0.0084 | 0.0421 | 0.0123 | -0.66 |
| uni_bi_tri minus uni_bi | paired | 20 | +0.0016 | -0.0047 to +0.0079 | 0.5922 | 1.0000 | 0.6592 | 0.12 |
| uni_bi_tri minus bilstm | paired | 20 | +0.0025 | -0.0059 to +0.0108 | 0.5454 | 1.0000 | 0.6933 | 0.14 |
| uni_bi_tri minus lr_count_all | paired | 20 | +0.0102 | +0.0023 to +0.0182 | 0.0143 | 0.0574 | 0.0174 | 0.60 |
| uni_bi minus lr_count_all | paired | 20 | +0.0086 | +0.0012 to +0.0160 | 0.0252 | 0.0755 | 0.0294 | 0.54 |

## S2 per-family recall (mean over seeds)

| model        |   fam_Adware |   fam_Backdoor |   fam_Downloader |   fam_Dropper |   fam_Spyware |   fam_Trojan |   fam_Virus |   fam_Worms |
|:-------------|-------------:|---------------:|-----------------:|--------------:|--------------:|-------------:|------------:|------------:|
| bilstm       |       0.6861 |         0.7676 |           0.7488 |        0.6888 |        0.7121 |       0.7446 |      0.6136 |      0.8706 |
| lr_count_all |       0.6385 |         0.6603 |           0.6334 |        0.5758 |        0.5974 |       0.6204 |      0.5335 |      0.8102 |
| lr_count_uni |       0.6623 |         0.6144 |           0.7043 |        0.5802 |        0.5601 |       0.6054 |      0.4885 |      0.7862 |
| uni          |       0.4836 |         0.6548 |           0.7293 |        0.6001 |        0.5918 |       0.6084 |      0.5389 |      0.7967 |
| uni_bi       |       0.6722 |         0.7116 |           0.7639 |        0.5976 |        0.6397 |       0.6725 |      0.5586 |      0.8167 |
| uni_bi_tri   |       0.5443 |         0.6087 |           0.7117 |        0.534  |        0.5458 |       0.5646 |      0.5042 |      0.7275 |

## Runs recorded

|                             |   seeds |
|:----------------------------|--------:|
| ('s2', 0, 'bilstm')         |      20 |
| ('s2', 0, 'lr_count_all')   |       1 |
| ('s2', 0, 'lr_count_uni')   |       1 |
| ('s2', 0, 'uni')            |      20 |
| ('s2', 0, 'uni_bi')         |      20 |
| ('s2', 0, 'uni_bi_tri')     |      20 |
| ('s3', 10, 'bilstm')        |      20 |
| ('s3', 10, 'lr_count_all')  |      20 |
| ('s3', 10, 'lr_count_uni')  |      20 |
| ('s3', 10, 'uni')           |      20 |
| ('s3', 10, 'uni_bi')        |      20 |
| ('s3', 10, 'uni_bi_tri')    |      20 |
| ('s3', 20, 'bilstm')        |      20 |
| ('s3', 20, 'lr_count_all')  |      20 |
| ('s3', 20, 'lr_count_uni')  |      20 |
| ('s3', 20, 'uni')           |      20 |
| ('s3', 20, 'uni_bi')        |      20 |
| ('s3', 20, 'uni_bi_tri')    |      20 |
| ('s3', 50, 'bilstm')        |      20 |
| ('s3', 50, 'lr_count_all')  |      20 |
| ('s3', 50, 'lr_count_uni')  |      20 |
| ('s3', 50, 'uni')           |      20 |
| ('s3', 50, 'uni_bi')        |      20 |
| ('s3', 50, 'uni_bi_tri')    |      20 |
| ('s3', 100, 'bilstm')       |      20 |
| ('s3', 100, 'lr_count_all') |      20 |
| ('s3', 100, 'lr_count_uni') |      20 |
| ('s3', 100, 'uni')          |      20 |
| ('s3', 100, 'uni_bi')       |      20 |
| ('s3', 100, 'uni_bi_tri')   |      20 |
| ('s3', 200, 'bilstm')       |      20 |
| ('s3', 200, 'lr_count_all') |      20 |
| ('s3', 200, 'lr_count_uni') |      20 |
| ('s3', 200, 'uni')          |      20 |
| ('s3', 200, 'uni_bi')       |      20 |
| ('s3', 200, 'uni_bi_tri')   |      20 |