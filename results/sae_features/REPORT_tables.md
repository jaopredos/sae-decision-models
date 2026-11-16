# Tabelas geradas dos artefatos (sem interpretação)

## Jobs

| job | o que |
|---|---|
| 3603 | extract-tokens (curto) |
| 3605 | extract-tokens (completo) |
| 3604 | corpus (curto) |
| 3606 | corpus (completo) |
| 3609 | sae-train (curto) |
| 3610 | probe-oof |
| 3613 | sae-train enc_17 |
| 3614 | sae-train enc_08 |
| 3615 | sae-train enc_17 literal |
| 3618 | features enc_08 |
| 3619 | quality enc_08 |
| 3620 | features enc_17 |
| 3621 | quality enc_17 |
| 3623 | quality enc_17 literal |
| 3626/3631 | features enc_08 (CMH; 3631 com estrato por visão) |
| 3627/3632 | features enc_17 (CMH; 3632 com estrato por visão) |
| 3628 | sae-train enc_08 dom20 |
| 3629 | quality enc_08 dom20 |
| 3617/3625 | partc-extract (curto/completo) |
| 3624/3633 | partc-tokens (curto/completo) |
| 3634 | sae-train base inglês |
| 3635 | quality base |
| 3636 | features base |
| 3637 | features fine-tuned (SAE do base) |
| 3638 | partc-compare 1ª passada |
| 3640 | massive-markers |
| 3641 | sae-train cópia ft |
| 3643 | quality cópia ft |
| 3644 | features cópia ft |
| 3647 | partc-compare 2ª passada (pendente) |

## Qualidade dos SAEs

### ml_enc08_k32

**FVE e L0 (dev e texto genérico retido)**

| section | group | frac_massive | fve | l0 | n_tokens |
|---|---|---|---|---|---|
| domain_dev | all_non_massive |  | 0.954 | 32.000 | 283622.000 |
| domain_dev | all_tokens_literal |  | 0.954 | 32.000 | 283622.000 |
| domain_dev | cls | 0.000 |  |  |  |
| domain_dev | end_sep | 0.000 |  |  |  |
| domain_dev | head_sep | 0.000 |  |  |  |
| domain_dev | marker | 0.000 |  |  |  |
| domain_dev | markers | 0.000 | 0.777 | 32.000 | 3161.000 |
| domain_dev | option_text | 0.000 | 0.963 | 32.000 | 39334.000 |
| domain_dev | question | 0.000 | 0.961 | 32.000 | 13197.000 |
| domain_dev | special |  | 0.996 | 32.000 | 2700.000 |
| domain_dev | state | 0.000 | 0.949 | 32.000 | 225230.000 |
| generic_val | all_non_massive |  | 0.830 | 32.000 | 1000000.000 |
| generic_val | all_tokens_literal | 0.000 |  |  |  |
| generic_val | arb_Arab |  | 0.765 | 32.000 | 35778.000 |
| generic_val | cmn_Hani |  | 0.753 | 32.000 | 40031.000 |
| generic_val | deu_Latn |  | 0.774 | 32.000 | 26179.000 |
| generic_val | en |  | 0.849 | 32.000 | 666267.000 |
| generic_val | fra_Latn |  | 0.802 | 32.000 | 18669.000 |
| generic_val | hin_Deva |  | 0.790 | 32.000 | 32871.000 |
| generic_val | ita_Latn |  | 0.791 | 32.000 | 26923.000 |
| generic_val | jpn_Jpan |  | 0.757 | 32.000 | 27547.000 |
| generic_val | kor_Hang |  | 0.797 | 32.000 | 25956.000 |
| generic_val | por_Latn |  | 0.795 | 32.000 | 18428.000 |
| generic_val | rus_Cyrl |  | 0.762 | 32.000 | 33229.000 |
| generic_val | spa_Latn |  | 0.800 | 32.000 | 22781.000 |
| generic_val | tur_Latn |  | 0.763 | 32.000 | 25341.000 |

**Latentes mortas**

| group | metric | value |
|---|---|---|
| training_counter | frac_dead | 0.000 |
| eval_tokens | frac_never_fired | 0.001 |

**Fidelidade a jusante (dev)**

| group | accuracy | delta_accuracy_pp | kl_nat | argmax_flip_rate | n_decisions |
|---|---|---|---|---|---|
| mean_non_massive_all | 0.296 | -3.222 | 0.400 | 0.787 |  |
| mean_non_massive_markers | 0.333 | 0.556 | 0.096 | 0.271 |  |
| original | 0.328 |  |  |  | 900.000 |
| replace_all_incl_massive | 0.330 | 0.222 | 0.064 | 0.156 |  |
| replace_non_massive_all | 0.330 | 0.222 | 0.064 | 0.156 |  |
| replace_non_massive_markers | 0.330 | 0.222 | 0.001 | 0.008 |  |
| zero_non_massive_all | 0.303 | -2.444 | 0.400 | 0.829 |  |
| zero_non_massive_markers | 0.350 | 2.222 | 0.353 | 0.520 |  |

**Sinalizadores do protocolo:** FVE nos marcadores 0.777 está mais de 0,10 abaixo da geral 0.954

### ml_enc17_k32

**FVE e L0 (dev e texto genérico retido)**

| section | group | frac_massive | fve | l0 | n_tokens |
|---|---|---|---|---|---|
| domain_dev | all_non_massive |  | 0.899 | 32.000 | 271205.000 |
| domain_dev | all_tokens_literal |  | 0.938 | 32.000 | 283622.000 |
| domain_dev | cls | 1.000 |  |  |  |
| domain_dev | end_sep | 0.766 |  |  |  |
| domain_dev | head_sep | 1.000 |  |  |  |
| domain_dev | marker | 0.000 |  |  |  |
| domain_dev | markers | 0.000 | 0.812 | 32.000 | 3161.000 |
| domain_dev | massive |  | 0.885 | 32.000 | 12417.000 |
| domain_dev | option_text | 0.040 | 0.927 | 32.000 | 37772.000 |
| domain_dev | question | 0.079 | 0.923 | 32.000 | 12160.000 |
| domain_dev | special |  | 0.924 | 32.000 | 2700.000 |
| domain_dev | state | 0.033 | 0.889 | 32.000 | 217901.000 |
| generic_val | all_non_massive |  | 0.761 | 32.000 | 967130.000 |
| generic_val | all_tokens_literal | 0.033 |  |  |  |
| generic_val | arb_Arab |  | 0.718 | 32.000 | 34837.000 |
| generic_val | cmn_Hani |  | 0.712 | 32.000 | 38861.000 |
| generic_val | deu_Latn |  | 0.725 | 32.000 | 25396.000 |
| generic_val | en |  | 0.772 | 32.000 | 643123.000 |
| generic_val | fra_Latn |  | 0.747 | 32.000 | 18036.000 |
| generic_val | hin_Deva |  | 0.751 | 32.000 | 31993.000 |
| generic_val | ita_Latn |  | 0.732 | 32.000 | 26097.000 |
| generic_val | jpn_Jpan |  | 0.710 | 32.000 | 26726.000 |
| generic_val | kor_Hang |  | 0.733 | 32.000 | 25161.000 |
| generic_val | por_Latn |  | 0.744 | 32.000 | 17871.000 |
| generic_val | rus_Cyrl |  | 0.720 | 32.000 | 32300.000 |
| generic_val | spa_Latn |  | 0.740 | 32.000 | 22090.000 |
| generic_val | tur_Latn |  | 0.741 | 32.000 | 24639.000 |

**Latentes mortas**

| group | metric | value |
|---|---|---|
| training_counter | frac_dead | 0.000 |
| eval_tokens | frac_never_fired | 0.003 |

**Fidelidade a jusante (dev)**

| group | accuracy | delta_accuracy_pp | kl_nat | argmax_flip_rate | n_decisions |
|---|---|---|---|---|---|
| mean_non_massive_all | 0.267 | -6.111 | 0.396 | 0.554 |  |
| mean_non_massive_markers | 0.292 | -3.556 | 0.188 | 0.297 |  |
| original | 0.328 |  |  |  | 900.000 |
| replace_all_incl_massive | 0.326 | -0.222 | 0.013 | 0.088 |  |
| replace_non_massive_all | 0.321 | -0.667 | 0.013 | 0.089 |  |
| replace_non_massive_markers | 0.326 | -0.222 | 0.002 | 0.033 |  |
| zero_non_massive_all | 0.400 | 7.222 | 0.408 | 0.708 |  |
| zero_non_massive_markers | 0.336 | 0.778 | 0.342 | 0.439 |  |

**Sinalizadores do protocolo:** nenhum

### en_base_enc17_k32

**FVE e L0 (dev e texto genérico retido)**

| section | group | frac_massive | fve | l0 | n_tokens |
|---|---|---|---|---|---|
| domain_dev | all_non_massive |  | 0.912 | 32.000 | 254230.000 |
| domain_dev | all_tokens_literal |  | 0.957 | 32.000 | 269616.000 |
| domain_dev | cls | 0.000 |  |  |  |
| domain_dev | end_sep | 0.079 |  |  |  |
| domain_dev | head_sep | 1.000 |  |  |  |
| domain_dev | marker | 0.032 |  |  |  |
| domain_dev | markers | 0.032 | 0.913 | 32.000 | 3060.000 |
| domain_dev | massive |  | 0.932 | 32.000 | 15386.000 |
| domain_dev | option_text | 0.072 | 0.952 | 32.000 | 36587.000 |
| domain_dev | question | 0.101 | 0.966 | 32.000 | 11989.000 |
| domain_dev | special |  | 0.960 | 32.000 | 2700.000 |
| domain_dev | state | 0.048 | 0.899 | 32.000 | 200865.000 |
| generic_val | all_non_massive |  | 0.719 | 32.000 | 974694.000 |
| generic_val | all_tokens_literal | 0.025 |  |  |  |
| generic_val | en |  | 0.719 | 32.000 | 974694.000 |

**Latentes mortas**

| group | metric | value |
|---|---|---|
| training_counter | frac_dead | 0.000 |
| eval_tokens | frac_never_fired | 0.007 |

**Fidelidade a jusante (dev)**

| group | accuracy | delta_accuracy_pp | kl_nat | argmax_flip_rate | n_decisions |
|---|---|---|---|---|---|
| mean_non_massive_all | 0.246 | -9.778 | 0.172 | 0.539 |  |
| mean_non_massive_markers | 0.313 | -3.000 | 0.077 | 0.174 |  |
| original | 0.343 |  |  |  | 900.000 |
| replace_all_incl_massive | 0.339 | -0.444 | 0.006 | 0.050 |  |
| replace_non_massive_all | 0.341 | -0.222 | 0.006 | 0.048 |  |
| replace_non_massive_markers | 0.343 | 0.000 | 0.000 | 0.012 |  |
| zero_non_massive_all | 0.277 | -6.667 | 0.170 | 0.569 |  |
| zero_non_massive_markers | 0.324 | -1.889 | 0.089 | 0.266 |  |

**Sinalizadores do protocolo:** 3.2% dos marcadores são massivos (> 1%)


## Features de ml_enc08_k32


**Contagens: teste ingênuo × Cochran-Mantel-Haenszel** (conjunto de descoberta; PROTOCOLO emenda 3)

| visão | critério | features com suporte | ingênuo q<=0,05 | CMH (estrato da visão) mais erro | CMH menos erro | CMH 1ª passada mais erro | CMH 1ª passada menos erro |
|---|---|---|---|---|---|---|---|
| pred | error | 543 | 364 | 9 | 11 | 9 | 11 |
| corr | error | 625 | 476 | 68 | 65 | 221 | 198 |
| state | error | 10213 | 3226 | 324 | 544 | 42 | 77 |
| pred | probe_right_laya_wrong | 468 | 348 | 43 | 43 | 43 | 43 |
| corr | probe_right_laya_wrong | 472 | 368 | 18 | 13 | 151 | 147 |
| state | probe_right_laya_wrong | 9441 | 3869 | 979 | 802 | 345 | 557 |

**Reprodutibilidade descoberta -> dev/teste** (todas as features com suporte nos dois lados, sem depender da seleção)

| visão | critério | split | features nos dois | Spearman z | Spearman lift ingênuo | sinal igual no top-20 | sinal igual no top-100 |
|---|---|---|---|---|---|---|---|
| pred | error | dev | 224 | 0.480 | 0.893 | 0.950 | 0.840 |
| pred | error | test | 365 | 0.459 | 0.926 | 0.950 | 0.850 |
| corr | error | dev | 218 | 0.657 | 0.932 | 1.000 | 0.880 |
| corr | error | test | 365 | 0.632 | 0.966 | 1.000 | 0.840 |
| state | error | dev | 7030 | 0.486 | 0.568 | 1.000 | 0.990 |
| state | error | test | 8650 | 0.418 | 0.526 | 1.000 | 1.000 |
| pred | probe_right_laya_wrong | dev | 129 | 0.762 | 0.925 | 1.000 | 0.860 |
| pred | probe_right_laya_wrong | test | 235 | 0.575 | 0.958 | 0.950 | 0.790 |
| corr | probe_right_laya_wrong | dev | 141 | 0.645 | 0.957 | 0.950 | 0.820 |
| corr | probe_right_laya_wrong | test | 248 | 0.542 | 0.974 | 0.800 | 0.790 |
| state | probe_right_laya_wrong | dev | 5870 | 0.688 | 0.735 | 1.000 | 1.000 |
| state | probe_right_laya_wrong | test | 7690 | 0.569 | 0.676 | 1.000 | 1.000 |


**pred / error**: 9 features com q(CMH) <= 0,05 e mais erro (de 543 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 3898.000 | 91.000 | 0.780 | 1.212 | 4.301 | 5.311 | 0.000 |  | 18.000 | 2.257 | 34.000 |
| 8265.000 | 159.000 | 0.635 | 0.987 | 2.670 | 4.202 | 0.002 | 1.997 | 34.000 | 0.997 | 68.000 |
| 10432.000 | 529.000 | 0.739 | 1.148 | 1.768 | 4.009 | 0.004 | 0.954 | 75.000 | 0.892 | 194.000 |
| 8477.000 | 25.000 | 0.640 | 0.994 | 4.774 | 3.740 | 0.009 |  | 6.000 |  | 9.000 |
| 5519.000 | 252.000 | 0.643 | 0.999 | 1.821 | 3.537 | 0.017 | 3.734 | 37.000 | 2.702 | 98.000 |
| 4603.000 | 627.000 | 0.579 | 0.899 | 1.594 | 3.404 | 0.026 | 0.143 | 116.000 | 0.338 | 276.000 |
| 5969.000 | 188.000 | 0.436 | 0.678 | 23.040 | 3.371 | 0.027 | -0.607 | 31.000 | 1.944 | 74.000 |
| 871.000 | 64.000 | 0.562 | 0.874 | 2.592 | 3.240 | 0.035 |  | 15.000 | 2.429 | 32.000 |
| 9379.000 | 59.000 | 0.678 | 1.053 | 2.623 | 3.093 | 0.050 |  | 9.000 | 0.854 | 22.000 |


**corr / error**: 68 features com q(CMH) <= 0,05 e mais erro (de 625 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 4131.000 | 26.000 | 0.962 | 1.494 | 99.306 | 7.257 | 0.000 |  | 5.000 |  | 3.000 |
| 7763.000 | 33.000 | 0.909 | 1.412 | 49.081 | 7.145 | 0.000 |  | 6.000 |  | 10.000 |
| 6612.000 | 24.000 | 0.958 | 1.489 | 90.850 | 6.943 | 0.000 |  | 5.000 |  | 3.000 |
| 9656.000 | 35.000 | 0.714 | 1.110 | 10.388 | 6.083 | 0.000 |  | 7.000 |  | 4.000 |
| 2678.000 | 60.000 | 0.950 | 1.476 | 8.271 | 5.880 | 0.000 |  | 15.000 |  | 18.000 |
| 10217.000 | 1970.000 | 0.723 | 1.123 | 2.247 | 5.592 | 0.000 | 2.967 | 348.000 | 2.975 | 762.000 |
| 3987.000 | 284.000 | 0.831 | 1.291 | 3.210 | 5.450 | 0.000 | 3.691 | 48.000 | 3.065 | 112.000 |
| 279.000 | 348.000 | 0.871 | 1.353 | 4.870 | 5.375 | 0.000 | 2.997 | 64.000 | 3.180 | 133.000 |
| 1778.000 | 39.000 | 0.795 | 1.235 | 8.989 | 4.764 | 0.000 |  | 6.000 | 1.485 | 27.000 |
| 3863.000 | 821.000 | 0.782 | 1.215 | 1.898 | 4.636 | 0.000 | 2.140 | 144.000 | 1.376 | 290.000 |


**state / error**: 324 features com q(CMH) <= 0,05 e mais erro (de 10213 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 4315.000 | 3080.000 | 0.683 | 1.062 | 1.716 | 6.964 | 0.000 | 3.146 | 526.000 | 3.101 | 1187.000 |
| 5732.000 | 3038.000 | 0.686 | 1.066 | 1.845 | 6.836 | 0.000 | 3.168 | 517.000 | 2.800 | 1195.000 |
| 10287.000 | 394.000 | 0.713 | 1.108 | 2.150 | 6.173 | 0.000 | 2.998 | 90.000 | 2.716 | 146.000 |
| 6605.000 | 2818.000 | 0.669 | 1.039 | 1.565 | 5.634 | 0.000 | 1.604 | 448.000 | 4.097 | 1088.000 |
| 5235.000 | 4155.000 | 0.647 | 1.005 | 1.756 | 5.584 | 0.000 | 2.769 | 683.000 | 3.552 | 1639.000 |
| 2200.000 | 385.000 | 0.688 | 1.069 | 1.928 | 5.368 | 0.000 | 2.327 | 80.000 | 1.823 | 121.000 |
| 10325.000 | 1609.000 | 0.729 | 1.133 | 1.679 | 5.221 | 0.000 | 2.074 | 261.000 | 1.848 | 615.000 |
| 1648.000 | 270.000 | 0.722 | 1.122 | 2.034 | 5.113 | 0.000 | 0.975 | 45.000 | 1.963 | 98.000 |
| 4610.000 | 617.000 | 0.663 | 1.030 | 1.736 | 5.050 | 0.000 | 2.710 | 117.000 | 1.741 | 247.000 |
| 10304.000 | 2758.000 | 0.674 | 1.047 | 1.603 | 4.959 | 0.000 | 1.206 | 454.000 | 3.527 | 1075.000 |


**pred / probe_right_laya_wrong**: 43 features com q(CMH) <= 0,05 e mais erro (de 468 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 3898.000 | 41.000 | 0.976 | 1.508 | 88.889 | 7.303 | 0.000 |  | 11.000 |  | 10.000 |
| 5519.000 | 117.000 | 0.709 | 1.096 | 6.212 | 6.789 | 0.000 |  | 17.000 | 3.387 | 31.000 |
| 871.000 | 40.000 | 0.650 | 1.005 | 15.362 | 6.136 | 0.000 |  | 11.000 |  | 19.000 |
| 6160.000 | 84.000 | 0.702 | 1.086 | 7.774 | 5.853 | 0.000 |  | 15.000 | 3.009 | 29.000 |
| 8265.000 | 70.000 | 0.614 | 0.949 | inf | 5.644 | 0.000 |  | 15.000 | 3.593 | 36.000 |
| 9247.000 | 1147.000 | 0.600 | 0.927 | 3.507 | 5.474 | 0.000 | 1.594 | 209.000 | 3.854 | 458.000 |
| 7763.000 | 92.000 | 1.000 | 1.546 | inf | 4.881 | 0.000 |  | 19.000 | 1.633 | 34.000 |
| 2678.000 | 81.000 | 0.975 | 1.507 | inf | 4.839 | 0.000 |  | 16.000 | 1.633 | 33.000 |
| 12224.000 | 376.000 | 0.758 | 1.171 | 19.556 | 4.703 | 0.000 | 0.974 | 91.000 | 2.893 | 154.000 |
| 4779.000 | 183.000 | 0.372 | 0.574 | 4.174 | 4.320 | 0.000 | 2.178 | 33.000 | 1.629 | 74.000 |


**corr / probe_right_laya_wrong**: 18 features com q(CMH) <= 0,05 e mais erro (de 472 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 1495.000 | 627.000 | 0.797 | 1.232 | 6.764 | 5.783 | 0.000 | 3.344 | 105.000 | 2.279 | 218.000 |
| 10432.000 | 277.000 | 0.819 | 1.267 | 3.103 | 4.626 | 0.000 | 4.026 | 40.000 | 2.325 | 97.000 |
| 3863.000 | 412.000 | 0.842 | 1.302 | 3.334 | 4.618 | 0.000 | 5.093 | 65.000 | 2.596 | 137.000 |
| 6060.000 | 454.000 | 0.756 | 1.168 | 3.025 | 4.198 | 0.001 | 1.553 | 92.000 | -1.467 | 191.000 |
| 9598.000 | 1244.000 | 0.784 | 1.211 | 2.180 | 4.067 | 0.002 | 3.241 | 203.000 | 0.606 | 487.000 |
| 72.000 | 25.000 | 0.640 | 0.989 | 5.399 | 3.775 | 0.006 |  | 4.000 |  | 10.000 |
| 2018.000 | 89.000 | 1.000 | 1.546 | inf | 3.510 | 0.012 |  | 10.000 | -0.319 | 24.000 |
| 5801.000 | 196.000 | 0.541 | 0.836 | 1.939 | 3.392 | 0.017 | 3.364 | 33.000 | 2.733 | 60.000 |
| 1336.000 | 230.000 | 0.609 | 0.941 | 2.186 | 3.347 | 0.019 | 3.439 | 39.000 | 0.398 | 76.000 |
| 7103.000 | 25.000 | 0.960 | 1.484 | inf | 3.300 | 0.022 |  | 1.000 |  | 13.000 |


**state / probe_right_laya_wrong**: 979 features com q(CMH) <= 0,05 e mais erro (de 9441 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 10287.000 | 213.000 | 0.822 | 1.270 | 5.397 | 8.521 | 0.000 | 3.483 | 57.000 | 2.961 | 85.000 |
| 5235.000 | 2261.000 | 0.648 | 1.001 | 3.079 | 7.861 | 0.000 | 2.711 | 386.000 | 4.578 | 897.000 |
| 4315.000 | 1649.000 | 0.700 | 1.082 | 2.421 | 7.800 | 0.000 | 2.682 | 296.000 | 2.957 | 627.000 |
| 5065.000 | 2062.000 | 0.631 | 0.976 | 2.787 | 7.666 | 0.000 | 3.549 | 344.000 | 3.147 | 781.000 |
| 4610.000 | 351.000 | 0.721 | 1.114 | 3.230 | 7.438 | 0.000 | 3.594 | 73.000 | 1.481 | 158.000 |
| 11929.000 | 2098.000 | 0.629 | 0.972 | 3.233 | 7.373 | 0.000 | 4.339 | 344.000 | 3.940 | 807.000 |
| 9977.000 | 2097.000 | 0.629 | 0.971 | 3.231 | 7.371 | 0.000 | 4.339 | 344.000 | 3.940 | 807.000 |
| 2200.000 | 217.000 | 0.756 | 1.168 | 3.754 | 7.257 | 0.000 | 2.726 | 51.000 | 1.528 | 74.000 |
| 9687.000 | 2195.000 | 0.660 | 1.020 | 2.242 | 6.968 | 0.000 | 2.090 | 401.000 | 2.277 | 843.000 |
| 5732.000 | 1669.000 | 0.701 | 1.083 | 2.491 | 6.789 | 0.000 | 3.100 | 298.000 | 0.957 | 643.000 |


**Atribuição no marcador previsto: features que empurram para a resposta errada (atribuição média mais negativa)** (conjunto de descoberta, com dev e teste ao lado)

| feature | erros | atribuição média | desvio | fração negativa | média dev | média test |
|---|---|---|---|---|---|---|
| 9247.0000 | 3283.0000 | -0.0128 | 0.0375 | 0.2415 | -0.0152 | -0.0121 |
| 9598.0000 | 3283.0000 | -0.0033 | 0.0087 | 0.3399 | -0.0031 | -0.0034 |
| 5061.0000 | 3283.0000 | -0.0032 | 0.0118 | 0.0737 | -0.0028 | -0.0027 |
| 2944.0000 | 3283.0000 | -0.0019 | 0.0070 | 0.0914 | -0.0017 | -0.0017 |
| 5066.0000 | 3283.0000 | -0.0014 | 0.0048 | 0.0868 | -0.0012 | -0.0012 |
| 4538.0000 | 3283.0000 | -0.0011 | 0.0055 | 0.1721 | -0.0011 | -0.0012 |
| 2125.0000 | 3283.0000 | -0.0009 | 0.0031 | 0.1045 | -0.0010 | -0.0009 |
| 174.0000 | 3283.0000 | -0.0009 | 0.0035 | 0.0789 | -0.0008 | -0.0007 |
| 10951.0000 | 3283.0000 | -0.0008 | 0.0026 | 0.1033 | -0.0012 | -0.0009 |
| 10581.0000 | 3283.0000 | -0.0008 | 0.0486 | 0.3375 | -0.0024 | -0.0007 |

**Atribuição no marcador previsto: features que empurram para a correta (mais positiva)** (conjunto de descoberta, com dev e teste ao lado)

| feature | erros | atribuição média | desvio | fração negativa | média dev | média test |
|---|---|---|---|---|---|---|
| 11755.0000 | 3283.0000 | 0.1845 | 0.2887 | 0.2400 | 0.1824 | 0.1761 |
| 10060.0000 | 3283.0000 | 0.0673 | 0.0589 | 0.0326 | 0.0650 | 0.0676 |
| 926.0000 | 3283.0000 | 0.0478 | 0.1193 | 0.3457 | 0.0455 | 0.0464 |
| 5674.0000 | 3283.0000 | 0.0340 | 0.0363 | 0.0228 | 0.0354 | 0.0347 |
| 2474.0000 | 3283.0000 | 0.0320 | 0.0515 | 0.2979 | 0.0330 | 0.0344 |
| 3929.0000 | 3283.0000 | 0.0244 | 0.0282 | 0.0859 | 0.0261 | 0.0258 |
| 10623.0000 | 3283.0000 | 0.0190 | 0.0353 | 0.0819 | 0.0206 | 0.0213 |
| 7627.0000 | 3283.0000 | 0.0184 | 0.0261 | 0.0091 | 0.0174 | 0.0194 |
| 3713.0000 | 3283.0000 | 0.0164 | 0.0275 | 0.1447 | 0.0174 | 0.0177 |
| 4285.0000 | 3283.0000 | 0.0104 | 0.0278 | 0.3567 | 0.0099 | 0.0105 |


**Viés de prior (Laya escolhe a opção da hipótese mais do que o gabarito?)**

| hypothesis | subset | n | p_laya_picks | p_gold_is | diff | laya_picks_wilson_lo | laya_picks_wilson_hi |
|---|---|---|---|---|---|---|---|
| score_highest_level | discovery | 2040 | 0.196 | 0.260 | -0.064 | 0.179 | 0.214 |
| score_highest_level | dev | 360 | 0.153 | 0.256 | -0.103 | 0.119 | 0.194 |
| score_highest_level | test | 800 | 0.176 | 0.269 | -0.092 | 0.151 | 0.204 |
| choice_majority_label | discovery | 1541 | 0.173 | 0.473 | -0.300 | 0.155 | 0.193 |
| choice_majority_label | dev | 259 | 0.212 | 0.483 | -0.270 | 0.167 | 0.266 |
| choice_majority_label | test | 600 | 0.177 | 0.418 | -0.242 | 0.148 | 0.209 |
| noul_true | discovery | 1519 | 0.757 | 0.510 | 0.247 | 0.735 | 0.778 |
| noul_true | dev | 281 | 0.769 | 0.434 | 0.335 | 0.716 | 0.814 |
| noul_true | test | 600 | 0.772 | 0.508 | 0.263 | 0.736 | 0.803 |
| noul_false | discovery | 1519 | 0.243 | 0.490 | -0.247 | 0.222 | 0.265 |
| noul_false | dev | 281 | 0.231 | 0.566 | -0.335 | 0.186 | 0.284 |
| noul_false | test | 600 | 0.228 | 0.492 | -0.263 | 0.197 | 0.264 |


## Features de ml_enc17_k32


**Contagens: teste ingênuo × Cochran-Mantel-Haenszel** (conjunto de descoberta; PROTOCOLO emenda 3)

| visão | critério | features com suporte | ingênuo q<=0,05 | CMH (estrato da visão) mais erro | CMH menos erro | CMH 1ª passada mais erro | CMH 1ª passada menos erro |
|---|---|---|---|---|---|---|---|
| pred | error | 626 | 334 | 11 | 16 | 11 | 16 |
| corr | error | 692 | 500 | 106 | 110 | 263 | 230 |
| state | error | 10850 | 3584 | 288 | 569 | 44 | 79 |
| pred | probe_right_laya_wrong | 491 | 301 | 36 | 27 | 36 | 27 |
| corr | probe_right_laya_wrong | 514 | 395 | 48 | 56 | 172 | 156 |
| state | probe_right_laya_wrong | 10261 | 4379 | 721 | 815 | 193 | 267 |

**Reprodutibilidade descoberta -> dev/teste** (todas as features com suporte nos dois lados, sem depender da seleção)

| visão | critério | split | features nos dois | Spearman z | Spearman lift ingênuo | sinal igual no top-20 | sinal igual no top-100 |
|---|---|---|---|---|---|---|---|
| pred | error | dev | 248 | 0.470 | 0.867 | 0.750 | 0.770 |
| pred | error | test | 418 | 0.444 | 0.897 | 0.950 | 0.820 |
| corr | error | dev | 251 | 0.712 | 0.952 | 1.000 | 0.900 |
| corr | error | test | 429 | 0.774 | 0.964 | 1.000 | 0.980 |
| state | error | dev | 8521 | 0.450 | 0.521 | 1.000 | 0.990 |
| state | error | test | 9742 | 0.445 | 0.531 | 1.000 | 1.000 |
| pred | probe_right_laya_wrong | dev | 154 | 0.634 | 0.923 | 0.950 | 0.870 |
| pred | probe_right_laya_wrong | test | 287 | 0.634 | 0.912 | 1.000 | 0.900 |
| corr | probe_right_laya_wrong | dev | 157 | 0.718 | 0.963 | 1.000 | 0.850 |
| corr | probe_right_laya_wrong | test | 304 | 0.725 | 0.967 | 1.000 | 0.980 |
| state | probe_right_laya_wrong | dev | 7565 | 0.555 | 0.694 | 1.000 | 1.000 |
| state | probe_right_laya_wrong | test | 9008 | 0.565 | 0.672 | 1.000 | 1.000 |


**pred / error**: 11 features com q(CMH) <= 0,05 e mais erro (de 626 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 5487.000 | 400.000 | 0.848 | 1.317 | 4.582 | 6.206 | 0.000 | 1.727 | 81.000 | 3.659 | 160.000 |
| 6763.000 | 860.000 | 0.593 | 0.921 | 1.503 | 4.047 | 0.005 | -0.100 | 142.000 | 0.339 | 324.000 |
| 9155.000 | 43.000 | 0.698 | 1.084 | 4.541 | 3.734 | 0.013 |  | 6.000 |  | 12.000 |
| 5073.000 | 387.000 | 0.682 | 1.060 | 1.751 | 3.689 | 0.013 | 0.991 | 70.000 | 2.256 | 144.000 |
| 1453.000 | 93.000 | 0.473 | 0.735 | 2.511 | 3.557 | 0.018 |  | 11.000 | -0.198 | 35.000 |
| 9252.000 | 76.000 | 0.461 | 0.715 | 2.756 | 3.417 | 0.024 |  | 12.000 | 0.283 | 22.000 |
| 4805.000 | 62.000 | 0.968 | 1.503 | 8.509 | 3.412 | 0.024 |  | 15.000 | 1.751 | 22.000 |
| 6778.000 | 624.000 | 0.747 | 1.160 | 1.662 | 3.389 | 0.024 | 0.047 | 99.000 | 0.851 | 237.000 |
| 1678.000 | 237.000 | 0.717 | 1.114 | 1.770 | 3.329 | 0.027 | 2.256 | 44.000 | 2.221 | 83.000 |
| 10601.000 | 478.000 | 0.695 | 1.079 | 1.671 | 3.226 | 0.037 | 1.332 | 66.000 | 1.507 | 163.000 |


**corr / error**: 106 features com q(CMH) <= 0,05 e mais erro (de 692 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 6763.000 | 1861.000 | 0.812 | 1.261 | 4.055 | 13.237 | 0.000 | 5.035 | 344.000 | 6.574 | 718.000 |
| 7314.000 | 2988.000 | 0.743 | 1.155 | 4.645 | 12.248 | 0.000 | 1.825 | 536.000 | 7.559 | 1164.000 |
| 3684.000 | 1432.000 | 0.844 | 1.310 | 3.546 | 10.816 | 0.000 | 3.105 | 270.000 | 5.460 | 583.000 |
| 6438.000 | 1316.000 | 0.852 | 1.323 | 3.869 | 9.201 | 0.000 | 2.631 | 244.000 | 5.677 | 513.000 |
| 9368.000 | 998.000 | 0.808 | 1.255 | 4.045 | 8.955 | 0.000 | 2.808 | 175.000 | 3.954 | 373.000 |
| 4319.000 | 222.000 | 0.869 | 1.351 | 6.741 | 8.482 | 0.000 | 4.219 | 40.000 | 4.562 | 97.000 |
| 4805.000 | 76.000 | 0.974 | 1.513 | 19.083 | 6.949 | 0.000 |  | 14.000 |  | 13.000 |
| 8536.000 | 225.000 | 0.996 | 1.547 | 100.314 | 6.700 | 0.000 | 3.024 | 48.000 | 3.468 | 77.000 |
| 4851.000 | 161.000 | 0.863 | 1.341 | 10.246 | 6.698 | 0.000 | 1.634 | 29.000 | 2.228 | 77.000 |
| 13.000 | 583.000 | 0.746 | 1.159 | 2.397 | 6.271 | 0.000 | 2.408 | 96.000 | 3.669 | 231.000 |


**state / error**: 288 features com q(CMH) <= 0,05 e mais erro (de 10850 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 4962.000 | 3404.000 | 0.671 | 1.042 | 1.516 | 5.663 | 0.000 | 1.621 | 527.000 | 1.946 | 1305.000 |
| 8062.000 | 3261.000 | 0.660 | 1.025 | 1.573 | 5.537 | 0.000 | 3.066 | 524.000 | 3.380 | 1303.000 |
| 9349.000 | 1567.000 | 0.698 | 1.085 | 1.507 | 5.129 | 0.000 | 1.746 | 277.000 | 2.803 | 622.000 |
| 2280.000 | 3292.000 | 0.646 | 1.003 | 1.440 | 5.084 | 0.000 | 1.864 | 544.000 | 1.846 | 1327.000 |
| 6930.000 | 1411.000 | 0.748 | 1.163 | 1.838 | 5.077 | 0.000 | 2.293 | 231.000 | 1.192 | 554.000 |
| 4051.000 | 2646.000 | 0.674 | 1.047 | 1.647 | 5.047 | 0.000 | 3.092 | 416.000 | 2.274 | 1034.000 |
| 6221.000 | 509.000 | 0.717 | 1.114 | 1.742 | 5.038 | 0.000 | 1.424 | 104.000 | 2.111 | 148.000 |
| 5165.000 | 2304.000 | 0.701 | 1.089 | 1.483 | 4.942 | 0.000 | -0.575 | 398.000 | 1.153 | 839.000 |
| 11909.000 | 666.000 | 0.655 | 1.017 | 1.671 | 4.796 | 0.000 | 2.304 | 125.000 | 1.466 | 260.000 |
| 5680.000 | 402.000 | 0.689 | 1.070 | 1.808 | 4.779 | 0.000 | 1.510 | 98.000 | 1.484 | 174.000 |


**pred / probe_right_laya_wrong**: 36 features com q(CMH) <= 0,05 e mais erro (de 491 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 6763.000 | 489.000 | 0.575 | 0.949 | 2.118 | 5.268 | 0.000 | 0.187 | 83.000 | 1.462 | 173.000 |
| 1678.000 | 126.000 | 0.690 | 1.141 | 3.244 | 5.266 | 0.000 | 2.192 | 24.000 | 3.240 | 40.000 |
| 4805.000 | 30.000 | 1.000 | 1.652 | inf | 5.129 | 0.000 |  | 10.000 |  | 14.000 |
| 5487.000 | 272.000 | 0.926 | 1.530 | 4.986 | 4.488 | 0.000 | 0.754 | 67.000 | 3.382 | 110.000 |
| 8514.000 | 196.000 | 0.316 | 0.523 | 4.958 | 4.467 | 0.000 | 1.775 | 37.000 | 0.888 | 92.000 |
| 518.000 | 65.000 | 0.615 | 1.017 | 3.512 | 4.210 | 0.001 |  | 11.000 |  | 18.000 |
| 9155.000 | 25.000 | 0.680 | 1.123 | 10.081 | 4.005 | 0.002 |  | 4.000 |  | 10.000 |
| 10601.000 | 236.000 | 0.661 | 1.092 | 2.501 | 3.954 | 0.002 | 0.815 | 28.000 | 2.998 | 80.000 |
| 2819.000 | 27.000 | 0.889 | 1.468 | 5.685 | 3.779 | 0.004 |  | 5.000 |  | 8.000 |
| 4880.000 | 93.000 | 0.806 | 1.332 | 2.654 | 3.617 | 0.006 | 2.126 | 21.000 | 3.378 | 35.000 |


**corr / probe_right_laya_wrong**: 48 features com q(CMH) <= 0,05 e mais erro (de 514 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 7314.000 | 1553.000 | 0.722 | 1.193 | 6.822 | 10.403 | 0.000 | 1.402 | 273.000 | 5.331 | 602.000 |
| 6763.000 | 971.000 | 0.786 | 1.298 | 3.463 | 8.684 | 0.000 | 3.516 | 197.000 | 4.294 | 381.000 |
| 3684.000 | 818.000 | 0.813 | 1.343 | 4.052 | 8.253 | 0.000 | 1.986 | 156.000 | 2.655 | 338.000 |
| 9368.000 | 458.000 | 0.825 | 1.363 | 4.725 | 6.885 | 0.000 | 3.422 | 82.000 | 3.863 | 177.000 |
| 4319.000 | 117.000 | 0.838 | 1.384 | 6.840 | 6.861 | 0.000 | 3.991 | 25.000 | 4.738 | 42.000 |
| 6438.000 | 837.000 | 0.851 | 1.405 | 3.609 | 6.339 | 0.000 | 1.245 | 167.000 | 3.752 | 329.000 |
| 4066.000 | 49.000 | 0.837 | 1.382 | 7.915 | 5.588 | 0.000 |  | 5.000 | 3.642 | 23.000 |
| 4476.000 | 1244.000 | 0.662 | 1.094 | 2.585 | 5.391 | 0.000 | 2.375 | 222.000 | 5.179 | 489.000 |
| 12216.000 | 545.000 | 0.727 | 1.200 | 2.914 | 5.351 | 0.000 | 2.138 | 97.000 | 3.143 | 201.000 |
| 8213.000 | 588.000 | 0.736 | 1.216 | 2.976 | 5.141 | 0.000 | 1.295 | 88.000 | 1.740 | 194.000 |


**state / probe_right_laya_wrong**: 721 features com q(CMH) <= 0,05 e mais erro (de 10261 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 4051.000 | 1455.000 | 0.646 | 1.067 | 2.534 | 7.016 | 0.000 | 3.452 | 223.000 | 3.459 | 542.000 |
| 5680.000 | 233.000 | 0.717 | 1.184 | 3.349 | 6.792 | 0.000 | 1.457 | 61.000 | 2.915 | 96.000 |
| 28.000 | 115.000 | 0.765 | 1.264 | 5.040 | 6.571 | 0.000 | 1.724 | 31.000 | 2.862 | 38.000 |
| 10312.000 | 1437.000 | 0.645 | 1.066 | 2.361 | 6.392 | 0.000 | 3.221 | 221.000 | 3.882 | 527.000 |
| 5961.000 | 209.000 | 0.699 | 1.154 | 2.986 | 6.220 | 0.000 | 1.288 | 37.000 | 2.258 | 75.000 |
| 9349.000 | 851.000 | 0.684 | 1.130 | 2.004 | 6.218 | 0.000 | 2.223 | 153.000 | 2.466 | 314.000 |
| 584.000 | 2033.000 | 0.596 | 0.985 | 1.895 | 6.189 | 0.000 | 1.626 | 347.000 | 2.943 | 819.000 |
| 8062.000 | 1741.000 | 0.612 | 1.011 | 1.946 | 6.159 | 0.000 | 3.524 | 277.000 | 3.467 | 695.000 |
| 1538.000 | 1447.000 | 0.648 | 1.071 | 2.313 | 6.159 | 0.000 | 3.390 | 225.000 | 3.718 | 524.000 |
| 5155.000 | 1419.000 | 0.644 | 1.064 | 2.300 | 6.140 | 0.000 | 3.405 | 217.000 | 3.882 | 517.000 |


**Atribuição no marcador previsto: features que empurram para a resposta errada (atribuição média mais negativa)** (conjunto de descoberta, com dev e teste ao lado)

| feature | erros | atribuição média | desvio | fração negativa | média dev | média test |
|---|---|---|---|---|---|---|
| 10589.0000 | 3283.0000 | -0.1086 | 0.1203 | 0.8212 | -0.0988 | -0.1075 |
| 723.0000 | 3283.0000 | -0.0832 | 0.0964 | 0.6171 | -0.0856 | -0.0823 |
| 3929.0000 | 3283.0000 | -0.0240 | 0.0949 | 0.2717 | -0.0304 | -0.0279 |
| 1317.0000 | 3283.0000 | -0.0178 | 0.0232 | 0.5702 | -0.0181 | -0.0189 |
| 5078.0000 | 3283.0000 | -0.0164 | 0.0682 | 0.4438 | -0.0153 | -0.0155 |
| 9742.0000 | 3283.0000 | -0.0113 | 0.0344 | 0.1432 | -0.0101 | -0.0102 |
| 796.0000 | 3283.0000 | -0.0107 | 0.0330 | 0.2796 | -0.0130 | -0.0127 |
| 4001.0000 | 3283.0000 | -0.0105 | 0.0315 | 0.2154 | -0.0131 | -0.0114 |
| 2932.0000 | 3283.0000 | -0.0105 | 0.0335 | 0.1553 | -0.0132 | -0.0101 |
| 8084.0000 | 3283.0000 | -0.0099 | 0.0325 | 0.2205 | -0.0082 | -0.0104 |

**Atribuição no marcador previsto: features que empurram para a correta (mais positiva)** (conjunto de descoberta, com dev e teste ao lado)

| feature | erros | atribuição média | desvio | fração negativa | média dev | média test |
|---|---|---|---|---|---|---|
| 2892.0000 | 3283.0000 | 0.2374 | 0.3466 | 0.2498 | 0.2470 | 0.2435 |
| 3412.0000 | 3283.0000 | 0.0826 | 0.1354 | 0.0996 | 0.0914 | 0.0890 |
| 7314.0000 | 3283.0000 | 0.0554 | 0.0786 | 0.0088 | 0.0536 | 0.0616 |
| 4787.0000 | 3283.0000 | 0.0478 | 0.2465 | 0.3996 | 0.0310 | 0.0408 |
| 4382.0000 | 3283.0000 | 0.0172 | 0.0428 | 0.0006 | 0.0181 | 0.0182 |
| 7603.0000 | 3283.0000 | 0.0137 | 0.0744 | 0.0052 | 0.0128 | 0.0141 |
| 10189.0000 | 3283.0000 | 0.0123 | 0.0954 | 0.1940 | 0.0101 | 0.0121 |
| 4476.0000 | 3283.0000 | 0.0122 | 0.0309 | 0.0110 | 0.0095 | 0.0103 |
| 11737.0000 | 3283.0000 | 0.0101 | 0.0270 | 0.0189 | 0.0135 | 0.0116 |
| 6438.0000 | 3283.0000 | 0.0100 | 0.0347 | 0.0000 | 0.0096 | 0.0101 |


**Viés de prior (Laya escolhe a opção da hipótese mais do que o gabarito?)**

| hypothesis | subset | n | p_laya_picks | p_gold_is | diff | laya_picks_wilson_lo | laya_picks_wilson_hi |
|---|---|---|---|---|---|---|---|
| score_highest_level | discovery | 2040 | 0.196 | 0.260 | -0.064 | 0.179 | 0.214 |
| score_highest_level | dev | 360 | 0.153 | 0.256 | -0.103 | 0.119 | 0.194 |
| score_highest_level | test | 800 | 0.176 | 0.269 | -0.092 | 0.151 | 0.204 |
| choice_majority_label | discovery | 1541 | 0.173 | 0.473 | -0.300 | 0.155 | 0.193 |
| choice_majority_label | dev | 259 | 0.212 | 0.483 | -0.270 | 0.167 | 0.266 |
| choice_majority_label | test | 600 | 0.177 | 0.418 | -0.242 | 0.148 | 0.209 |
| noul_true | discovery | 1519 | 0.757 | 0.510 | 0.247 | 0.735 | 0.778 |
| noul_true | dev | 281 | 0.769 | 0.434 | 0.335 | 0.716 | 0.814 |
| noul_true | test | 600 | 0.772 | 0.508 | 0.263 | 0.736 | 0.803 |
| noul_false | discovery | 1519 | 0.243 | 0.490 | -0.247 | 0.222 | 0.265 |
| noul_false | dev | 281 | 0.231 | 0.566 | -0.335 | 0.186 | 0.284 |
| noul_false | test | 600 | 0.228 | 0.492 | -0.263 | 0.197 | 0.264 |


## Features de en_base_enc17_k32


**Contagens: teste ingênuo × Cochran-Mantel-Haenszel** (conjunto de descoberta; PROTOCOLO emenda 3)

| visão | critério | features com suporte | ingênuo q<=0,05 | CMH (estrato da visão) mais erro | CMH menos erro | CMH 1ª passada mais erro | CMH 1ª passada menos erro |
|---|---|---|---|---|---|---|---|
| pred | error | 510 | 289 | 50 | 47 | 50 | 47 |
| corr | error | 568 | 449 | 68 | 62 | 258 | 179 |
| state | error | 14071 | 1217 | 2715 | 1196 | 2321 | 1113 |

**Reprodutibilidade descoberta -> dev/teste** (todas as features com suporte nos dois lados, sem depender da seleção)

| visão | critério | split | features nos dois | Spearman z | Spearman lift ingênuo | sinal igual no top-20 | sinal igual no top-100 |
|---|---|---|---|---|---|---|---|
| pred | error | dev | 203 | 0.730 | 0.876 | 1.000 | 0.880 |
| pred | error | test | 310 | 0.758 | 0.872 | 1.000 | 0.940 |
| corr | error | dev | 235 | 0.710 | 0.963 | 1.000 | 0.920 |
| corr | error | test | 371 | 0.726 | 0.967 | 1.000 | 0.980 |
| state | error | dev | 10934 | 0.616 | 0.359 | 1.000 | 1.000 |
| state | error | test | 12651 | 0.689 | 0.421 | 1.000 | 1.000 |


**pred / error**: 50 features com q(CMH) <= 0,05 e mais erro (de 510 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 12639.000 | 1115.000 | 0.659 | 1.036 | 2.161 | 7.416 | 0.000 | 2.380 | 172.000 | 3.959 | 430.000 |
| 10151.000 | 530.000 | 0.887 | 1.394 | 5.024 | 7.390 | 0.000 | 2.297 | 85.000 | 5.383 | 206.000 |
| 10960.000 | 214.000 | 0.897 | 1.410 | 5.472 | 7.159 | 0.000 | 2.626 | 38.000 | 6.390 | 82.000 |
| 6418.000 | 1029.000 | 0.778 | 1.223 | 2.049 | 7.127 | 0.000 | 4.925 | 167.000 | 5.278 | 393.000 |
| 9716.000 | 339.000 | 0.838 | 1.317 | 2.971 | 6.854 | 0.000 | 3.796 | 61.000 | 4.319 | 134.000 |
| 11372.000 | 462.000 | 0.920 | 1.446 | 3.594 | 6.549 | 0.000 | 2.553 | 72.000 | 5.732 | 176.000 |
| 451.000 | 365.000 | 0.888 | 1.395 | 3.350 | 6.187 | 0.000 | 2.889 | 61.000 | 5.434 | 140.000 |
| 12037.000 | 292.000 | 0.873 | 1.373 | 3.175 | 6.180 | 0.000 | 2.150 | 49.000 | 4.520 | 122.000 |
| 14487.000 | 2770.000 | 0.686 | 1.077 | 3.019 | 6.087 | 0.000 | 1.438 | 471.000 | 5.893 | 1076.000 |
| 3734.000 | 443.000 | 0.867 | 1.362 | 2.618 | 5.594 | 0.000 | 2.106 | 80.000 | 4.684 | 178.000 |


**corr / error**: 68 features com q(CMH) <= 0,05 e mais erro (de 568 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 15421.000 | 1715.000 | 0.957 | 1.504 | 11.197 | 15.868 | 0.000 | 6.127 | 289.000 | 9.865 | 694.000 |
| 6869.000 | 139.000 | 0.942 | 1.481 | 407.500 | 10.533 | 0.000 |  | 19.000 | 2.648 | 47.000 |
| 1156.000 | 53.000 | 0.943 | 1.483 | inf | 8.449 | 0.000 |  | 12.000 |  | 17.000 |
| 6734.000 | 2336.000 | 0.660 | 1.037 | 14.603 | 7.379 | 0.000 | 3.276 | 407.000 | 5.501 | 899.000 |
| 10221.000 | 128.000 | 0.906 | 1.424 | 32.459 | 7.306 | 0.000 | 3.429 | 31.000 | 3.472 | 58.000 |
| 7939.000 | 89.000 | 0.933 | 1.466 | 14.096 | 7.036 | 0.000 | 2.963 | 20.000 | 3.295 | 37.000 |
| 15079.000 | 108.000 | 0.972 | 1.528 | 12.947 | 6.954 | 0.000 |  | 14.000 | 2.768 | 44.000 |
| 13523.000 | 204.000 | 0.877 | 1.379 | 6.091 | 6.740 | 0.000 | 3.364 | 28.000 | 3.055 | 85.000 |
| 14487.000 | 2759.000 | 0.684 | 1.075 | 4.843 | 6.598 | 0.000 | 1.327 | 465.000 | 5.662 | 1063.000 |
| 2633.000 | 103.000 | 0.573 | 0.900 | 8.861 | 6.412 | 0.000 |  | 14.000 | 5.899 | 33.000 |


**state / error**: 2715 features com q(CMH) <= 0,05 e mais erro (de 14071 com suporte); top 10 por z do CMH (descoberta) com os mesmos números em dev e teste:

| feature | ativas | taxa de erro | lift ingênuo | OR CMH | z CMH | q CMH | z CMH dev | ativas dev | z CMH test | ativas test |
|---|---|---|---|---|---|---|---|---|---|---|
| 9405.000 | 2602.000 | 0.632 | 0.994 | 3.475 | 11.451 | 0.000 | 3.504 | 423.000 | 7.647 | 1009.000 |
| 10507.000 | 2565.000 | 0.634 | 0.997 | 3.030 | 10.854 | 0.000 | 5.281 | 417.000 | 7.091 | 1001.000 |
| 5208.000 | 1326.000 | 0.674 | 1.060 | 3.785 | 10.181 | 0.000 | 4.512 | 221.000 | 6.746 | 511.000 |
| 3027.000 | 1446.000 | 0.669 | 1.052 | 3.217 | 10.124 | 0.000 | 3.482 | 250.000 | 6.602 | 563.000 |
| 7516.000 | 1302.000 | 0.672 | 1.056 | 3.919 | 10.093 | 0.000 | 5.097 | 219.000 | 6.762 | 503.000 |
| 5572.000 | 1338.000 | 0.673 | 1.058 | 3.601 | 10.027 | 0.000 | 4.882 | 224.000 | 6.755 | 529.000 |
| 12783.000 | 1320.000 | 0.672 | 1.056 | 3.692 | 9.950 | 0.000 | 4.812 | 219.000 | 6.776 | 508.000 |
| 5739.000 | 1293.000 | 0.669 | 1.051 | 3.820 | 9.914 | 0.000 | 4.931 | 215.000 | 6.773 | 508.000 |
| 953.000 | 1294.000 | 0.669 | 1.052 | 3.811 | 9.914 | 0.000 | 4.734 | 216.000 | 6.618 | 500.000 |
| 1072.000 | 1295.000 | 0.669 | 1.052 | 3.813 | 9.901 | 0.000 | 4.814 | 218.000 | 6.040 | 508.000 |





**Atribuição no marcador previsto: features que empurram para a resposta errada (atribuição média mais negativa)** (conjunto de descoberta, com dev e teste ao lado)

| feature | erros | atribuição média | desvio | fração negativa | média dev | média test |
|---|---|---|---|---|---|---|
| 2786.0000 | 3245.0000 | -0.1701 | 0.1923 | 0.8015 | -0.1552 | -0.1667 |
| 7558.0000 | 3245.0000 | -0.0871 | 0.0735 | 0.7994 | -0.0794 | -0.0902 |
| 13827.0000 | 3245.0000 | -0.0543 | 0.0524 | 0.7510 | -0.0506 | -0.0536 |
| 2130.0000 | 3245.0000 | -0.0436 | 0.0809 | 0.4499 | -0.0383 | -0.0390 |
| 9304.0000 | 3245.0000 | -0.0282 | 0.0563 | 0.4317 | -0.0267 | -0.0317 |
| 15124.0000 | 3245.0000 | -0.0082 | 0.0214 | 0.2536 | -0.0079 | -0.0081 |
| 6211.0000 | 3245.0000 | -0.0080 | 0.0316 | 0.1504 | -0.0062 | -0.0082 |
| 11955.0000 | 3245.0000 | -0.0076 | 0.0456 | 0.0659 | -0.0048 | -0.0064 |
| 4741.0000 | 3245.0000 | -0.0069 | 0.0136 | 0.3686 | -0.0057 | -0.0065 |
| 8309.0000 | 3245.0000 | -0.0057 | 0.0145 | 0.3525 | -0.0062 | -0.0062 |

**Atribuição no marcador previsto: features que empurram para a correta (mais positiva)** (conjunto de descoberta, com dev e teste ao lado)

| feature | erros | atribuição média | desvio | fração negativa | média dev | média test |
|---|---|---|---|---|---|---|
| 11133.0000 | 3245.0000 | 0.1103 | 0.0799 | 0.0222 | 0.1009 | 0.1062 |
| 2351.0000 | 3245.0000 | 0.0821 | 0.0707 | 0.0074 | 0.0715 | 0.0806 |
| 11776.0000 | 3245.0000 | 0.0730 | 0.0863 | 0.0604 | 0.0662 | 0.0725 |
| 15436.0000 | 3245.0000 | 0.0347 | 0.0376 | 0.0025 | 0.0329 | 0.0340 |
| 12238.0000 | 3245.0000 | 0.0343 | 0.0472 | 0.0000 | 0.0297 | 0.0344 |
| 4534.0000 | 3245.0000 | 0.0190 | 0.0339 | 0.1131 | 0.0192 | 0.0193 |
| 1384.0000 | 3245.0000 | 0.0148 | 0.0402 | 0.0083 | 0.0106 | 0.0122 |
| 3314.0000 | 3245.0000 | 0.0130 | 0.0320 | 0.0379 | 0.0141 | 0.0118 |
| 7445.0000 | 3245.0000 | 0.0123 | 0.0142 | 0.0539 | 0.0111 | 0.0117 |
| 6734.0000 | 3245.0000 | 0.0108 | 0.0151 | 0.0046 | 0.0113 | 0.0122 |


**Viés de prior (Laya escolhe a opção da hipótese mais do que o gabarito?)**

| hypothesis | subset | n | p_laya_picks | p_gold_is | diff | laya_picks_wilson_lo | laya_picks_wilson_hi |
|---|---|---|---|---|---|---|---|
| score_highest_level | discovery | 2040 | 0.043 | 0.260 | -0.217 | 0.035 | 0.053 |
| score_highest_level | dev | 360 | 0.031 | 0.256 | -0.225 | 0.017 | 0.054 |
| score_highest_level | test | 800 | 0.031 | 0.269 | -0.237 | 0.021 | 0.046 |
| choice_majority_label | discovery | 1541 | 0.197 | 0.473 | -0.276 | 0.178 | 0.218 |
| choice_majority_label | dev | 259 | 0.266 | 0.483 | -0.216 | 0.216 | 0.323 |
| choice_majority_label | test | 600 | 0.198 | 0.418 | -0.220 | 0.168 | 0.232 |
| noul_true | discovery | 1519 | 0.652 | 0.510 | 0.142 | 0.627 | 0.675 |
| noul_true | dev | 281 | 0.701 | 0.434 | 0.267 | 0.645 | 0.752 |
| noul_true | test | 600 | 0.673 | 0.508 | 0.165 | 0.635 | 0.710 |
| noul_false | discovery | 1519 | 0.348 | 0.490 | -0.142 | 0.325 | 0.373 |
| noul_false | dev | 281 | 0.299 | 0.566 | -0.267 | 0.248 | 0.355 |
| noul_false | test | 600 | 0.327 | 0.492 | -0.165 | 0.290 | 0.365 |


## Parte C: base x fine-tuned

**Entradas idênticas:** {"train": {"tokens_a": 1748500, "tokens_b": 1748500, "identical_token_ids_and_roles": true}, "test": {"tokens_a": 582370, "tokens_b": 582370, "identical_token_ids_and_roles": true}}

**FVE do SAE do base** (dev): no base {"all_non_massive": 0.9121961114915191, "markers": 0.9129714282828167}; no fine-tuned {"all_non_massive": 0.7743153823840143, "markers": 0.6517814624599467}; queda 0.1379.

note: "FVE drop > 0.05 but no --sae-ft given: the base SAE is used on the fine-tuned model unchanged"

Features com IC 95% do delta fora de zero (descoberta): 876

**Acurácia por configuração**

| model_config | split | subset | n | accuracy |
|---|---|---|---|---|
| base_default_512/192 | train | all | 6000 | 0.3615 |
| base_default_512/192 | train | dev | 900 | 0.3444 |
| base_default_512/192 | test | all | 2000 | 0.3625 |
| base_common_1024/256 | train | all | 6000 | 0.3607 |
| base_common_1024/256 | train | dev | 900 | 0.3433 |
| base_common_1024/256 | test | all | 2000 | 0.3630 |
| finetuned_1024/256 | train | all | 6000 | 0.8538 |
| finetuned_1024/256 | train | dev | 900 | 0.8567 |
| finetuned_1024/256 | test | all | 2000 | 0.7685 |

**Features que mais mudam (|delta| do contraste correta - erradas, IC 95% fora de zero, descoberta), top 10**

| feature | contrast_base | contrast_ft | delta | ci_lo | ci_hi | lift_error_base | lift_error_ft | mean_attr_base | mean_attr_ft |
|---|---|---|---|---|---|---|---|---|---|
| 2786.0000 | 0.3034 | 3.1370 | 2.8336 | 2.7134 | 2.9555 | 1.0173 | 1.0720 | -0.1701 | -0.1293 |
| 11776.0000 | 0.2365 | -1.0781 | -1.3146 | -1.4409 | -1.1787 | 1.0218 | 1.0162 | 0.0730 | 0.0742 |
| 7558.0000 | 0.2062 | 1.4808 | 1.2745 | 1.2303 | 1.3219 | 1.0470 | 1.0789 | -0.0871 | -0.0341 |
| 15421.0000 | -0.1064 | -1.3058 | -1.1994 | -1.2466 | -1.1538 | 0.9396 | 0.4478 | 0.0011 | 0.0011 |
| 12238.0000 | -0.1248 | -1.2519 | -1.1270 | -1.1683 | -1.0845 | 1.0699 | 1.3045 | 0.0343 | 0.0236 |
| 8737.0000 | -0.0268 | -0.7527 | -0.7259 | -0.7677 | -0.6863 | 1.1502 | 1.3975 | 0.0038 | 0.0034 |
| 9304.0000 | 0.4326 | 1.1497 | 0.7171 | 0.6864 | 0.7486 | 1.0206 | 1.1568 | -0.0282 | -0.0038 |
| 15436.0000 | -0.1380 | -0.7946 | -0.6566 | -0.6987 | -0.6156 | 1.0764 | 1.3227 | 0.0347 | 0.0148 |
| 11133.0000 | 0.1801 | -0.3399 | -0.5200 | -0.5807 | -0.4534 | 0.9930 | 1.0180 | 0.1103 | 0.0748 |
| 2130.0000 | 0.0657 | 0.4479 | 0.3823 | 0.3582 | 0.4077 | 1.1265 | 1.0032 | -0.0436 | 0.0007 |
