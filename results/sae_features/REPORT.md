# SAE e features que explicam o erro do Laya no typed-decisions (etapas 4 e 5)

Tudo aqui é **correlacional**, exceto a fidelidade a jusante (que mede o SAE, não as features). A **ablação causal** da seção 7 foi feita depois, com protocolo próprio. Critérios e limiares foram fixados antes dos jobs em `PROTOCOLO.md`; as oito
emendas (feitas por causa do que os dados mostraram; a 4 e a 5 são da ablação, a 6 da rotação e da semente, a 7 da causa da sensibilidade à posição, a 8 das camadas finais, da semente 1 das features e dos modelos em inglês) estão datadas lá e resumidas na seção 2 e nas seções 7 a 10. As tabelas completas, geradas dos artefatos
sem interpretação, estão em `REPORT_tables.md`.

## Resumo

- **Camadas:** `enc_17` (melhor do probe no dev, 0,597) e `enc_08` (primeira camada com ganho claro sobre o prior do dev: 0,583 contra 0,510).
  A regra literal "primeira que passa do prior" degenera (25 de 26 camadas já passam).
- **Os SAEs reconstroem bem e são fiéis a jusante.** Trocar a ativação da camada pela reconstrução do SAE muda a acurácia do Laya no dev em
  +0,2 p.p. (`enc_08`) e −0,7 p.p. (`enc_17`), com KL de 0,065 e 0,013 nat (limiar do protocolo: 3 p.p. e 0,10); trocar pela média ou por zeros dá KL ≈ 0,40.
  FVE no dev: 0,954 (`enc_08`) e 0,900 (`enc_17`). Um sinalizador disparou: em `enc_08` a FVE só nos marcadores de opção é 0,777 (0,18 abaixo da geral).
- **O viés de prior do Laya é forte e replica em dev e teste:** em `choice` escolhe a classe majoritária 17% das vezes (gabarito: 47%) e prefere o índice 3
  (47% contra 9%); em `noul` responde `true` em 76–77% (gabarito: 43–51%); em `score` subescolhe o nível mais alto (20% contra 26%).
- **O ranking ingênuo de features por taxa de erro mede principalmente qual opção Laya escolheu**, não "entendimento do caso": na visão do marcador da opção
  prevista, 364 features significativas viram 20 sob o teste estratificado (pergunta × opção). O que sobrevive replica em direção (top-20 por |z| com 75–100% de
  concordância de sinal em dev e teste), mas é fraco e, nos exemplos, **majoritariamente lexical** (tokens como ` river`, `_`, ` of`, ` over`). Não atribuí rótulos.
- **A informação mal lida existe e é mensurável:** 2.113 das 6.000 decisões de treino (probe fora da dobra) e 702 das 2.000 de teste são "o probe acerta, o Laya erra".
- **Ablação causal (seção 7):** zerar as 20 features de maior atribuição negativa no marcador da opção escolhida move a margem na direção da resposta correta e **replica no teste** nas duas camadas
  (muito acima de conjuntos aleatórios pareados). Mas o efeito **não é relevante nem específico**: em `enc_08` é ~2% da margem (corrige 0,8% dos erros) e em `enc_17` é ~20% da margem, porém a acurácia
  total **cai** (−0,7 p.p. no dev, −1,4 p.p. no teste), porque a intervenção quebra 17% das decisões certas. A previsão de primeira ordem acerta bem o efeito de features individuais (Spearman 0,90–0,93).
- **Rotação de opções (seção 8):** o viés de prior em `choice` **não é posicional** (maior desvio por posição 3,2 p.p., abaixo do critério de 5 p.p.): é de conteúdo. Em `score` o Laya quase nunca escolhe a primeira opção (−18 p.p.). Com a ordem balanceada, a acurácia do Laya (0,334) fica indistinguível do acaso (0,318). O probe de `enc_08` depende da ordem original (0,50 → 0,27 nas rotações); o de `enc_17` mantém conteúdo (0,52 → 0,42, acaso ~0,24).
- **Causa da sensibilidade à posição (seção 10):** em `score`, remover o subespaço de posição dos marcadores da camada 22 (ou da 14) reduz a taxa de troca em 16 a 18 p.p. (~20%) sem perder acurácia, e zerar só 5 a 20 features do SAE de `enc_17` a reduz em ~10 p.p.; em `enc_08` as features não têm efeito. A posição é legível em todas as camadas (acurácia ~1,0); uma sonda treinada nas rotações chega a 0,559, então o conteúdo está lá. O conjunto de camadas 14 + 22 chega a −20,5 p.p. (25%), o efeito se repete com a semente 1 do SAE, e **só o checkpoint multilíngue tem o fenômeno**: o Laya base em inglês quase não é sensível (troca 0,31 contra 0,83). Eu esperava poucos efeitos e errei.
- **Sementes (seção 9):** o efeito agregado da ablação se repete com a semente 1 nas duas camadas, mas só ~50% das features têm correspondente e a lista da ablação não é estável (15% e 35%): o ranking de features individuais depende da semente.
- **Parte C (base × fine-tuned em inglês):** entradas idênticas confirmadas; a configuração comum não distorce o base (0,3625 contra 0,3630). No base, marcadores massivos
  (fora do SAE) ocorrem **só na opção `true` das perguntas `noul`** (em 1/3 delas); nessas, o base responde `true` em 90–92% dos casos quando esse marcador é massivo e em 54–55% quando não é
  (no fine-tuned a relação some: 47% contra 56–57%). Mas atenção: a atribuição por gradiente no fine-tuned explica quase nada de h·g (0–15%) e deve ser lida como diagnóstico.

## 1. Camadas escolhidas e por quê

Probe linear por camada no `laya-multilingual` (etapa 3, `results/probe_typed/`), seleção no dev (180 casos, 900 decisões):

| camada | acurácia no dev | KL no dev |
|---|---|---|
| `enc_12` | 0,597 | 0,290 |
| `enc_17` | 0,597 | 0,288 |
| `enc_08` | 0,583 | 0,300 |

- `enc_17`: melhor no dev; empata com `enc_12` e vence no desempate pela menor KL.
- `enc_08`: o prior do dev (frequência do rótulo por (workflow, pergunta) em treino-menos-dev) acerta **0,510**; 25 de 26 camadas já o superam, `enc_01` por só 2,6 p.p.
  (dentro do ruído de 900 decisões). `enc_08` é o início do patamar (+7 p.p.).
- O prior aparece com três valores (0,461 do enunciado, sem origem identificada; 0,483 no teste; 0,510 no dev). A justificativa das camadas usa só o do dev.
- Em inglês (Parte C), a camada vem do probe do **base** no dev: `enc_17` (0,647), pela emenda 2.

## 2. Protocolo e desvios declarados

- **Dados:** descoberta = treino-menos-dev (1.020 casos, 5.100 decisões); dev = verificação; teste = só replicação. Nenhuma camada, limiar ou feature foi escolhida no teste.
- **Corpus do SAE (50M tokens):** 10% domínio (prompts de treino-menos-dev repetidos, ~3,2 épocas de 1,55M tokens), 45% inglês (FineWeb, `sample-10BT`), 45% multilíngue
  (FineWeb-2, 12 idiomas; o FineWeb-2 não tem inglês). Ativações geradas durante o treino (streaming), sem guardar ~150 GB.
- **SAE:** TopK (Gao et al.), k = 32, 16 × hidden latentes (12.288 no multilíngue; 16.384 em inglês), AuxK, decoder de norma unitária, seed 0.
- **Emenda 1 (antes de qualquer job): tokens massivos.** No `enc_17` os `[CLS]` têm norma ~19.000 (mediana 93) e ~4,8% dos tokens comuns também são "massivos"
  (norma > 5× a mediana). A FVE literal fica ~0,999 de graça. O SAE treina só em tokens não massivos; a rodada literal (todos os tokens) foi mantida como diagnóstico (seção 3).
- **Emenda 2 (antes da Parte C):** pareamento de features por vizinho mais próximo mútuo (cosseno ≥ 0,7) em vez de húngaro (16.384² não é prático); camada em inglês pelo dev.
- **Emenda 3 (depois de ver o `enc_08`):** o teste de Cochran-Mantel-Haenszel (CMH) estratifica pela identidade da opção a que a visão se refere. Foi uma escolha feita
  **depois** de ver contagens agregadas (sem selecionar feature alguma), então é exploratória; a primeira passada (estrato pergunta × opção prevista em todas as visões)
  está nas colunas `cmh_qp_*` e as tabelas mostram as duas.
- **Emendas 4 e 5 (ablação causal de features, §7)**, **6 (rotação das opções e segunda semente do SAE, §8 e §9)**, **7 (causa da sensibilidade à posição: sondas, remoção do subespaço de posição, features de posição, §10.1 a §10.5)** e **8 (camadas finais, aditividade, semente 1 e modelos em inglês, §10.6 a §10.8)**: cada uma escrita **antes** de rodar, com expectativas e critérios; os desvios, os palpites errados e o que não foi feito estão nos adendos do `PROTOCOLO.md`.

## 3. Qualidade dos SAEs

Dev (900 decisões) e 1M de tokens de texto genérico retido. "Substituir" = trocar a ativação da camada pela reconstrução; escala = trocar pela média dos marcadores.

| SAE | FVE dev (não massivos) | FVE só marcadores | FVE genérico | Δ acurácia, substituir todos (p.p.) | KL, todos | Δ acurácia, só marcadores | KL, só marcadores | sinalizadores |
|---|---|---|---|---|---|---|---|---|
| `ml_enc08_k32` | 0,954 | 0,777 | 0,830 | +0,2 | 0,0645 | +0,2 | 0,0005 | FVE marcadores 0,18 abaixo da geral |
| `ml_enc08_k32_dom20` (20% de domínio) | 0,963 | 0,861 | 0,828 | −1,0 | 0,0503 | −0,2 | 0,0003 | FVE marcadores 0,102 abaixo (limite 0,10) |
| `ml_enc17_k32` | 0,900 | 0,813 | 0,761 | −0,7 | 0,0128 | −0,2 | 0,0020 | nenhum |
| `ml_enc17_k32_literal` (todos os tokens) | 0,898 (0,9997 literal) | 0,807 | 0,755 | +0,7 | 0,0126 | +0,3 | 0,0014 | nenhum |
| `en_base_enc17_k32` (inglês) | 0,912 | 0,913 | 0,719 | −0,2 | 0,0059 | 0,0 | 0,0002 | 3,2% dos marcadores são massivos (> 1%) |
| `en_ft_enc17_k32_copy` (cópia no fine-tuned)* | 0,915 | — | 0,733 | −1,6 | 0,0019 | −0,2 | 0,0001 | 4,5% dos marcadores são massivos |

\* O dev é contaminado para o fine-tuned (acurácia ~85%); só vale a variação, não o nível.
Como escala, trocar os marcadores pela média dá KL 0,096 (`enc_08`), 0,188 (`enc_17`), 0,077 (base em inglês) e 0,016 (fine-tuned).
Latentes mortas (sem disparo nos últimos 5M tokens): 0% em `enc_08`, 0,02% em `enc_17`, 0% em inglês.

**Leituras que corrigem o que eu esperava:**
1. **A exclusão de tokens massivos quase não mudou a qualidade nos tokens comuns:** o SAE literal de `enc_17` (que treinou com os massivos) reconstrói os não massivos tão bem
   quanto o treinado sem eles (0,898 contra 0,900) e tem a mesma fidelidade (KL 0,0126 contra 0,0128). Eu tinha previsto que a MSE dominada pelos massivos atrapalharia o aprendizado
   dos tokens comuns; com esta largura os dados não mostram esse efeito. O que a emenda 1 corrigiu foi a **métrica**: 0,9997 de FVE literal é trivial.
2. **A FVE só dos marcadores usa a variância dentro do grupo** (todos são `[MASK]`), que é pequena; por isso é mais severa que as demais, e a fidelidade a jusante (KL 0,0005 em `enc_08`) não indica
   problema. A variante com 20% de domínio melhora a FVE dos marcadores (0,777 → 0,861) sem custo no texto genérico, mas o sinalizador ficou formalmente disparado por 0,002.
   O SAE original é o primário em todas as análises; a variante não foi usada para estatísticas de features.
3. Em `enc_08` nenhum token é massivo (FVE literal = FVE não massiva); em `enc_17` os marcadores multilíngues também não são (0 de 21.300).

## 4. Viés de prior do Laya (independe do SAE)

Fração de decisões em que o Laya escolhe a opção da hipótese, contra a do gabarito (`prior_bias_summary.json`):

| hipótese | descoberta (Laya / gabarito) | dev | teste |
|---|---|---|---|
| `choice`: classe majoritária da pergunta | 0,173 / 0,473 | 0,212 / 0,483 | 0,177 / 0,418 |
| `noul`: responde `true` | 0,757 / 0,510 | 0,769 / 0,434 | 0,772 / 0,508 |
| `score`: nível mais alto | 0,196 / 0,260 | 0,153 / 0,256 | 0,176 / 0,269 |

Distribuição das respostas na descoberta (previsto / gabarito): `choice` índice 0: 0,29/0,18; 1: 0,15/0,21; 2: 0,05/0,45; **3: 0,47/0,09**; `noul` `false` 0,24/0,49, `true` 0,76/0,51;
`score` nível 1: 0,44/0,26 e nível 3: 0,19/0,30. Das três hipóteses do pedido, a de severidade alta em `score` **não** se confirma (o Laya subestima); as outras duas, sim, nas três divisões.

## 5. Features no multilíngue (`enc_08` e `enc_17`)

**Três visões** por decisão: ativação no marcador da opção prevista (`pred`), no da correta (`corr`) e o máximo sobre os tokens do estado (`state`), em versão binária e ponderada; critério
"Laya errou" (`error`) e "Laya errou entre as decisões que o probe fora da dobra acerta" (`probe_right_laya_wrong`).

**O ranking ingênuo é dominado pela identidade da opção.** As 12 features de maior lift na visão `pred` do `enc_08` têm taxa de erro ~100% e várias compartilham a mesma confusão
(`action|3>1`, 111 decisões, lift ~21): são features no marcador da opção 3 quando o Laya a escolhe, e o Laya erra sempre que escolhe essa opção em `action`. Por isso o CMH:

| `enc_08`, descoberta | com suporte | ingênuo q≤0,05 | CMH (estrato da visão) mais erro / menos erro | CMH 1ª passada |
|---|---|---|---|---|
| `pred` / `error` | 543 | 364 | **9 / 11** | 9 / 11 |
| `corr` / `error` | 625 | 476 | 68 / 65 | 221 / 198 |
| `state` / `error` | 10.213 | 3.226 | 324 / 544 | 42 / 77 |
| `pred` / probe acerta | 468 | 348 | 43 / 43 | 43 / 43 |
| `corr` / probe acerta | 472 | 368 | 18 / 13 | 151 / 147 |
| `state` / probe acerta | 9.441 | 3.869 | 979 / 802 | 345 / 557 |

Em `enc_17` o padrão é o mesmo (`pred`/`error`: 334 → 11/16; `state`/`error`: 3.584 → 288/569; tabela completa em `REPORT_tables.md`). No `enc_08`, a visão `corr` mostrou na 1ª passada que o estrato estava errado para ela
(476 → 419 sobreviventes); com o estrato certo (a opção correta) sobram 133 (68 + 65).

**Replicação, sem depender da seleção** (todas as features com suporte na descoberta e no outro lado): Spearman de z entre descoberta e teste 0,42–0,76 em `enc_08` e 0,44–0,77 em `enc_17`
(o do lift ingênuo é 0,52–0,97, alto porque é dominado pela identidade da opção); concordância de sinal nas 20 features de maior |z|: 95–100% (80% em `corr`/probe acerta no teste do `enc_08`; 75% em `pred`/`error` no dev do `enc_17`).
A significância individual em dev e teste é baixa (poucas decisões ativas): as features mais bem ranqueadas no CMH muitas vezes têm só 24–41 ativações, com razões de chances enormes (49–99) e 3–10 ativações em dev e teste; só as
com suporte maior replicam de forma individual (por exemplo `enc_08` `corr`/`error` feature 10217: 1.970 ativas, z = 5,6 / 3,0 / 3,0 na descoberta / dev / teste).

**O que as features parecem ser (só descrição dos exemplos; interpretação em branco nos cartões):** nos cartões das 4 melhores por visão no CMH, os tokens que mais ativam são em geral lexicais e de baixo nível
(` river`, `_`, `country`, ` of`, ` over`, ` happened`, `sv`), com algumas ligadas ao nome de uma entidade; nenhuma é um "conceito" que os exemplos sustentem sozinhos. Há também duplicatas (as features 9977 e 11929 do `enc_08`
têm ativações quase idênticas: 2.097 e 2.098 na descoberta, 344 no dev e 807 no teste nas duas). Os cartões estão em `feature_cards/` (um critério por índice).

**Atribuição por gradiente** (decisões erradas: 3.888 no treino e 1.301 no teste; `Δ = logit[correta] − logit[prevista]`):

| | fração de h·g explicada pelas features, treino / teste | argmax em fp32 ≠ oficial |
|---|---|---|
| `enc_08` | **0,87 / 0,85** | 1,0% / 1,3% |
| `enc_17` | **0,31 / 0,26** | 1,0% / 1,3% |

Em `enc_08` a atribuição é útil: concentrada (as 10 maiores features somam 79% de |atribuição média| no marcador previsto e 69% no correto), e as 50 mais negativas na descoberta são negativas também no teste (100%; Spearman 0,69–0,71).
Em `enc_17` a maior parte do efeito linear fica no erro de reconstrução e no viés, então **a atribuição lá deve ser lida com cautela**. A atribuição é de primeira ordem e ignora o termo de erro do SAE.

**"Probe acerta, Laya erra":** 2.113 de 6.000 decisões de treino (predições fora da dobra) e 702 de 2.000 no teste. Os rankings nesse subconjunto (`feature_stats_*_probe_right_laya_wrong.parquet`) mostram o mesmo padrão:
muito do sinal ingênuo é identidade de opção; com CMH sobram 43 features (`pred`) e 979 (`state`) com mais erro no `enc_08`. Várias das features de `state` mais fortes têm lift ingênuo ≈ 1,0 mas OR do CMH 2–3 (a associação só aparece dentro de cada pergunta).

## 6. Parte C: base × fine-tuned (par em inglês, `enc_17`)

`convaiinnovations/laya` (base) × subfolder `typed-decisions` (fine-tuned), mesma revisão `e333cb7c…`; os pesos do fine-tuned avulso são idênticos (mesmo OID). Configuração comum 1024/256 em tudo.

- **Entradas idênticas:** os `token_id` e os papéis dos tokens são iguais nos dois modelos, nos dois splits (1.748.500 e 582.370 tokens).
- **A configuração comum não distorce o base:** acurácia no teste 0,3625 (512/192) contra 0,3630 (1024/256). Fine-tuned: **0,7685 no teste** (o README diz 0,766); no treino 0,854 e no dev 0,857,
  que são **contaminados** (ele foi treinado nos workflows do benchmark; NÃO VERIFICADO em qual split, mas os números mostram que viu o treino). Toda acurácia e associação com acerto do fine-tuned vem do teste.
- **Probe no base, dev:** melhor camada do backbone `enc_17` (0,647); no teste 0,621, contra prior 0,483, controle 0,463 e o próprio base 0,363.
- **SAE do base no fine-tuned:** FVE 0,774 (0,652 nos marcadores) contra 0,912 no base: queda de 0,138 (limite 0,05). Pela emenda 2 treinei uma **cópia** do SAE nas ativações do fine-tuned (partida a quente, 10M tokens): FVE no fine-tuned **0,915**.
- **Marcadores massivos (fora do SAE) e a resposta do modelo** (`massive_markers_*.json`). Em `enc_17` os marcadores massivos ocorrem **só na segunda opção (`true`) das perguntas `noul`**
  (P(massivo) = 0 em `choice` e `score`, e em todas as posições exceto a 1). Uma primeira versão desta análise comparava com todas as opções e inflava o efeito; o cálculo correto condiciona a `noul` e à posição:

| `noul`, treino / teste | P(marcador `true` massivo) | P(responde `true`) (gabarito) | P(responde `true` \| `true` massivo) | P(responde `true` \| `true` não massivo) |
|---|---|---|---|---|
| base | 33% / 33% | 0,659 / 0,673 (0,498 / 0,508) | **0,903 / 0,915** | 0,537 / 0,552 |
| fine-tuned | 50% / 50% | 0,518 / 0,515 (0,498 / 0,508) | 0,469 / 0,467 | 0,567 / 0,563 |

  No base, quando o marcador `true` é massivo, o modelo responde `true` 36 pontos mais vezes do que quando não é (e o base já tem viés para `true`, seção 4). No fine-tuned a relação desaparece (até inverte um pouco).
  É uma associação dentro de um único tipo de pergunta, não causalidade, e é invisível às features (esses marcadores ficam fora do SAE). No multilíngue não há marcador massivo.
- **Diferenças por feature, 1ª passada** (SAE do base nos dois lados; contraste = ativação no marcador da opção correta − média nas erradas; bootstrap pareado por caso, 1.000 reamostras):
  876 features têm IC95% fora de zero na descoberta (469 no dev, 649 no teste); as 50 maiores |Δ| têm o mesmo sinal no teste (100%) e o Spearman de Δ entre descoberta e teste é 0,55. As maiores mudanças: feature 2786 (contraste 0,30 → 3,14; Δ = 2,83, IC 2,71–2,96; teste 2,56), 11776 (Δ = −1,31), 7558 (Δ = +1,27). **Ressalva:** com FVE de só 0,774 no fine-tuned, parte dessas mudanças pode ser leitura imperfeita do SAE; a 2ª passada (cópia do SAE e pareamento) separa isso.
- **Atribuição por gradiente no fine-tuned:** fração de h·g explicada 0,12 / 0,08 (SAE do base) e 0,15 / −0,004 (cópia), treino / teste; no base em inglês é negativa (−0,25 / −0,41). Trato como **diagnóstico não confiável**, não como resultado (a razão é de somas com sinal, então pode ficar negativa quando o efeito das features se opõe ao total h·g; não indica erro de cálculo, e sim que o SAE não captura a direção relevante do gradiente).
- **2ª passada (cópia do SAE no fine-tuned, pareamento por vizinho mútuo; job 3647, `part_c_sae_ft/`):** 14.893 das 16.384 features (91%) ficaram pareadas (cosseno ≥ 0,7; mediana do cosseno do vizinho mais próximo 0,94);
  a FVE da cópia no fine-tuned é 0,915 (0,922 só nos marcadores). Com o lado do fine-tuned lido pela cópia, 915 features pareadas têm IC95% do Δ fora de zero na descoberta (454 no dev, 646 no teste);
  as 50 maiores |Δ| têm o mesmo sinal no teste (100%) e o Spearman de Δ entre descoberta e teste é 0,57. **As mesmas features lideram nas duas passadas** (2786: contraste 0,30 → 3,22, Δ = 2,91, IC 2,79–3,04, teste 2,63;
  15421: Δ = −1,36; 12238: −1,30; 7558: +1,25; 11776: −1,24; 6496: −1,00; 8737: −0,88; 9304: +0,88), então as maiores mudanças **não são artefato** de o SAE do base ler mal o fine-tuned.
  O que isso mede: a ativação dessas features no marcador da opção correta, comparada com a média nas opções erradas, muda com o fine-tuning; não diz o que elas representam. Cartões das 30 maiores: `feature_cards/en_changed_base_vs_ft/`
  (exemplos e estatísticas lidos no **base**; o campo "Interpretação" está em branco). As estatísticas de erro dessas features no fine-tuned só valem no teste (contaminação).
  Associação entre mudança de contraste e acerto não foi analisada por feature além do que está em `feature_stats_base_vs_ft.parquet` (colunas `lift_error_*`, `q_error_*`).
  **O que os cartões das 30 maiores mostram (descrição dos exemplos, sem rótulo):** em 14 delas a maioria dos exemplos de maior ativação está no próprio `[MASK]` de uma opção (70 dos 150 exemplos máximos);
  nas demais, nos tokens da pergunta, das opções ou do estado. Entre as 8 de maior |Δ|, as de Δ positivo (2786, 7558, 9304) têm cartões cujos exemplos máximos estão no marcador de uma opção, e as de Δ negativo
  incluem 15421 e 12238 (também no marcador) e 11776, 6496 e 8737 (na pergunta, no estado e nas opções). Δ negativo quer dizer que a diferença (opção correta − erradas) fica menor, não que a feature "represente o erro".
  Em 12238 e 15436 o base errou em todos os 5 exemplos máximos mostrados, o que é só uma observação dos exemplos, não uma estatística.

## 7. Ablação causal (emendas 4 e 5 do `PROTOCOLO.md`)

**O que foi feito.** Zerar um conjunto de features nas ativações da camada (`SAEIntervention`, modo `edit`: o erro de reconstrução do SAE fica no fluxo, só a contribuição das features some) e medir a mudança nos logits do Laya.
Margem m = logit[correta] − logit[escolha sem intervenção]; **Δm > 0 aproxima da resposta correta**. Duas variantes, cada uma com a sua lista escolhida **só na descoberta**: `pred` (zera só no `[MASK]` da opção que o Laya escolhe;
lista por `a_pred`; **primária**) e `all` (zera em todos os tokens não massivos; lista por atribuição total; controle de escala). Conjuntos das K primeiras de cada lista, K ∈ {1, 5, 20, 100}; **K = 20 é o desfecho principal**.
Nulo: 50 conjuntos aleatórios por K e variante, pareados por frequência de ativação. Dev (900 decisões, 606 erros) para a análise e teste (2.000 decisões, 1.300 erros) só para replicar as mesmas listas.
Jobs 3874 (`enc_08`) e 3875 (`enc_17`); artefatos em `ablation/<SAE>/` (`selected_features.json`, e `dev/` e `test/` com `conditions.csv`, `summary.json`, `calibration.csv`).

**Dois desvios de rumo, declarados antes de qualquer dev ou teste** (PROTOCOLO, emenda 5 e adendo): um teste de encanamento em casos de **descoberta** mostrou que o desenho original (zerar em *todos* os marcadores, lista por `a_pred`) dava o controle positivo com sinal trocado,
porque zerava também o marcador da opção correta; as variantes foram refeitas. Esse teste também mostrou o efeito pequeno e a quebra de decisões certas, e por isso o critério de "relevante" passou a exigir que a acurácia total não caia.
O limiar de relevância (≥ 10% da margem média das decisões erradas, ~2,1 logits, ou ≥ 1% dos erros corrigidos) foi escrito **depois** de eu ver que o efeito de K = 20 na descoberta era de ~1% da margem; não é neutro.

**Resultados, K = 20 e K = 100 (dev; teste entre parênteses).** Δacc é a variação de acurácia em todas as decisões (acurácia sem intervenção: 0,327 no dev, 0,350 no teste).

| camada | variante | K | Δm (logit) | Δm / \|m\| | erros corrigidos | certas quebradas | Δacc (p.p.) | nulo p95 |
|---|---|---|---|---|---|---|---|---|
| `enc_08` | `pred` | 20 | +0,040 (+0,033) | 1,9% (1,6%) | 0,8% (0,9%) | 2,4% (1,3%) | −0,2 (+0,2) | +0,002 (+0,000) |
| `enc_08` | `pred` | 100 | +0,051 (+0,045) | 2,5% (2,1%) | 1,5% (1,0%) | 3,4% (2,6%) | −0,1 (−0,3) | −0,000 (−0,000) |
| `enc_08` | `all` | 20 | +0,563 (+0,536) | 27% (26%) | 11,7% (10,5%) | 19,0% (18,7%) | +1,7 (+0,3) | +0,014 (+0,013) |
| `enc_08` | `all` | 100 | +1,340 (+1,335) | 64% (64%) | 27,4% (26,0%) | 38,4% (36,3%) | +5,9 (+4,2) | +0,038 (+0,051) |
| `enc_17` | `pred` | 20 | +0,421 (+0,433) | 20% (21%) | 7,6% (6,9%) | 17,7% (16,7%) | −0,7 (−1,4) | +0,003 (+0,005) |
| `enc_17` | `pred` | 100 | +0,564 (+0,577) | 27% (28%) | 10,4% (9,6%) | 25,2% (23,4%) | −1,2 (−2,0) | +0,010 (+0,010) |
| `enc_17` | `all` | 20 | +0,874 (+0,958) | 42% (46%) | 15,4% (15,5%) | 29,6% (30,9%) | +0,7 (−0,8) | +0,007 (+0,007) |
| `enc_17` | `all` | 100 | +1,330 (+1,377) | 64% (66%) | 21,0% (22,2%) | 38,1% (37,7%) | +1,7 (+1,3) | +0,017 (+0,017) |

Todos os IC95% por bootstrap sobre casos excluem zero e todos os p empíricos são 0,0196 (o mínimo possível com 50 sorteios: nenhum conjunto aleatório chegou perto). O **controle positivo** (lista pos, K = 20, variante `pred`) dá Δm negativo e abaixo do percentil 5 do nulo:
−0,55 (−0,53) em `enc_08` e −0,91 (−0,85) em `enc_17`.

**Calibração da primeira ordem (dev, 25 features individuais por lista; só no dev, porque o teste não roda condições individuais).** Variante `pred`: Spearman 0,93 (`enc_08`) e 0,90 (`enc_17`), 98% e 96% de concordância de sinal, e |medido| / |previsto| = 1,16 nas duas.
Variante `all`: Spearman 0,90 e 0,80, concordância de sinal 90% e 96%, razão 0,81 e 1,92 (em `enc_17` a ablação em todos os tokens amplifica o efeito previsto por quase 2×, uma não linearidade). A atribuição por gradiente prevê bem o efeito de **features individuais** no marcador, mesmo em `enc_17`, onde explicava só ~30% do total de h·g.

**Leitura contra o critério fixado.**
- **Distinguível e replicado: sim**, nas duas camadas e nas duas variantes (critério da emenda 4: IC95% fora de zero e acima do percentil 95 do nulo no dev, mesmo sinal e acima do percentil 95 no teste; controle positivo ok).
- **Relevante (variante `pred`, a que responde à pergunta): não.** Em `enc_08` o efeito de K = 20 é ~2% da margem e corrige 0,8% dos erros: **distinguível mas desprezível**, como eu esperava. Em `enc_17` o efeito é grande (~20% da margem, 7,6% dos erros corrigidos), mas
  a acurácia total **cai** nas duas divisões (−0,7 e −1,4 p.p.): a intervenção quebra 17% das decisões certas, aproximadamente tantas quanto conserta (cerca de 46 erros corrigidos e 52 acertos quebrados no dev). Pelo adendo da emenda 5: "efeito grande, mas não específico".
- **Variante `all`** (controle de escala, não evidência de mecanismo, pelo protocolo): o efeito é grande e a acurácia total sobe em quase todos os casos (+1,7 e +5,9 p.p. em `enc_08` no dev; em `enc_17` K = 20 sobe no dev e cai no teste), mas à custa de quebrar 19–38% das decisões certas.
  Pela letra do critério isto contaria como "relevante" em `enc_08` e em `enc_17` com K = 100; não leio como mecanismo isolado, e sim como uma perturbação ampla do modelo que, dado o viés de prior (seção 4), recupera parte dos erros.

**O que isto sustenta e o que não sustenta.**
- Sustenta: as features de maior atribuição negativa no marcador da opção escolhida têm efeito causal real e replicável, **na direção prevista pela atribuição**, sobre a margem entre a resposta correta e a escolhida. Zerar features sem relação (nulo pareado) não faz nada.
- Não sustenta que existam "features do erro". Em `enc_08` o efeito de qualquer conjunto de até 100 features é pequeno, o que é coerente com um viés de prior **distribuído** por muitas features, cada uma com atribuição média de ordem 0,001–0,013 logit. Em `enc_17` o efeito é maior,
  mas **não específico dos erros**: uma leitura consistente com os números (não verificada) é que essas features codificam "a opção que está sendo escolhida" em geral, e zerá-las reduz a confiança na escolhida tanto quando ela está errada quanto quando está certa.
- Não sustenta um rótulo para nenhuma feature: nenhuma feature foi interpretada.

**Limitações específicas.** (1) Só zeragem de features já ranqueadas pela atribuição de primeira ordem; não testei fixar valores, nem features escolhidas por outros critérios (por exemplo pelo CMH), nem o k = 64. (2) O nulo foi pareado por frequência de ativação e, em `enc_17`, 21% dos sorteios caíram em faixa vizinha
(4% em `enc_08`), porque as features de maior atribuição são muito frequentes; os nulos ficam em ≤ 0,017 logit, bem abaixo dos efeitos, então a conclusão não depende disso. (3) Os p empíricos têm piso de 0,0196. (4) Features duplicadas (ativações quase idênticas) podem contar duas vezes numa lista.
(5) A variante `pred` zera no marcador da opção escolhida *sem intervenção*; não há variante que zere o marcador da opção correta (a atribuição `a_corr` ficou sem teste causal). (6) Um só SAE por camada (seed 0, k = 32).

## 8. Rotação de opções: posição ou conteúdo (emenda 6, parte A)

**O que foi feito.** Cada decisão com k opções foi mostrada ao Laya nas k rotações cíclicas da ordem das opções (`option_order` do pacote); o gabarito não muda. Como cada opção passa uma vez por cada posição, a média sobre rotações separa a preferência por **posição** da preferência por **conteúdo**.
Treino (1.200 casos, 21.300 linhas decisão × rotação) e teste (400 casos, 7.100 linhas); jobs 4173 (extração) e 4179 (análise). A rotação 0 reproduz exatamente a extração da etapa 3 (diferença 0 nas probabilidades e nos vetores). Os números abaixo são do teste; o treino repete o padrão (`rotation/behaviour.json`).

**A1, comportamento.** Desvio da fração de escolhas em cada posição em relação a 1/k (IC95% por bootstrap sobre casos), teste:

| tipo (n decisões) | pos. 0 | pos. 1 | pos. 2 | pos. 3 | pos. 4 |
|---|---|---|---|---|---|
| `choice`, 4 opções (400) | +1,8 p.p. | −0,3 | **−3,2** [−4,5; −2,0] | +1,8 | |
| `choice`, 5 opções (200) | +2,1 | −0,5 | −0,1 | −0,4 | −1,1 |
| `noul`, 2 opções (600) | −2,1 | +2,1 | | | |
| `score`, 4 opções (700) | **−18,2** [−19,1; −17,3] | +4,6 | +2,8 | **+10,9** [+9,6; +12,0] | |
| `score`, 5 opções (100) | −17,6 | +11,0 | +3,2 | −0,8 | +4,2 |

- **Critério da emenda 6 (viés posicional em `choice`: alguma posição a ≥ 5 p.p. de 1/k com IC fora de zero): NÃO atendido.** O maior desvio é 3,2 p.p. O viés de índice que a seção 4 mostrava em `choice` (47% das escolhas no índice 3 contra 9% do gabarito) **não é posicional**: com a posição balanceada, a preferência por posição some. É preferência pelo **conteúdo** da opção que, na ordem original, ocupa o índice 3.
- **`score` tem um viés posicional grande**: o Laya quase nunca escolhe a primeira opção listada (6,8% contra 25% esperados) e favorece a última. O critério da emenda 6 era para `choice`; este achado é descritivo, mas é o maior efeito posicional medido. `noul` tem ±2 p.p. (abaixo de 5).
- **Troca de escolha com a rotação** (fração das decisões em que a opção escolhida muda): `choice` 44% (4 opções) e 30% (5 opções); `score` 82% e 91%; `noul` 6%. Nenhuma decisão de `choice` ou `score` escolhe a mesma *posição* em todas as rotações; em `choice` 56% e 70% escolhem sempre a mesma *opção* (conteúdo).
- **Acurácia**: média sobre as rotações (cada decisão conta uma vez) e a da rotação 0 (a ordem original), contra o acaso esperado:

| tipo | rotação 0 | média das rotações [IC95%] | acaso esperado |
|---|---|---|---|
| `choice` | 0,295 | 0,260 [0,234; 0,289] | 0,233 |
| `noul` | 0,493 | 0,487 [0,443; 0,529] | 0,500 |
| `score` | 0,282 | 0,276 [0,255; 0,297] | 0,244 |
| todas | 0,349 | **0,334 [0,317; 0,353]** | 0,318 |

  **Com a ordem das opções balanceada, a acurácia do Laya multilingual (0,334) fica indistinguível do acaso (0,318 está dentro do IC95%)**; a margem de 3 p.p. sobre o acaso que ele tinha na ordem original (0,349) em parte depende dessa ordem. Não testei a diferença pareada entre a rotação 0 e as demais.

**A2, o probe lê conteúdo ou posição.** O probe da etapa 3, treinado **só na ordem original**, aplicado aos marcadores rotacionados do teste, em decisões com k ≥ 4 (rotações 0 a 3; acaso ≈ 0,24):

| camada | acurácia por rotação (0, 1, 2, 3) | amplitude (melhor − pior) |
|---|---|---|
| `enc_08` | 0,500 · 0,413 · 0,338 · **0,274** | **0,226** |
| `enc_17` | 0,519 · 0,455 · 0,424 · **0,423** | 0,096 |
| Laya | 0,288 · 0,295 · 0,249 · 0,244 | 0,051 |

- **A expectativa da emenda ("o probe é mais estável que o Laya") foi desmentida:** a amplitude do probe é maior que a do Laya nas duas camadas (Laya − probe: −0,175, IC95% [−0,213; −0,127] em `enc_08`; −0,046, IC95% [−0,082; −0,012] em `enc_17`).
- **`enc_08`**: na rotação 3 o probe cai para 0,274, perto do acaso. A vantagem dele nessa camada depende da posição/ordem das opções, não só do conteúdo.
- **`enc_17`**: cai de 0,52 para 0,42, mas **continua bem acima do acaso (~0,24) e do Laya (0,24 a 0,29)** nas rotações 1 a 3: o conteúdo da opção é legível nessa camada, ainda que parte do ganho também venha da posição.
- Em `noul` (2 opções): probe `enc_17` 0,70 → 0,59 e `enc_08` 0,68 → 0,66; Laya 0,49 → 0,48.
- **Consequência para a etapa 3:** o 0,574 do probe em `enc_17` vem de um probe treinado e testado na ordem original; esta análise mostra que, por conteúdo apenas, o ganho é menor (~0,42 nas rotações 1 a 3 em k ≥ 4, contra 0,52 na ordem original, nas mesmas decisões). Um probe treinado nas rotações foi feito depois (seção 10.1): chega a 0,559 no teste, indistinguível do 0,574; o probe antigo se apoiava na posição, mas a informação de conteúdo está na representação.

**A3, posição e conteúdo por camada** (η² de duas vias do vetor do marcador, por pergunta, quadrado latino; teste; resíduo = variação dentro da mesma posição e opção, isto é, o resto do caso):

| camada | η² posição | η² conteúdo | resíduo |
|---|---|---|---|
| `enc_01` | 0,962 | 0,005 | 0,033 |
| `enc_04` | 0,913 | 0,046 | 0,041 |
| `enc_08` | 0,132 | 0,775 | 0,093 |
| `enc_12` | 0,153 | 0,717 | 0,130 |
| `enc_17` | 0,088 | 0,531 | 0,382 |
| `enc_22` | 0,034 | 0,438 | 0,529 |
| `head_2` | 0,056 | 0,416 | 0,527 |

A posição domina até a camada 4 e o conteúdo toma o lugar entre as camadas 4 e 6 (figura `rotation/eta_by_layer.png`); depois o conteúdo decai e o resíduo cresce com a profundidade. Coerente com a expectativa (descritiva, sem limiar). `emb` fica fora (todos os marcadores são o mesmo vetor).

**A4, posição e conteúdo nas features da ablação** (η² por feature nos marcadores; lista neg da variante `pred`, K = 20, contra 100 conjuntos aleatórios pareados por frequência; as listas são as da emenda 5):

| camada | η² posição da lista (nulo p95) | η² conteúdo da lista (nulo p95) | leitura pelo critério |
|---|---|---|---|
| `enc_08` | 0,102 (0,140) | 0,684 (0,631) | "mais de conteúdo"; não "mais posicional" |
| `enc_17` | 0,125 (0,061) | 0,545 (0,440) | mais de conteúdo **e** mais posicional que o acaso |

  Teste e treino dão os mesmos números a três casas. Entre todas as features com suporte, a mediana do η² é 0,033 (posição) e 0,334 (conteúdo) em `enc_08`, e 0,018 e 0,125 em `enc_17`; só 12% e 7% têm mais posição que conteúdo. **As features das listas da ablação codificam sobretudo o conteúdo da opção; em `enc_17` a lista tem também mais posição que o acaso.**

**O que isto sustenta.** Para o trabalho sobre viés posicional: (i) o viés de prior do Laya em `choice` é de conteúdo, não de posição (o critério pré-fixado não foi atendido); (ii) em `score` há viés posicional grande; (iii) a posição é codificada nas camadas iniciais (1 a 4) e o conteúdo a partir da 5; (iv) as features ligadas ao erro na ablação são em maioria de conteúdo.
**O que não sustenta:** que o Laya "acerta mais do que o acaso" quando a ordem é balanceada (o IC inclui o acaso); nada sobre o fine-tuned (não foi rotacionado).

**Limites e declarações.** Rotações cíclicas não esgotam as permutações (k! para k ≥ 3). O A2 foi restrito a k ≥ 4 e rotações 0 a 3 (para comparar rotações com o mesmo conjunto de decisões): escolha de código feita antes de rodar no cluster e não escrita no protocolo; o adendo da emenda 6 a registra. A figura de preferência por posição troca o painel de `choice` com 5 opções pelo de `score` com 4, depois de eu ver que o efeito grande está em `score`.
`score` mostra "level i" no texto; permutar a ordem mantém os rótulos, mas é um tipo de pergunta diferente dos outros.

## 9. Estabilidade entre sementes do SAE (emenda 6, parte B)

SAEs de semente 1 (mesmos hiperparâmetros, orçamento e regra de massivos; jobs 4171/4172, qualidade 4174/4176, features 4175/4177, ablação 4180/4181, pareamento 4182).

- **Qualidade equivalente (confirmada):** FVE no dev 0,954 contra 0,954 (`enc_08`) e 0,899 contra 0,899 (`enc_17`); KL da substituição total 0,064 contra 0,065 e 0,014 contra 0,013; o sinalizador de FVE dos marcadores em `enc_08` repete (0,791).
- **Pareamento de features (expectativa > 80%): DESMENTIDA.** Por vizinho mútuo do decoder (cosseno ≥ 0,7), só **48,6% (`enc_08`) e 52,5% (`enc_17`)** das features da semente 0 têm correspondente na semente 1 (cosseno mediano do vizinho mais próximo 0,70 e 0,74).
  Parte disso pode ser *feature splitting* (uma feature de uma semente dividida em várias na outra), que o pareamento um-para-um não capta; não verifiquei.
- **Estabilidade da lista (critério: ≥ 50%): NÃO atendida.** Das 20 features da lista neg da semente 0, só 9 (`enc_08`) e 8 (`enc_17`) têm correspondente; destas, 3 e 7 estão entre as 100 de maior atribuição negativa da semente 1: **15% e 35% das 20.** Os índices não coincidem em nenhum caso.
- **A conclusão da ablação é robusta à semente (critério atendido)**, mesmo com as listas diferentes. Variante `pred`, K = 20, dev (teste entre parênteses):

| camada | semente | Δm (logit) | % da margem | erros corrigidos | certas quebradas | Δacc (p.p.) | controle pos |
|---|---|---|---|---|---|---|---|
| `enc_08` | 0 | +0,040 (+0,033) | 1,9% (1,6%) | 0,8% (0,9%) | 2,4% (1,3%) | −0,2 (+0,2) | −0,55 |
| `enc_08` | 1 | +0,042 (+0,038) | 2,0% (1,8%) | 1,0% (1,0%) | 2,7% (2,0%) | −0,2 (−0,1) | −0,53 |
| `enc_17` | 0 | +0,421 (+0,433) | 20% (21%) | 7,6% (6,8%) | 17,7% (16,7%) | −0,7 (−1,4) | −0,91 |
| `enc_17` | 1 | +0,597 (+0,602) | 29% (29%) | 12,4% (11,5%) | 30,6% (24,4%) | −1,7 (−1,1) | −0,74 |

  Mesmo sinal, acima do percentil 95 do nulo e IC fora de zero nas duas sementes; mesma leitura (desprezível em `enc_08`; grande mas não específico em `enc_17`, com a acurácia total caindo). O tamanho do efeito em `enc_17` varia com a semente (20% contra 29% da margem).
- **O que isto implica:** o resultado **agregado** da ablação (o efeito de um conjunto de features no marcador escolhido) é reprodutível; o **ranking de features individuais** (listas, cartões, "a feature X") **depende da semente** e deve ser descrito como candidato daquela semente, nunca como "a feature do erro".

## 10. O que causa a sensibilidade à posição (emenda 7)

**Pergunta e medida.** "O que acontece por dentro que causa sensibilidade à posição?" Sensibilidade = **taxa de troca** (fração das decisões cuja escolha não é a mesma nas k rotações cíclicas das opções). Foco em `score`, escolhido **depois** de ver o resultado da seção 8 (declarado no `PROTOCOLO.md`); `choice` e `noul` saem ao lado nos mesmos arquivos (`conditions.csv`). Três experimentos, com critérios e expectativas fixados antes de rodar.
Jobs: sondas 4581; subespaço `single` 4582 e `onward` 4583; features `enc_08` 4584 e `enc_17` 4585 (versões curtas 4519 a 4523 antes). Paridade do argmax em lote com o `predict` oficial rotacionado: 0,9975 (dev) e 0,9976 (teste). Artefatos: `position_probes/`, `position_causal/{subspace_single,subspace_onward,features_ml_enc08_k32,features_ml_enc17_k32}/`.

### 10.1 E1, o que cada camada guarda (sondas sobre os vetores de rotação; teste)
| camada | posição legível (`score`; todas) | identidade da opção legível | sonda de resposta treinada nas rotações (`score`; todas) |
|---|---|---|---|
| 1 | 1,00; 1,00 | 1,00; 1,00 | 0,36; 0,45 |
| 5 | 1,00; 1,00 | 1,00; 1,00 | 0,42; 0,51 |
| 8 | 1,00; 1,00 | 1,00; 1,00 | 0,43; 0,53 |
| 12 | 1,00; 1,00 | 1,00; 1,00 | 0,46; 0,54 |
| 17 | 1,00; 1,00 | 1,00; 1,00 | **0,46; 0,56** |
| 22 | 0,98; 0,96 | 1,00; 1,00 | 0,44; 0,55 |
| `final_norm` | 0,98; 0,96 | 1,00; 1,00 | 0,44; 0,55 |
| `h2` (cabeça) | 0,64; 0,55 | 0,95; 0,95 | 0,36; 0,49 |

Acaso: 0,24 (`score`) e 0,26 (todas) para os dois classificadores; para a sonda de resposta, 0,24 e 0,32; o Laya com a ordem balanceada acerta 0,276 (`score`) e 0,334 (todas). A camada da sonda de resposta foi escolhida no dev (`enc_17`): teste **0,559 [0,539; 0,579]** (todas) e **0,460 [0,433; 0,487]** (`score`). Figura: `position_probes/probes_by_layer.png`.
- **A posição da opção é legível por um classificador linear em todas as camadas do encoder**, quase perfeitamente, e só cai na cabeça de decisão (`h2`: 0,64). Isso convive com o η² de posição baixo nas camadas tardias (seção 8): pouca variância não é pouca informação.
- **A identidade da opção também é legível desde a camada 1** (1,00): dentro de um grupo de pergunta cada opção tem texto diferente. A expectativa da emenda ("a partir da camada 5") estava errada; o que muda na camada 5 é o que domina a *variância*, não o que é *legível*.
- **Há conteúdo da resposta legível sem a posição:** critério da emenda (acurácia no teste na camada do dev acima do acaso balanceado 0,318 por ≥ 5 p.p., com IC95% acima de 0,318) **atendido**. 0,559 é indistinguível do 0,574 do probe da etapa 3 (IC95% inclui 0,574): **corrige a ressalva da seção 8**; o probe antigo se apoiava na posição (cai nas rotações), mas a representação carrega conteúdo suficiente para chegar ao mesmo número quando a posição é descorrelacionada da resposta. A cabeça de decisão perde parte disso (0,49 em `h2`).

### 10.2 E2, remover o subespaço de posição dos marcadores (`score`)
Direções entre-classe estimadas na descoberta (dimensões do subespaço de posição: 2 a 3 nas camadas 1 a 4, 5 na camada 8, 10 na 14, 9 na 17, 7 na 22; o de conteúdo, controle, bate no teto de 16 na maioria das camadas, isto é, remove *mais* dimensões que o de posição). 20 subespaços aleatórios de mesma dimensão como nulo por célula. Taxa de troca de base: 0,858 (dev) e 0,829 (teste). Δ em pontos percentuais, controle = remover o subespaço de conteúdo.

**Só na camada L (`single`):**

| camada | Δ troca dev (IC95%) | Δ acc dev | controle de conteúdo dev | qualifica no dev | Δ troca teste (IC95%) | Δ acc teste | controle teste | qualifica no teste |
|---|---|---|---|---|---|---|---|---|
| 1 | +7,5 [+3,3; +11,7] | −3,1 | +5,6 | não | — | — | — | — |
| 2 | +7,2 [+4,2; +10,3] | −0,8 | +11,1 | não | — | — | — | — |
| 3 | +2,8 [−0,3; +5,8] | −0,3 | +0,8 | não | — | — | — | — |
| 4 | +0,6 [−1,7; +2,8] | +0,2 | +5,8 | não | — | — | — | — |
| 5 | +6,9 [+3,9; +10,0] | −0,5 | +6,4 | não | — | — | — | — |
| 6 | +3,1 [+0,6; +5,8] | −0,3 | +1,4 | não | — | — | — | — |
| 8 | −4,4 [−6,7; −2,2] | +0,2 | −0,8 | **sim** | −3,2 [−4,9; −1,8] | −0,4 | +0,1 | **sim** |
| 10 | +3,3 [+1,1; +5,8] | −0,1 | −1,4 | não | — | — | — | — |
| 12 | −8,9 [−12,8; −5,3] | +0,3 | −4,7 | não | — | — | — | — |
| 14 | −18,3 [−22,5; −14,4] | +0,0 | −5,6 | **sim** | −17,9 [−20,6; −15,6] | −1,5 | −3,6 | **sim** |
| 17 | −11,7 [−15,6; −7,8] | −1,7 | +2,5 | **sim** | −13,2 [−16,0; −10,2] | −3,2 | +1,1 | não |
| 20 | +3,9 [+0,0; +7,8] | −4,2 | +9,7 | não | — | — | — | — |
| 22 | −14,2 [−19,2; −8,9] | +2,7 | +0,0 | **sim** | −16,4 [−19,8; −13,1] | +1,6 | −0,5 | **sim** |

**Da camada L até a 22 (`onward`):** só a camada 22 qualifica (é a mesma intervenção que `single` na 22; números idênticos, o que confere o determinismo). Dev completo:

| camada | Δ troca dev (IC95%) | Δ acc dev | controle de conteúdo dev | qualifica no dev | Δ troca teste (IC95%) | Δ acc teste | controle teste | qualifica no teste |
|---|---|---|---|---|---|---|---|---|
| 1 | +9,4 [+5,5; +13,1] | −1,3 | +13,3 | não | — | — | — | — |
| 2 | +3,6 [−1,1; +8,3] | −3,9 | +9,2 | não | — | — | — | — |
| 3 | +5,3 [+0,5; +9,7] | −3,4 | +4,2 | não | — | — | — | — |
| 4 | +3,1 [−2,5; +7,5] | −3,6 | +2,2 | não | — | — | — | — |
| 5 | +5,3 [+0,8; +9,7] | −5,1 | +2,5 | não | — | — | — | — |
| 6 | +6,4 [+1,7; +10,8] | −3,3 | +5,6 | não | — | — | — | — |
| 8 | +2,5 [−2,2; +6,9] | −3,8 | +2,2 | não | — | — | — | — |
| 10 | +2,2 [−2,8; +6,4] | −2,6 | +2,8 | não | — | — | — | — |
| 12 | −2,8 [−8,3; +2,5] | −4,1 | +1,1 | não | — | — | — | — |
| 14 | −2,2 [−7,5; +2,8] | −3,7 | −3,9 | não | — | — | — | — |
| 17 | −0,3 [−5,0; +4,4] | −1,4 | +5,3 | não | — | — | — | — |
| 20 | −6,1 [−11,1; −1,4] | −4,6 | −1,4 | não | — | — | — | — |
| 22 | −14,2 [−19,2; −8,9] | +2,7 | +0,0 | **sim** | −16,4 [−19,8; −13,1] | +1,6 | −0,5 | **sim** |

- **Qualificam e replicam no teste: 8 (efeito pequeno), 14 e 22.** A 17 reduz a troca em 13 p.p. no teste, mas **a acurácia cai 3,2 p.p. e não qualifica**. A 12 quase qualificou no dev (−8,9 p.p.), mas o controle de conteúdo também mexeu (−4,7) e a especificidade a barrou.
- **A camada 22 é a mais limpa**: a troca cai de 0,829 para 0,665 no teste (−16,4 p.p., cerca de **20% da sensibilidade**), a acurácia sobe 1,6 p.p. (IC95% [−0,3; +3,5]), o controle de conteúdo não mexe (−0,5) e `choice` quase não muda. O desvio da primeira opção vai de −18,1 p.p. para +3,9 p.p. e o da última de +10,9 para −4,1: **o sinal de posição no marcador da camada 22 carrega o viés posicional do `score`**. A 14 tem o maior efeito (−17,9 p.p., 22% da sensibilidade), um pouco menos específico (controle −3,6).
- **Nas camadas 1 a 6 e 10 a 20 sem efeito útil:** remover a posição ali *aumenta* a troca (+2 a +8 p.p.) e perturba o modelo (acurácia −4,2 p.p. na 20).
- **O `onward` não soma:** remover da 14 até a 22 dá −2,2 p.p. (IC95% inclui zero), embora cada uma das camadas 14 e 22 sozinha dê −14 a −18. **Não sei explicar** (hipóteses, como o modelo compensar ou desorganizar-se com a remoção em várias camadas, NÃO VERIFICADAS).
- **Expectativa da emenda DESMENTIDA:** eu esperava efeito pequeno em `single` e nenhuma ou poucas células qualificadas.

### 10.3 E3, zerar as features do SAE de maior η² de posição (semente 0; `score`)
Lista escolhida só na descoberta (η² de posição em `enc_08`: 0,92, 0,85, 0,84, 0,75, 0,74 as 5 maiores, 0,17 a 100ª; em `enc_17`: 0,89, 0,79, 0,65, 0,62, 0,53 e 0,10), zeradas em todos os marcadores; controle = as K de maior η² de conteúdo; nulo = 20 conjuntos pareados por frequência. Avaliação primária no teste; dev como segunda conferência independente.

**`enc_08`** (1.032 features com suporte):

| K | Δ troca teste (IC95%) | Δ acc teste | controle de conteúdo | nulo (percentil 5; média) | qualifica no teste | Δ troca dev | qualifica no dev |
|---|---|---|---|---|---|---|---|
| 5 | −0,9 [−1,8; −0,1] | −0,2 | +0,0 | −4,63; −0,47 | não | −1,7 | não |
| 20 | −0,7 [−1,5; +0,0] | +0,1 | +0,0 | −4,63; −1,09 | não | −0,8 | não |
| 100 | +10,0 [+7,4; +12,8] | −1,8 | +0,1 | −4,02; −0,36 | não | +8,3 | não |

**`enc_17`** (1.535 features com suporte):

| K | Δ troca teste (IC95%) | Δ acc teste | controle de conteúdo | nulo (percentil 5; média) | qualifica no teste | Δ troca dev | qualifica no dev |
|---|---|---|---|---|---|---|---|
| 5 | −10,0 [−12,2; −7,8] | −0,7 | +0,0 | −0,26; +0,59 | **sim** | −9,4 | sim |
| 20 | −10,9 [−13,2; −8,6] | −0,6 | +0,0 | −0,38; +0,48 | **sim** | −10,0 | sim |
| 100 | −6,6 [−9,1; −4,1] | −1,9 | +3,9 | −0,63; +2,46 | não | −5,0 | não |

- **`enc_08`: nenhuma célula qualifica** (expectativa confirmada). Com K = 100 o modelo só é perturbado (a troca sobe 10 p.p.).
- **`enc_17`: 5 ou 20 features bastam** (K = 5 e K = 20 qualificam no teste e no dev): a troca cai 10 a 11 p.p. (cerca de 12% da sensibilidade) com acurácia praticamente intacta (−0,6 p.p., IC95% inclui zero), controle de conteúdo em zero e `choice`/`noul` sem mudança; o nulo tem média +0,5 p.p. Com K = 100 o efeito diminui (−6,6) e a especificidade falha (controle +3,9). **Expectativa da emenda DESMENTIDA em `enc_17`.**
- A primeira opção vai só de −18 para −13 p.p. (+5 p.p.): essas features carregam **parte** do sinal de posição, bem menos que o subespaço inteiro da camada 22.

### 10.4 Contra as expectativas fixadas antes de rodar
| experimento | expectativa | resultado |
|---|---|---|
| E1 posição legível | ≥ acaso + 20 p.p. até a camada 22 | confirmada (1,00 até a 21) |
| E1 identidade legível | ≥ 0,9 a partir da camada 5 | desmentida na direção: 1,00 desde a 1 |
| E1 resposta sem posição | abaixo de 0,574 e acima do Laya | critério atendido (0,559); indistinguível de 0,574 |
| E2 `single` | efeito pequeno | **desmentida**: camadas 14 e 22 reduzem a troca em 16 a 18 p.p. |
| E2 `onward` | grande e pouco específico | confirmada em 1 a 10 (perturba), mas `onward` na 22 qualifica e não soma |
| E3 `enc_08` | sem célula qualificada | confirmada |
| E3 `enc_17` | sem célula qualificada | **desmentida**: K = 5 e 20 qualificam |

### 10.5 O que sustenta e os limites
- **Sustenta:** a sensibilidade à posição do `score` tem uma causa interna localizada: o sinal de posição presente nos marcadores nas camadas médias e tardias (14 e 22, com 5 a 20 features do SAE de `enc_17` carregando parte dele), que a cabeça de decisão lê. Removê-lo, sem perda de acurácia, tira entre 12% e 25% da taxa de troca (o conjunto 14 + 22 chega a 25%).
- **Não sustenta:** que seja a **única** causa (sobram ~65% de decisões trocando); que valha para `choice` ou `noul` (o efeito aparece em `score`, onde a sensibilidade e o viés posicional são grandes); o mecanismo causal só foi testado no multilíngue (o base em inglês quase não é sensível, §10.8).
- **Comparações múltiplas:** 26 células foram avaliadas no dev (13 camadas × 2 modos), e 6 no E3. Quatro células distintas qualificaram no dev em E2 (`single` nas camadas 8, 14, 17 e 22; a do `onward` na 22 é a mesma intervenção), um número que o acaso sozinho (p ≤ 0,05) poderia produzir em 1 ou 2; mas três replicam no teste (8, 14 e 22) e os efeitos das duas maiores são dezenas de vezes maiores que a dispersão do nulo. O p empírico de 0,048 é o piso possível com 20 sorteios.
- **Limites:** a remoção é só nos marcadores e o subespaço é o das médias por posição (linear); `score` é uma escala ordinal, então "posição" e "ordem esperada da escala" não se separam por rotação; o nulo do E3 mostra dispersão de vários p.p. em `enc_08`, mais larga que a do subespaço; E3 com as sementes 0 e 1 (os rankings de features diferem entre sementes, seção 9; o efeito agregado se repete, §10.7); `final_norm` e a cabeça foram testadas na emenda 8 (§10.6): o efeito vai até a `final_norm` e não há na cabeça.

### 10.6 E4, depois da camada 22 e aditividade (emenda 8; `score`)
Mesma receita do E2 (subespaço de posição estimado na descoberta, nulo de 20 subespaços aleatórios, critérios da emenda 7); job 4621; paridade 0,9975/0,9976. Na `head_1` e na `head_2` o subespaço de posição tem 1 dimensão; na `final_norm`, 5.
Aditividade = Δ(conjunto) / soma dos Δ individuais das camadas (os individuais são os do E2). Taxa de troca de base: 0,858 (dev) e 0,829 (teste).

| célula | Δ troca dev | Δ acc dev | controle de conteúdo dev | deslocamento da 1ª opção dev (base −19 p.p.) | qualifica dev | Δ troca teste (IC95%) | Δ acc teste | controle teste | qualifica teste | aditividade dev; teste |
|---|---|---|---|---|---|---|---|---|---|---|
| `final_norm` (23) só nela | −13,9 | +1,8 | −3,1 | +20,9 | **sim** | −15,0 [−18,4; −11,9] | −0,5 | −4,5 | **sim** | — |
| `head_1` (24) só nela | +8,9 | −0,2 | +10,6 | +45,6 | não | — | — | — | — | — |
| `head_2` (25) só nela | +12,5 | −3,1 | +12,5 | +60,0 | não | — | — | — | — | — |
| da 22 ao fim (`tail`) | +12,5 | −1,0 | +8,9 | +79,5 | não | — | — | — | — | — |
| da 23 ao fim (`tail`) | +13,9 | −2,2 | +7,8 | +81,0 | não | — | — | — | — | — |
| **14 + 22** | −19,7 | +3,1 | −1,4 | +24,5 | **sim** | −20,5 [−24,1; −17,1] | +2,4 | −1,4 | **sim** | 0,61; 0,60 |
| 17 + 22 | −19,2 | +2,3 | +1,9 | +32,3 | **sim** | −16,2 [−20,1; −12,4] | +0,8 | +2,5 | **sim** | 0,74; 0,55 |
| 14 + 17 + 22 | −14,2 | +1,0 | −1,1 | +38,7 | **sim** | −10,4 [−14,1; −6,6] | +0,5 | +0,6 | **sim** | 0,32; 0,22 |

- **A `final_norm` repete a camada 22** (−15,0 p.p. no teste, acurácia intacta): a mesma corrente reescalada.
- **Na cabeça de decisão remover a posição só perturba** (a troca *sobe* 9 a 14 p.p.; a posição ali já é pouco legível, 0,64 nas sondas). Por isso o `tail` (da 22 ao fim) também piora.
- **O melhor efeito de todos é o conjunto 14 + 22: −20,5 p.p. no teste (IC95% [−24,1; −17,1]), com a acurácia subindo 2,4 p.p.** e o controle de conteúdo em −1,4: cerca de **25% da sensibilidade** (0,829 → 0,624). O desvio da primeira opção passa de −19 para +5 p.p. no dev (a aversão desaparece e há um pequeno excesso).
- **Aditividade parcial:** 14+22 dá 0,61 (dev) e 0,60 (teste) da soma dos efeitos individuais, 17+22 dá 0,74 e 0,55, e 14+17+22 só 0,32 e 0,22: acrescentar a 17 piora (o efeito cai de −20,5 para −10,4). Sub-aditivo, o que combina com o `onward` não somar (§10.2).
- **Expectativas da emenda 8:** (a) `final_norm` qualifica e fica perto da 22: **confirmada**; (b) cabeça com efeito menor ou nenhum: **confirmada** (nenhum; piora); (c) conjuntos não passam do melhor efeito individual (~18 p.p.): **desmentida em parte**, 14+22 passa um pouco (−20,5); aditividade abaixo de 0,6: **no limite** (0,61 no dev, 0,60 no teste).

### 10.7 E5, o efeito das features de posição com a semente 1 (emenda 8)
Jobs 4623 (`enc_17`) e 4624 (`enc_08`); lista escolhida na descoberta com o SAE de semente 1; mesmos critérios.

**`enc_17`, semente 1** (1.532 features com suporte):

| K | Δ troca teste (IC95%) | Δ acc teste | controle | nulo (p5; média) | qualifica teste | Δ troca dev | qualifica dev |
|---|---|---|---|---|---|---|---|
| 5 | −6,8 [−8,8; −4,9] | −0,4 | +0,0 | −0,12; +0,43 | **sim** | −6,4 | sim |
| 20 | −8,4 [−10,4; −6,4] | −0,3 | +0,0 | −0,26; +0,81 | **sim** | −6,4 | sim |
| 100 | −2,4 [−4,6; −0,1] | −2,4 | +3,6 | −2,04; +1,99 | não | −3,1 | não |

**`enc_08`, semente 1:**

| K | Δ troca teste (IC95%) | Δ acc teste | controle | nulo (p5; média) | qualifica teste | Δ troca dev | qualifica dev |
|---|---|---|---|---|---|---|---|
| 5 | +1,5 [+0,6; +2,4] | +0,0 | +0,0 | −0,25; −0,04 | não | +2,2 | não |
| 20 | +1,7 [+0,8; +2,8] | −0,6 | +0,0 | −4,38; −0,71 | não | +1,1 | não |
| 100 | +8,9 [+6,1; +11,5] | −2,3 | +0,1 | −4,89; −1,30 | não | +6,7 | não |

- **O efeito agregado se repete em `enc_17`**: K = 5 e K = 20 qualificam no teste e no dev (−6,8 e −8,4 p.p. no teste; −10,0 e −10,9 com a semente 0), com acurácia intacta e controle de conteúdo em zero. **Critério da emenda 8 atendido.** O tamanho cai um pouco (−7 a −8 contra −10 a −11).
- **`enc_08` semente 1: nenhuma célula qualifica**, como na semente 0 (expectativa confirmada).
- **Sobreposição das listas** (K features de maior η² de posição da semente 0 com correspondente por decoder, cosseno ≥ 0,7, que também está entre as 100 de maior η² de posição da semente 1): K = 5, 3 têm correspondente e 3 estão na lista (0,60); K = 20, 11 e 8 (0,40); K = 100, 48 e 24 (0,24); nenhum índice coincide. Expectativa (≤ 0,5): atendida em K = 20 (0,40), não em K = 5 (0,60, com só 5 features). **Quase todas as features que têm correspondente continuam sendo de posição** (3 de 3 e 8 de 11), mas só metade delas tem correspondente: o resultado agregado é reprodutível, os índices individuais não.

### 10.8 E6, a sensibilidade em outros modelos (emenda 8; teste, 400 casos, comportamento)
Laya base e ajustado em inglês (`convaiinnovations/laya`, revisão `e333cb7c…`, configuração comum 1024/256), job 4622; o multilíngue é a referência (outra rede). Figura: `position_causal/flip_by_model.png`.

| modelo | troca `score` | troca `choice` | troca `noul` | acurácia média das rotações (rotação 0) `score` | `choice` | `noul` | 1ª opção em `score` (desvio de 25%) |
|---|---|---|---|---|---|---|---|
| Laya multilíngue | 0,833 | 0,393 | 0,062 | 0,276 (0,282) | 0,260 (0,295) | 0,487 (0,493) | −18,2 p.p. |
| Laya base (inglês) | 0,307 | 0,383 | 0,142 | 0,334 (0,326) | 0,291 (0,287) | 0,482 (0,488) | +1,3 p.p. |
| Laya ajustado (inglês) | 0,212 | 0,125 | 0,068 | 0,721 (0,728) | 0,732 (0,735) | 0,858 (0,857) | −0,2 p.p. |

Diferença da taxa de troca (pareada por caso, bootstrap): **ajustado − base**: `score` −9,5 [−14,1; −5,5], `choice` −25,8 [−30,4; −21,7], `noul` −7,3 [−10,8; −4,0]; **multilíngue − base**: `score` +52,5 [+48,2; +56,6], `choice` +1,0 [−4,2; +6,0], `noul` −8,0 [−11,5; −4,5].
- **A sensibilidade à posição do `score` é específica do checkpoint multilíngue**: o base em inglês tem 0,31 de troca (contra 0,83) e não tem a aversão à primeira opção (+1,3 p.p. contra −18,2). **As duas expectativas do protocolo sobre o base (troca ≥ 0,7 e aversão) estavam erradas.** Em `choice` o multilíngue e o base são parecidos (0,39 e 0,38).
- **Efeito do ajuste fino:** reduz a troca em `score` em 9,5 p.p. (IC95% exclui zero) e em `choice` em 25,8 p.p.; o ajustado é estável à ordem (acurácia 0,721 na média das rotações e 0,728 na ordem original em `score`). **O critério fixado (queda de pelo menos 10 p.p. em `score` com o IC95% abaixo de zero) NÃO foi atendido, por 0,5 p.p.**: o IC exclui zero, mas o limiar não foi retocado depois de ver o dado.
- **Consequência:** o mecanismo causal dos §10.2 e §10.6 vale para o multilíngue; **não sabemos se há algo parecido nos modelos em inglês** (quase não têm o fenômeno), e o que torna o multilíngue tão sensível (o treino multilíngue, a largura menor, outra coisa) não foi investigado. Ressalvas: são redes diferentes (ModernBERT-large contra mmBERT-base), a comparação não é causal, o ajustado viu o benchmark no treino (NÃO VERIFICADO qual split; só o teste vale) e o `laya` avisou que o checkpoint ajustado traz temperaturas fora do intervalo (afeta a calibração da confiança, não a escolha).

## 11. Limitações

- **Correlacional (seções 5 e 6).** O CMH controla pergunta e identidade da opção, não o resto do conteúdo do caso; features significativas são candidatas. A ablação da seção 7 testa só um recorte (as features de maior atribuição, em duas camadas, por zeragem).
- **Tokens massivos fora do SAE.** Se carregarem informação usada pelo scorer, as features não a mostram; no base em inglês eles estão associados à resposta `true` em `noul` (seção 6).
- **A atribuição é de primeira ordem e incompleta:** explica 85–87% de h·g em `enc_08`, 26–31% em `enc_17`, e ~0–15% (ou negativa) em inglês e no fine-tuned.
- **Escolhas de limiares e de estrato sem base empírica prévia** (FVE 0,80 e 0,10; fidelidade 3 p.p. e 0,10 nat; CMH por visão, emenda 3 declarada depois de ver contagens agregadas).
- **O probe é treinado no benchmark** (as predições do subconjunto "probe acerta" vêm de um leitor supervisionado), enquanto o Laya é zero-shot.
- **Fine-tuned:** contaminado em treino e dev; temperaturas fora de [0,5; 5] para perguntas com ≥ 11 opções são ajustadas pelo pacote (o typed-decisions tem ≤ 5, sem efeito aqui). O bloco `training` do config do base é idêntico ao do fine-tuned (possivelmente copiado); não usei.
- **Duplicatas e features raras:** há features com ativações idênticas; as de maior razão de chances têm pouquíssimas ativações em dev e teste.
- Duas sementes por SAE (0 e 1) e uma largura (16×). A semente 1 só tem qualidade, estatísticas de features, ablação e pareamento (seção 9); os cartões e a Parte C são da semente 0. A variante k = 64 foi treinada e medida em qualidade (seção 3), mas as análises de features e a ablação só existem para k = 32.
- Rotações só no multilíngue e só em ordem cíclica; o fine-tuned em inglês não foi rotacionado (seção 8).

## 12. Reprodutibilidade

- **Código:** branch `experiment/sae-features` (a partir de `experiment/linear-probe`). Ambiente: `uv.lock` (sha256 `68a486f3…`), Python 3.14.10, torch 2.15.0+cu132 (CUDA 13.2), laya 0.4.30, transformers 5.22.2; GPU: NVIDIA B200 MIG 3g.90gb; `pip_freeze.txt` ao lado.
- **Sementes:** 0 (inicialização do SAE, ordem dos dados, bootstrap, probes fora da dobra). Revisões: `convaiinnovations/laya` `7b928d82…` (multilíngue) e `e333cb7c…` (par em inglês); dataset `c76749ec…`.
- **Jobs (Sauron, B200 MIG, 89 GB):** extração de todos os tokens 3603 (curto) / 3605; corpus 3604 / 3606; treino do SAE 3609 (curto), 3613 (`enc_17`), 3614 (`enc_08`), 3615 (literal), 3628 (dom20); probes fora da dobra 3610;
  qualidade 3619 (`enc_08`), 3621 (`enc_17`), 3623 (literal), 3629 (dom20), 3635 (base), 3643 (cópia); features 3618/3626/3631 (`enc_08`, 3631 = estrato por visão), 3620/3627/3632 (`enc_17`, 3632 = estrato por visão);
  Parte C: extração 3617 (curto) / 3625, tokens 3624 / 3633, SAE do base 3634, features base 3636, features fine-tuned 3637, comparação 3638, marcadores massivos 3640, cópia do SAE 3641, features da cópia 3644, comparação com cópia 3647, cartões das features que mais mudam 3656.
  Ablação causal: 3874 (`enc_08`) e 3875 (`enc_17`).
  Emenda 8: `tail` e conjuntos 4621, features semente 1 4623 (`enc_17`) e 4624 (`enc_08`), rotação em inglês 4622; versões curtas 4617 a 4620.
  Posição (emenda 7): sondas 4581, subespaço 4582 (`single`) e 4583 (`onward`), features 4584 (`enc_08`) e 4585 (`enc_17`); versões curtas 4519 a 4523.
  Rotação e semente 1 (emenda 6): treino 4171/4172, extração das rotações 4173, qualidade 4174/4176, features 4175/4177, análise das rotações 4179, ablação 4180/4181, pareamento 4182.
- **RAM medida (pico, pedido):** corpus 1,8 GiB (3g); extração de tokens 8,0 GiB (8g, page cache do memmap); treino 6,0 GiB (8g); qualidade 2,8–4,0 GiB; features 2,7–4,4 GiB; Parte C: extração 9,1 GiB (10g), tokens 10,0 GiB (10g), SAE do base 7,8 GiB (10g).
- **Artefatos** (relativos à raiz; os parquets de estatísticas e `attribution/` não vão para o git: ficam no cluster em `~/workspace/sae-decision-models/results/sae_features/` e no Mac):
  `sae_quality.csv` e `sae_quality_flags_*.json`; `<SAE>/feature_stats_<visão>_<critério>.parquet` (colunas `cmh_*` e `cmh_qp_*`), `feature_stats_prior_bias.parquet`, `feature_stats_attribution.parquet`, `prior_bias_summary.json`, `decision_views.parquet`;
  `feature_cards/<SAE>/` (cartões com campo "Interpretação" em branco); `part_c/` (comparação, acurácias, marcadores massivos); pesos e configs em `checkpoints/sae/<SAE>/{sae.pt,train_meta.json,train_log.csv}` (6 SAEs, ignorados pelo git);
  `ablation/<SAE>/{selected_features.json,dev/,test/}` (ablação; `headline_dm.npz` guarda o Δm por decisão das condições principais);
  módulo reutilizável `src/sae.py` + `src/sae_intervention.py` (zera ou fixa features preservando o erro de reconstrução; o limiar de massivos vem no checkpoint).
