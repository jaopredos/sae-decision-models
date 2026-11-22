# SAE e features que explicam o erro do Laya no typed-decisions (etapas 4 e 5)

Tudo aqui é **correlacional**, exceto a fidelidade a jusante (que mede o SAE, não as features). A **ablação causal** da seção 7 foi feita depois, com protocolo próprio. Critérios e limiares foram fixados antes dos jobs em `PROTOCOLO.md`; as seis
emendas (feitas por causa do que os dados mostraram; a 4 e a 5 são da ablação, a 6 da rotação e da semente) estão datadas lá e resumidas na seção 2 e nas seções 7 a 9. As tabelas completas, geradas dos artefatos
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
- **Consequência para a etapa 3:** o 0,574 do probe em `enc_17` vem de um probe treinado e testado na ordem original; esta análise mostra que, por conteúdo apenas, o ganho é menor (~0,42 nas rotações 1 a 3 em k ≥ 4, contra 0,52 na ordem original, nas mesmas decisões). Não treinei um probe nas rotações (que mediria a legibilidade do conteúdo sem a posição correlacionada); fica como análise adicional.

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

## 10. Limitações

- **Correlacional (seções 5 e 6).** O CMH controla pergunta e identidade da opção, não o resto do conteúdo do caso; features significativas são candidatas. A ablação da seção 7 testa só um recorte (as features de maior atribuição, em duas camadas, por zeragem).
- **Tokens massivos fora do SAE.** Se carregarem informação usada pelo scorer, as features não a mostram; no base em inglês eles estão associados à resposta `true` em `noul` (seção 6).
- **A atribuição é de primeira ordem e incompleta:** explica 85–87% de h·g em `enc_08`, 26–31% em `enc_17`, e ~0–15% (ou negativa) em inglês e no fine-tuned.
- **Escolhas de limiares e de estrato sem base empírica prévia** (FVE 0,80 e 0,10; fidelidade 3 p.p. e 0,10 nat; CMH por visão, emenda 3 declarada depois de ver contagens agregadas).
- **O probe é treinado no benchmark** (as predições do subconjunto "probe acerta" vêm de um leitor supervisionado), enquanto o Laya é zero-shot.
- **Fine-tuned:** contaminado em treino e dev; temperaturas fora de [0,5; 5] para perguntas com ≥ 11 opções são ajustadas pelo pacote (o typed-decisions tem ≤ 5, sem efeito aqui). O bloco `training` do config do base é idêntico ao do fine-tuned (possivelmente copiado); não usei.
- **Duplicatas e features raras:** há features com ativações idênticas; as de maior razão de chances têm pouquíssimas ativações em dev e teste.
- Duas sementes por SAE (0 e 1) e uma largura (16×). A semente 1 só tem qualidade, estatísticas de features, ablação e pareamento (seção 9); os cartões e a Parte C são da semente 0. A variante k = 64 foi treinada e medida em qualidade (seção 3), mas as análises de features e a ablação só existem para k = 32.
- Rotações só no multilíngue e só em ordem cíclica; o fine-tuned em inglês não foi rotacionado (seção 8).

## 11. Reprodutibilidade

- **Código:** branch `experiment/sae-features` (a partir de `experiment/linear-probe`). Ambiente: `uv.lock` (sha256 `68a486f3…`), Python 3.14.10, torch 2.15.0+cu132 (CUDA 13.2), laya 0.4.30, transformers 5.22.2; GPU: NVIDIA B200 MIG 3g.90gb; `pip_freeze.txt` ao lado.
- **Sementes:** 0 (inicialização do SAE, ordem dos dados, bootstrap, probes fora da dobra). Revisões: `convaiinnovations/laya` `7b928d82…` (multilíngue) e `e333cb7c…` (par em inglês); dataset `c76749ec…`.
- **Jobs (Sauron, B200 MIG, 89 GB):** extração de todos os tokens 3603 (curto) / 3605; corpus 3604 / 3606; treino do SAE 3609 (curto), 3613 (`enc_17`), 3614 (`enc_08`), 3615 (literal), 3628 (dom20); probes fora da dobra 3610;
  qualidade 3619 (`enc_08`), 3621 (`enc_17`), 3623 (literal), 3629 (dom20), 3635 (base), 3643 (cópia); features 3618/3626/3631 (`enc_08`, 3631 = estrato por visão), 3620/3627/3632 (`enc_17`, 3632 = estrato por visão);
  Parte C: extração 3617 (curto) / 3625, tokens 3624 / 3633, SAE do base 3634, features base 3636, features fine-tuned 3637, comparação 3638, marcadores massivos 3640, cópia do SAE 3641, features da cópia 3644, comparação com cópia 3647, cartões das features que mais mudam 3656.
  Ablação causal: 3874 (`enc_08`) e 3875 (`enc_17`).
  Rotação e semente 1 (emenda 6): treino 4171/4172, extração das rotações 4173, qualidade 4174/4176, features 4175/4177, análise das rotações 4179, ablação 4180/4181, pareamento 4182.
- **RAM medida (pico, pedido):** corpus 1,8 GiB (3g); extração de tokens 8,0 GiB (8g, page cache do memmap); treino 6,0 GiB (8g); qualidade 2,8–4,0 GiB; features 2,7–4,4 GiB; Parte C: extração 9,1 GiB (10g), tokens 10,0 GiB (10g), SAE do base 7,8 GiB (10g).
- **Artefatos** (relativos à raiz; os parquets de estatísticas e `attribution/` não vão para o git: ficam no cluster em `~/workspace/sae-decision-models/results/sae_features/` e no Mac):
  `sae_quality.csv` e `sae_quality_flags_*.json`; `<SAE>/feature_stats_<visão>_<critério>.parquet` (colunas `cmh_*` e `cmh_qp_*`), `feature_stats_prior_bias.parquet`, `feature_stats_attribution.parquet`, `prior_bias_summary.json`, `decision_views.parquet`;
  `feature_cards/<SAE>/` (cartões com campo "Interpretação" em branco); `part_c/` (comparação, acurácias, marcadores massivos); pesos e configs em `checkpoints/sae/<SAE>/{sae.pt,train_meta.json,train_log.csv}` (6 SAEs, ignorados pelo git);
  `ablation/<SAE>/{selected_features.json,dev/,test/}` (ablação; `headline_dm.npz` guarda o Δm por decisão das condições principais);
  módulo reutilizável `src/sae.py` + `src/sae_intervention.py` (zera ou fixa features preservando o erro de reconstrução; o limiar de massivos vem no checkpoint).
