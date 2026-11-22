# Protocolo das etapas 4 e 5 (SAE + features), fixado antes de rodar qualquer job

Este arquivo é commitado **antes** do primeiro job. Os limiares abaixo são escolhas, sem base empírica prévia
(NÃO VERIFICADO que sejam razoáveis): servem para **sinalizar** e para decidir variantes, não para esconder resultado.
Se algum for violado, o `REPORT.md` diz qual e mostra o número. Nada aqui é ajustado depois de ver o dado; um desvio
entra como rodada nova, com o motivo, e a original fica nos artefatos.

## Pergunta e expectativa

- Pergunta: quais features de um SAE treinado numa camada do `laya-multilingual` se associam aos erros do Laya no
  typed-decisions, e a informação "mal lida" (o probe acerta, o Laya erra) aparece em features específicas?
- Expectativa antes de ver o dado: o SAE reconstrói melhor os tokens genéricos que os marcadores de opção
  (os marcadores são ~0,04% do corpus). Não há expectativa forte sobre quais features aparecem; qualquer ranking é
  correlacional. Só a fidelidade a jusante é uma medida de efeito, e mede o SAE, não as features.

## Dados e splits

- Casos: `LocalLLaMA/typed-decisions` rev `c76749ec58bd8c3d2ea706b31c333a9059c38f90`. Dev = 15% dos casos de treino
  (180 de 1.200, os de menor sha256 do `case_id`, `src.datasets.typed_dev_ids`). Descoberta = treino-menos-dev
  (1.020 casos, 5.100 decisões). Teste (400 casos, 2.000 decisões) só reporta.
- **Dev fora do treino do SAE.** Validação do SAE (FVE, fidelidade) e limiares usam o dev.
- Modelo: `convaiinnovations/laya` rev `7b928d828b7b0e022f929d9bd2e44165aa270148`, subfolder `multilingual`
  (`max_len=1024`, `head_max_len=256`, bf16 no forward, como na etapa 3).
- Camadas: `enc_17` (melhor no dev, 0,597; desempate por KL contra `enc_12`) e `enc_08` (primeira camada com ganho
  claro sobre o prior do dev: 0,583 contra 0,510). A regra "primeira camada que passa do prior" degenera: 25 de 26
  camadas já passam do prior do dev e `enc_01` ganha só 2,6 p.p., dentro do ruído de 900 decisões.

## Corpus do SAE (50M tokens por camada)

- 10% domínio: prompts de treino-menos-dev (todos os tokens não-padding, uma linha por pergunta), repetidos em ~3,2
  épocas (a base tem ~1,55M tokens).
- 45% inglês: `HuggingFaceFW/fineweb`, `sample/10BT`. 45% multilíngue: `HuggingFaceFW/fineweb-2`, 12 idiomas
  (por_Latn, spa_Latn, fra_Latn, deu_Latn, ita_Latn, rus_Cyrl, cmn_Hani, jpn_Jpan, arb_Arab, hin_Deva, tur_Latn,
  kor_Hang), cota igual por idioma. O FineWeb-2 não tem inglês.
- Texto genérico em trechos de até 512 tokens, formato `[CLS] texto [SEP]`. Sem padding. Ativações geradas durante o
  treino (streaming), com buffer embaralhado na GPU. Seed 0.

## SAE (TopK, Gao et al., arXiv:2406.04093)

- Largura 16 × hidden (12.288), k = 32 (primário). k = 64 só se sobrar tempo, rotulado como variante.
- Batch 4.096 tokens; Adam, lr 5e-4, β = (0,9; 0,999); decoder com norma unitária e remoção da componente paralela do
  gradiente; `b_dec` inicial = mediana geométrica (Weiszfeld); encoder inicial = decoderᵀ; entrada reescalada para
  norma média √hidden (escala guardada no config); AuxK com k_aux = 512 e coeficiente 1/32; latente morta = sem
  disparo nos últimos 5M tokens; seed 0 para inicialização, ordem dos dados e bootstrap.

## Qualidade do SAE e critérios de sinalização (todos no dev e em tokens de validação)

| medida | como | sinalizar se |
|---|---|---|
| FVE, todos os tokens | variância explicada em tokens do dev e em texto genérico retido | `FVE < 0,80` |
| FVE, marcadores de opção | só os `[MASK]` de opção do dev | `FVE_marcadores < FVE_todos − 0,10` |
| FVE sem [CLS]/[SEP] | idem, sem os tokens especiais (outliers de norma) | só reportado |
| L0 médio | features ativas por token | só reportado (esperado = k) |
| latentes mortas | % sem disparo na janela de 5M tokens | `> 20%` |
| fidelidade a jusante, acurácia | trocar a ativação da camada pela reconstrução (tokens todos; só marcadores), acurácia do Laya no dev | queda `> 3 p.p.` |
| fidelidade a jusante, KL | KL(saída original ‖ saída reconstruída), média por decisão | `> 0,10` nat |

Referência de escala: a mesma métrica com a camada trocada pela média e por zeros.

Se `FVE_marcadores < FVE_todos − 0,10`, treino uma variante com 20% de domínio e **reporto as duas**.

## Estatística das features (descoberta = treino-menos-dev; dev = verificação; teste = replicação)

- "Ativa" = ativação > 0. Três visões: marcador da opção prevista, marcador da opção correta, máximo sobre tokens do
  estado. Binária e ponderada pela magnitude. Estratos: tipo de pergunta, workflow.
- Suporte mínimo 20 decisões. Intervalo de Wilson 95%. Teste exato de Fisher (erro × feature ativa) com
  Benjamini-Hochberg por família (visão × estrato × critério), q = 0,05.
- Atribuição por gradiente (decisões erradas, fp32): `Δ = logit[correta] − logit[prevista]`, logits brutos,
  `a_{t,i} = z_{t,i} · (d_i · ∇_{h_t} Δ)`, agregada por feature em (marcador previsto / marcador correto / estado / demais).
  Limitação declarada: primeira ordem, sem o termo do erro de reconstrução.
- "Probe acerta, Laya erra" em treino/dev usa predições do probe **fora da dobra** (5 dobras por hash de caso, wd da
  etapa 3: 1e-5 em `enc_17`, 1e-3 em `enc_08`, seed 0). O `per_decision.parquet` da etapa 3 só entra no relato do teste.
- Viés de prior: hipóteses fixadas (severidade alta em `score`; evitar a resposta mais comum em `choice`; seguir
  `false`/`true` em `noul`), comparando a distribuição prevista com a do gabarito.
- Cartões: ~30 features por critério; campo de interpretação em branco.

## Gate da extração de todos os tokens

Os vetores nos marcadores devem reproduzir os `L08`/`L17` da etapa 3: erro relativo L2 por vetor `≤ 1e-3`
(mesmo forward bf16, vetores em fp16). A paridade das probabilidades com o `predict` segue exigindo |Δp| ≤ 1e-4.
Falhou: o job para e o dado é mostrado antes de seguir.

## Parte C (inglês): base × fine-tuned

- Ambos em `convaiinnovations/laya` rev `e333cb7c823cd1787276228260f56d02fbf5180b` (base = raiz; fine-tuned =
  subfolder `typed-decisions`, mesmos pesos do repositório avulso, OID `4fa56de7…`).
- Entrada comum: `max_len=1024`, `head_max_len=256` nos dois; hash dos `tokenizer.json` e `input_ids` linha a linha
  conferidos antes. Acurácia de cada um também com seus defaults (512/192 e 1024/256).
- Contaminação: o fine-tuned foi treinado nos workflows do benchmark (NÃO VERIFICADO em qual split). Acurácia e
  "associação com acerto" dele só no teste. O ranking do que mudou usa o conjunto de descoberta e é descritivo.
- SAE no base (16 × 1024 = 16.384, k = 32, mesmo protocolo, genérico em inglês). Se `FVE_ft < FVE_base − 0,05`,
  cópia ajustada nas ativações do fine-tuned (partindo do base, 10M tokens) e pareamento por cosseno do decoder
  (húngaro; pareada = cos ≥ 0,7; reporto quantas). Bootstrap pareado por caso, 1.000 reamostras.

## Fora de escopo agora

Ablação causal. O módulo `src/sae_intervention.py` deixa tudo pronto; ela **não** é rodada nesta etapa.

## Emenda 1: tokens massivos (feita antes de qualquer job, a partir de medição no smoke de 20 casos de treino)

**O que foi medido** (só treino, `data/activations/smoke_tokens`, 17.120 tokens): no `enc_17` o `[CLS]` tem norma média
~19.000 contra mediana 93 nos demais; os 100 `[CLS]` (0,6% dos tokens) carregam 73% da variância total, e dois canais
(580 e 488) ~79%. Além dos especiais, ~4,8% dos tokens (` the`, `_`, espaço: atenção-sink) têm norma 5× a 90× a mediana,
concentrada no canal 580. A separação é limpa: só 13 tokens entre 2× e 5× a mediana. Os marcadores de opção ficam no
regime normal (mediana 92, p99 110; 0 de 360 massivos). No `enc_08` nenhum token passa de 5× a mediana (a regra não
exclui nada). Num SAE minúsculo de teste, a FVE do protocolo literal deu 0,9996 (trivial, só reconstrói os massivos),
e 0,85 (domínio dev) e 0,63 (genérico) excluindo-os. Além de a FVE literal ser enganosa, a MSE e a escala (norma média)
ficam dominadas por esses tokens, o que atrapalha o aprendizado dos tokens comuns, onde estão os marcadores.

**Mudanças** (o resto do protocolo vale como estava):

1. **Token massivo** := norma da ativação na camada `> 5 ×` a mediana das normas dos tokens não especiais (`[CLS]`,
   `[SEP]` de cabeça e final ficam fora da mediana) do domínio treino-menos-dev. Limiar fixado uma vez por camada,
   sem rótulos e sem teste, e guardado dentro do SAE (`massive_threshold`).
2. O SAE treina **só em tokens não massivos** (domínio e genérico). A escala de entrada (norma média √hidden) é
   calculada sobre eles.
3. **FVE "todos os tokens" passa a ser a FVE sobre tokens não massivos**; os limiares de sinalização (0,80 e a
   diferença de 0,10 para os marcadores) valem para ela. A FVE literal (todos os não-padding) fica como diagnóstico.
4. **Fidelidade a jusante primária:** substituir pela reconstrução só os tokens não massivos (os massivos ficam
   intocados). Diagnóstico: substituir todos. Limiares inalterados (queda > 3 p.p.; KL > 0,10 nat).
5. Estatísticas de features e atribuição só em tokens não massivos. Reporto a fração de tokens massivos por papel e
   por fonte. Sinalizar se mais de 1% dos marcadores de opção forem massivos em alguma camada.
6. **Rodada literal (protocolo original, todos os tokens)** para `enc_17`, com `--massive-mult inf`, rotulada
   "literal" e reportada ao lado, para mostrar o efeito.

**Limitação a declarar:** a regra deixa os tokens massivos fora do SAE; se eles carregarem informação usada pelo
scorer, isso não aparece nas features. A fidelidade a jusante mede o efeito de reconstruir só o resto.

## Emenda 2: pareamento de features na Parte C e escolha da camada em inglês (antes de qualquer job da Parte C)

1. **Pareamento por vizinho mais próximo mútuo, não húngaro.** O protocolo dizia "húngaro" sobre o cosseno dos
   decoders. Com 16.384 × 16.384 features a atribuição húngara densa (~1 GB em float32 e custo cúbico) não é prática; o
   pareamento passa a ser por **vizinho mais próximo mútuo** (a melhor correspondência de A em B é também a melhor de B em
   A) com cosseno ≥ 0,7. Só vale quando a cópia do SAE ajustada ao fine-tuned é necessária (FVE_ft < FVE_base − 0,05);
   a cópia parte do SAE do base, então o alinhamento esperado é alto. Reporto quantas features ficam pareadas.
2. **Camada em inglês:** a do base com maior acurácia do probe no **dev** (não no teste), entre as camadas do backbone
   (`enc_01` a `enc_28`); empate pela menor KL, como na etapa 3. Se a camada escolhida tiver massivos (regra da emenda 1,
   recalculada para esse modelo), vale a mesma exclusão.
3. **Corpus do SAE em inglês:** 10% domínio (treino-menos-dev) e 90% inglês (FineWeb), mesmo total de 50M tokens, k = 32,
   largura 16 × 1.024 = 16.384, seed 0.
4. **FVE e fidelidade** do base com os mesmos limiares da emenda 1. A comparação com o fine-tuned usa as mesmas entradas
   (`max_len` 1.024, `head_max_len` 256); só acurácia e "associação com acerto" do fine-tuned são lidas no teste.

## Emenda 3: estratos do teste de Cochran-Mantel-Haenszel por visão (feita depois de ver os resultados do `enc_08`)

**O que motivou** (dado de treino-menos-dev, replicado em dev e teste): o ranking ingênuo por taxa de erro na visão "marcador
da opção prevista" é dominado por identidade da opção. Várias das 12 features de maior lift compartilham a mesma confusão
(`action|3>1`, 111 decisões) e Laya escolhe o índice 3 em 47% das decisões de `choice` contra 9% do gabarito. Primeira
passada do CMH (estratos = pergunta × opção prevista, igual para as três visões): das 364 features ingênuas na visão
prevista só 9 (mais erro) e 11 (menos erro) sobrevivem; na visão de estado 3.226 caem para 42 e 77; na visão da opção
correta 476 caem para 419, o que sugere que o estrato estava errado para essa visão.

**Mudança:** o estrato passa a ser a identidade da opção a que a visão se refere: `pred` = pergunta × opção prevista;
`corr` = pergunta × opção correta; `state` = só a pergunta (um token de estado não pertence a uma opção; condicionar na
opção prevista, que é consequência do estado, seria controlar um mediador). Colunas `cmh_*` = estrato por visão (primário);
`cmh_qp_*` = primeira passada, mantida. A escolha foi feita **depois** de ver as contagens do `enc_08` (mas só no conjunto
de descoberta/dev/teste agregado, sem selecionar nenhuma feature), então vale como análise exploratória declarada, e o
`REPORT.md` mostra as duas.

**Limites que seguem valendo:** o CMH controla pergunta e identidade de opção, não o resto do conteúdo do caso. Features
significativas aqui são candidatas, correlacionais.

## Emenda 4: ablação causal de features (etapa 6; liberada pelo usuário em 18/11/2026, depois de fechados o relatório e o k=64)

A ablação ficou fora das etapas 4 e 5. Este texto é escrito **antes** de qualquer execução; a única coisa vista antes foi o smoke de encanamento
em poucos casos do conjunto de **descoberta** (nenhum dev, nenhum teste), só para conferir que o código roda.

**Pergunta.** Zerar, nas ativações da camada, features cujo efeito linear de primeira ordem empurra para a resposta errada (atribuição por gradiente, seção 5 do relatório)
muda de fato a saída do Laya na direção da resposta correta, mais do que zerar features sem relação com o erro?

**Expectativa (antes de ver o dado).** Efeito pequeno e disperso. A atribuição de primeira ordem explica 85–87% de h·g em `enc_08` mas só 26–31% em `enc_17`, e as features individuais têm
atribuição média de ~0,001 a 0,013 logit; o bias de prior parece distribuído por muitas features. Espero que o conjunto de K = 20 features desloque a margem na direção prevista, mas pouco, e
que a previsão de primeira ordem acerte o sinal e subestime a magnitude, principalmente em `enc_17`. Não espero corrigir uma fração grande dos erros.

**Camadas e SAEs.** Primária: `enc_08` com `ml_enc08_k32` (onde a atribuição é mais completa). Secundária, mesma receita: `enc_17` com `ml_enc17_k32`. Resultados sempre rotulados por camada; sem escolher a melhor depois.

**Seleção das features (só na descoberta = treino-menos-dev).** Atribuição média no marcador da opção prevista (`a_pred`) sobre as decisões erradas da descoberta, entre as features com suporte
(≥ 20 decisões com a feature ativa no marcador previsto, como no resto do protocolo).
- Lista **neg** (candidatas a empurrar para a resposta errada): as 50 de `a_pred` média mais negativa. Lista **pos** (controle positivo): as 50 mais positivas.
- A seleção é gravada em `selected_features.json` **antes** de rodar dev e teste e é reutilizada tal como está.

**Intervenção.** `src/sae_intervention.py`, modo `edit` (preserva o erro de reconstrução), zerando as features do conjunto. Duas variantes, declaradas antes:
(i) **marcadores**: só nos tokens `[MASK]` das opções (primária); (ii) **todos os tokens** não massivos (secundária; é a que a atribuição de primeira ordem prevê exatamente).
Conjuntos: as K primeiras de cada lista, K ∈ {1, 5, 20}; **K = 20 é o desfecho principal**.

**Dados.** Dev (180 casos, 900 decisões) para a análise; teste (400 casos, 2.000 decisões) só como replicação das mesmas listas, sem reescolher nada. Os números do dev e do teste são sempre mostrados lado a lado.

**Medidas.** Por decisão, com `pred0` = argmax dos logits brutos sem intervenção (bf16, como no `predict`, em lotes por workflow) e `gold` = rótulo: margem m = logit[gold] − logit[pred0].
- **Desfecho principal:** Δm médio sobre as decisões em que o Laya (sem intervenção) erra. Positivo = a intervenção aproxima da resposta correta. IC95% por bootstrap pareado sobre casos (1.000 reamostras).
- Secundários: fração dos erros corrigidos (argmax passa a `gold`); fração dos acertos quebrados; variação de acurácia em todas as decisões.
- **Nulo:** 100 conjuntos aleatórios por K e por variante, sorteados entre as features com suporte, **pareados com a lista neg por quintil de frequência de ativação no marcador previsto** (n_active na descoberta).
  O mesmo nulo serve de referência para a lista pos (que pode ter outra distribuição de frequências; declarado como limitação).
- **Calibração da previsão de primeira ordem:** 25 features individuais de cada lista, variante "todos os tokens": correlação de Spearman entre o Δm medido e −(atribuição total média da feature, soma dos quatro grupos) nas mesmas decisões erradas.
- **Encanamento:** concordância entre o argmax em lote sem intervenção e o `model_pred` oficial (esperada ≥ 98%; abaixo disso, parar e olhar).

**Critério de decisão (fixado agora).** Há **efeito causal distinguível** do conjunto neg de K = 20 na camada/variante se, no **dev**, (a) o IC95% do Δm exclui zero e (b) o Δm fica acima do percentil 95 do nulo
(p empírico unilateral ≤ 0,05); e há **replicação** se, no **teste**, o Δm tem o mesmo sinal e também passa de (b). O controle positivo (lista pos) deve dar Δm negativo e abaixo do percentil 5 do nulo.
Sem replicação, ou com controle positivo falhando, o texto diz que o efeito não foi demonstrado. Nada é reajustado depois de ver o dado; se a leitura ficar ambígua, mostro as duas.

**Limites já conhecidos.** O SAE é ajustado a ativações, não a efeitos; zerar uma feature move a saída também por vias de segunda ordem. Com nulo de 100 sorteios, p empírico não passa de 0,0099. A ablação mede o que *este SAE* isola, não "o conceito" da feature (sem rótulos).
Features duplicadas (ativações quase idênticas) podem entrar juntas na lista e contar duas vezes; não foram removidas.

## Emenda 5: variantes da ablação alinhadas à atribuição (feita depois de um teste de encanamento em 80 casos de descoberta; antes de qualquer dev ou teste)

**O que o teste de encanamento mostrou** (só descoberta, 80 casos, 400 decisões; nenhum dev, nenhum teste): (a) a previsão de primeira ordem acerta bem a maioria das features individuais
(por exemplo feature 9247: medido −0,0096 contra previsto −0,0094), com exceções (feature 4689: −0,078 contra +0,019); (b) o controle positivo da emenda 4 saiu com sinal **oposto** ao previsto (lista pos, K = 20,
variante "marcadores": Δm = +0,67 em vez de negativo). Causa: a lista é escolhida pela atribuição no marcador da opção **prevista** (`a_pred`), mas a variante "marcadores" zerava as features em **todos** os marcadores,
inclusive o da opção correta, onde várias dessas features (10060, 5674, 2474) têm atribuição fortemente negativa. A intervenção não correspondia ao que a atribuição mede. Isso é falha do meu desenho, não resultado.
O mesmo teste mostrou que o efeito de K = 20 features é pequeno (neg: Δm ≈ +0,02 logit, contra uma margem média das decisões erradas de ~2 logits).

**Mudanças** (substituem as variantes e as listas da emenda 4; todo o resto da emenda 4 vale):
1. **Duas variantes, cada uma com a sua lista de seleção (só descoberta):**
   - `pred` (primária): zera as features **apenas no token `[MASK]` da opção prevista** (a opção que o Laya escolhe sem intervenção). Seleção: `a_pred` média nas decisões erradas da descoberta, suporte ≥ 20 decisões com a feature ativa no marcador previsto (visão `pred`).
   - `all`: zera as features em **todos os tokens não massivos**. Seleção: atribuição total (soma dos quatro grupos `a_pred + a_corr + a_state + a_other`), suporte ≥ 20 decisões na visão `state`. É a variante que a primeira ordem prevê exatamente.
   - A variante "todos os marcadores" foi retirada.
2. **K ∈ {1, 5, 20, 100}** (listas neg e pos de 100); K = 20 segue como desfecho principal e K = 100 é reportado também.
3. **Nulo: 50 conjuntos por K e por variante** (era 100; o custo do cluster dobrava), pareados pela lista neg da variante por quintil de n_active. O menor p empírico possível é 1/51 ≈ 0,02.
4. **Relevância, além de distinguibilidade.** Um efeito pode passar no critério da emenda 4 e ser desprezível. Fixo agora: o efeito é **relevante** se Δm ≥ 10% da margem média |m| das decisões erradas (medida no dev) **ou** se corrige ≥ 1% das decisões erradas.
   Este limiar é convencional; foi escrito **depois** de eu ter visto que o efeito de K = 20 na descoberta é de ~1% da margem, então não é um limiar neutro: ele tende a classificar efeitos desse tamanho como irrelevantes, e isso fica dito.
5. **Calibração:** 25 features individuais de cada lista (neg e pos), nas duas variantes, medido contra a previsão de primeira ordem exata de cada variante (`−média de a_pred` na variante `pred`; `−média da atribuição total` na `all`), no dev.
6. **Teste (replicação):** só as listas e os nulos de K = 20 e K = 100; sem condições individuais nem calibração (isso fica no dev). As listas são as do dev, sem reescolher nada.
7. **Critério de decisão:** o da emenda 4 (IC95% do Δm exclui zero e fica acima do percentil 95 do nulo no dev; replicação no teste com mesmo sinal e acima do percentil 95; controle positivo negativo e abaixo do percentil 5), agora aplicado a cada variante, **mais** a relevância do item 4.
   Distinguível mas irrelevante será escrito como tal.

**Adendo à emenda 5 (escrito depois de um 2º teste de encanamento, em 40 casos de descoberta, ainda antes de qualquer dev ou teste).** Esse teste mostrou: variante `pred`, calibração da primeira ordem quase exata
(Spearman 0,997, medido ≈ 0,83 do previsto), controle positivo com o sinal certo (Δm = −0,21 para K = 20) e efeito da lista neg pequeno (+0,027 logit, 1,2% da margem, 0 erros corrigidos); variante `all`, efeito grande
(+0,90 logit, 10% dos erros corrigidos) **mas com 43% das decisões certas quebradas**, e calibração fraca (Spearman 0,58). Como "relevante" pelo item 4 poderia premiar uma intervenção que só desorganiza o modelo, fixo agora:
- Um efeito só conta como **relevante** se, além do item 4, a **acurácia total não cair** (variação de acurácia em todas as decisões ≥ 0). Caso contrário o texto diz "efeito grande, mas não específico: a intervenção quebra decisões certas".
- A variante `pred` é a que responde à pergunta causal (efeito das features no marcador da opção escolhida); a `all` é reportada como controle de escala e de calibração, não como evidência de mecanismo.
- Um recurso de implementação do nulo: quando a faixa de frequência de uma feature não tem candidatos livres (a faixa mais alta é pequena), o sorteio usa a faixa vizinha mais próxima; o número de sorteios nessa situação é gravado em `summary.json` (no smoke: 44 de 1.008).

## Emenda 6: rotação de opções (ponte com o viés posicional) e segunda semente do SAE (escrita antes de qualquer execução, em 22/11/2026)

Motivo: o trabalho final trata de viés posicional em modelos System One, e até aqui as opções nunca foram permutadas, então posição e conteúdo da opção não estão separados (o viés de índice 3 em `choice`, por exemplo, é ambíguo).
A semente única dos SAEs é a outra fragilidade. Nada abaixo foi visto antes de escrever.

### Parte A: rotação de opções

**Desenho.** Para cada decisão com k opções, as k rotações cíclicas da ordem de apresentação: `option_order[j] = (j + r) mod k`, r = 0..k−1 (r = 0 é a ordem original; o pacote `laya` aceita `option_order` em `choice`, `score` e `noul`).
O gabarito não muda. Como cada opção passa uma vez por cada posição, o desenho é balanceado: a média sobre as rotações separa a preferência por **posição** da preferência por **conteúdo**.
Treino (para as análises A3 e A4) e teste (replicação e A1, A2). Os mesmos `predict` oficiais, vetores nos `[MASK]` de todas as camadas e probabilidades por rotação.

**A1, comportamento.** (i) Preferência posicional: P(escolher a posição j), média sobre rotações, contra 1/k, por tipo de pergunta, com IC95% por bootstrap sobre casos. (ii) Taxa de troca: fração de decisões cuja opção escolhida muda em alguma rotação.
(iii) Acurácia por rotação e média, contra a da ordem original. **Expectativa:** o viés de índice 3 em `choice` é em parte posicional. **Critério:** há viés posicional em `choice` se, no teste, alguma posição se afasta de 1/k em pelo menos 5 p.p. com IC95% fora de zero.
Se a preferência por posição for < 5 p.p. em todos os tipos, o texto dirá que o viés de prior da seção 4 é majoritariamente de conteúdo.

**A2, o probe lê conteúdo ou posição.** O probe linear da etapa 3 (camadas `enc_08` e `enc_17`, wd e treino da etapa 3, **só na ordem original**) aplicado aos marcadores rotacionados do teste. Medida: amplitude da acurácia entre a rotação de melhor e a de pior desempenho, para o probe e para o Laya, com IC por bootstrap sobre casos.
**Expectativa:** o probe é mais estável (menor amplitude) que o Laya. **Critério:** "mais estável" se a amplitude do probe for menor que a do Laya com IC95% da diferença fora de zero.

**A3, onde a posição e o conteúdo estão nas camadas.** ANOVA de dois fatores por pergunta sobre o vetor do marcador, camada a camada (o desenho de rotações cíclicas é um quadrado latino: posição e conteúdo são ortogonais): η² de posição e η² de conteúdo (soma de quadrados entre grupos sobre a soma total, somando as 768 dimensões, média sobre perguntas ponderada por n).
**Expectativa:** a posição domina nas camadas iniciais e o conteúdo cresce com a profundidade. Descritivo: sem limiar de decisão.

**A4, ligação com as features e a ablação.** η² de posição e de conteúdo de cada feature do SAE (camadas 8 e 17, k = 32, seed 0) nos marcadores do treino rotacionado. Compara-se a média de η² de posição das listas neg da ablação (variante `pred`, K = 20 e 100) com a de 100 conjuntos aleatórios pareados por frequência de ativação (mesma regra do nulo da emenda 5).
**Expectativa:** sem expectativa forte. **Critério:** a lista é "mais posicional que o acaso" se a média de η² de posição ficar acima do percentil 95 do nulo; "mais de conteúdo" se η² de conteúdo ficar acima do percentil 95. O teste rotacionado só replica o que o treino mostrar.

**Limites declarados.** Rotações cíclicas não esgotam as permutações (k! para k ≥ 3); a análise é do efeito de posição por deslocamento, não de todas as ordens. `score` tem níveis ordenados: permutar a ordem de apresentação mantém os rótulos "level i" no texto, então o conteúdo continua identificável, mas é um tipo de pergunta diferente dos demais; reporto por tipo.
Os η² usam o treino rotacionado em todos os casos (descoberta e dev juntos), sem selecionar nada.

### Parte B: segunda semente do SAE

SAE com `--seed 1` (mesmos hiperparâmetros, ordem dos dados e inicialização diferentes), k = 32, em `enc_08` e `enc_17`. Medidas: (i) qualidade (mesmos limiares da emenda 1); (ii) fração de features pareadas com a semente 0 por vizinho mútuo (cosseno ≥ 0,7);
(iii) **estabilidade da lista**: das 20 features da lista neg da semente 0 (variante `pred`), a fração com correspondente pareado na semente 1 que também está entre as 100 de maior atribuição negativa da semente 1; (iv) a ablação (variante `pred`, K = 20) com a lista da semente 1.
**Expectativa:** qualidade equivalente; pareamento alto (> 80%) porque as camadas são as mesmas e os dados também; estabilidade da lista moderada. **Critério:** a conclusão da ablação é "robusta à semente" se, na semente 1, o efeito de K = 20 tiver o mesmo sinal, ficar acima do percentil 95 do nulo e levar à mesma leitura de relevância;
a lista é "estável" se (iii) ≥ 50%. Senão, o texto diz que o ranking de features depende da semente. O item (iv) é o mais caro e fica condicionado ao tempo do cluster; se não rodar, o texto diz.
