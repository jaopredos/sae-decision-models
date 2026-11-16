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
