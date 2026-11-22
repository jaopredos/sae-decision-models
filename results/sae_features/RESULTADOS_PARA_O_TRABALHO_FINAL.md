# Resultados para o trabalho final: probes, SAEs e ablação no Laya (typed-decisions)

Documento de passagem para quem vai escrever o trabalho final. Ele resume o que foi feito e medido, com os números que podem ser citados, o que se pode afirmar e o que **não** se pode.
Os detalhes e as tabelas completas estão em `results/sae_features/REPORT.md` (técnico) e `results/probe_typed/REPORT.md` (probes). Quando um número aparecer aqui, a fonte está entre parênteses.
Tudo foi calculado no branch `experiment/sae-features` do repositório `jaopredos/sae-decision-models`.

---

## 1. Em uma página

**Pergunta.** O Laya (modelo de decisão "System One" baseado em ModernBERT/mmBERT) erra muito no benchmark typed-decisions em zero-shot: o checkpoint multilingual acerta ~35%, abaixo de sempre responder
a classe mais comum de cada pergunta (48%) e pouco acima do acaso (32%). **A informação para acertar está dentro do modelo? Se está, o que o impede de usá-la?**

**O que encontramos, em cinco frases.**
1. **A informação existe.** Um classificador linear simples (probe) lendo as representações internas, no token `[MASK]` de cada opção, acerta **57,4%** no teste, 22 pontos acima do próprio Laya e 9 acima do prior.
2. **O erro tem um viés sistemático e forte:** o Laya foge da resposta mais comum em perguntas de escolha e responde `true` em perguntas sim/não muito mais do que o gabarito. **Esse viés é de conteúdo, não de posição** (teste com as opções rotacionadas, §8); a posição pesa muito só nas perguntas de nota (`score`).
3. **Treinamos SAEs (autoencoders esparsos) em duas camadas.** Eles reconstroem bem as ativações e são fiéis ao comportamento do modelo; as features que acompanham o erro são, na maior parte, **lexicais e ligadas à opção escolhida**, não conceitos claros.
4. **Intervir nessas features funciona só um pouco:** zerar as 20 features de maior atribuição no marcador da opção escolhida move o modelo na direção da resposta certa, de forma replicável, mas o efeito é **pequeno** (camada 8) ou **não específico** (camada 17, derruba quase tantas decisões certas quanto conserta).
5. **Comparando o Laya base com a versão ajustada ao benchmark (inglês),** as features que mais mudam existem e replicam, mas a leitura delas é limitada, e o SAE do base lê mal o modelo ajustado.

**Em uma linha para o resumo do trabalho:** o Laya sabe mais do que diz (a resposta é legível linearmente nas camadas intermediárias), e o erro vem de uma leitura final enviesada por preferências de prior espalhadas em muitas features, não de poucas "features do erro".

**Duas correções importantes depois da rotação das opções (§8 e §9):** (i) com a ordem das opções balanceada, a acurácia do Laya (0,334) fica indistinguível do acaso (0,318), então o "Laya acerta um pouco acima do acaso" depende da ordem original; (ii) o ranking de features individuais **depende da semente do SAE** (só ~50% das features têm correspondente); só o efeito agregado da ablação se repete.

---

## 2. Dados e protocolo (o que precisa estar no trabalho para o resultado valer)

- **Dataset:** `LocalLLaMA/typed-decisions`, revisão `c76749ec58bd8c3d2ea706b31c333a9059c38f90`. 1.200 casos de treino (6.000 decisões) e 400 de teste (2.000 decisões), 4 workflows (invoice, security incidents, customer service, agent trace),
  5 perguntas por caso, três tipos de pergunta (`choice`, `score`, `noul` = sim/não). O gabarito é uma distribuição suave; a "resposta correta" é o rótulo (`label`), que está sempre entre os de maior probabilidade.
- **Modelo principal:** `convaiinnovations/laya`, subpasta `multilingual` (mmBERT-base, 22 camadas, 768 dimensões), revisão `7b928d82…`. Par em inglês: `convaiinnovations/laya` (base) e `typed-decisions` (ajustado), revisão `e333cb7c…`.
- **Separação dos dados.** *Dev* = 15% dos casos de treino, escolhidos por hash do id (180 casos, 900 decisões). *Descoberta* = treino sem o dev. **O teste nunca foi usado para escolher nada** (camada, hiperparâmetro, feature): só para medir e replicar.
- **Critérios fixados antes de rodar** estão em `PROTOCOLO.md`; mudanças viraram "emendas" datadas (6 no total). O trabalho final deve dizer que houve emendas e por quê (ver `PROTOCOLO.md` e `REPORT.md` §2).
- **A entrada do modelo é a do próprio pacote `laya`** (não reimplementamos a formatação), com os limites do checkpoint (1024 tokens no multilíngue). Nenhum caso teve o texto cortado.
- **Conferência de exatidão:** as ativações foram capturadas dentro do `predict` oficial e as probabilidades recalculadas bateram com as do pacote (diferença 0 nas 8.000 decisões).

---

## 3. Probe linear por camada (etapa 3)

**O que é:** em cada uma das 26 pontos do modelo (embeddings, 22 camadas, normalização final, 2 camadas da cabeça de decisão) treinamos um pontuador linear sobre o vetor do `[MASK]` de cada opção, com softmax entre as opções e perda contra o gabarito suave.
O **controle** é o mesmo probe treinado com os gabaritos embaralhados entre casos da mesma pergunta (mede o que dá para acertar só sabendo a pergunta e as opções).

| modelo | acurácia no teste |
|---|---|
| **Probe, camada 17 (escolhida no dev)** | **0,574 ± 0,001** |
| Prior (resposta mais comum por pergunta, do treino) | 0,483 |
| Controle (rótulos embaralhados) | 0,462 ± 0,021 |
| Laya multilingual | 0,349 |
| Aleatório | 0,318 |

- Probe contra o Laya: **+22,5 p.p.** (IC95% por bootstrap sobre casos: +19,4 a +25,7). Probe contra o prior: **+9,1 p.p.** (IC95% +6,8 a +11,6). Contra o controle: +11 p.p. (a parte que vem do caso específico).
- **Forma da curva:** sobe rápido até a camada ~3 (0,54) e fica num patamar de ~0,55–0,58 da camada 8 até a normalização final; nas duas camadas da cabeça de decisão cai para 0,52 e 0,51. A camada 19 tem o maior teste (0,583), mas foi o dev que escolheu a 17 (empate com a 12); a diferença é ruído.
- A camada `emb` (0,27) é só piso de sanidade: lá todos os marcadores são o mesmo vetor.
- **Ressalva obrigatória:** o probe é treinado com dados do próprio benchmark e o Laya é zero-shot. "Probe > Laya" mostra que a informação é legível linearmente, não que o Laya "deveria" acertar sem esses dados.
- **Ressalva da rotação (§8):** o probe foi treinado e testado na ordem original das opções. Com as opções rotacionadas, o da camada 8 cai de 0,50 para 0,27 (perto do acaso), então nessa camada ele depende da ordem; o da camada 17 cai de 0,52 para 0,42, que ainda é bem acima do acaso (~0,24) e do Laya: aí há conteúdo legível. **O 0,574 não deve ser apresentado como "conteúdo puro".**
- **Prior com três valores:** o enunciado original citava 0,461; pela definição usada (resposta mais frequente por workflow e pergunta no treino) o valor é 0,483 no teste e 0,510 no dev. Não conseguimos reproduzir 0,461; **o trabalho deve usar 0,483 e dizer qual definição usou.**
- Fonte: `results/probe_typed/REPORT.md`, `results/probe_typed/probe_by_layer.csv`.

---

## 4. O viés do Laya (independe de SAE)

Fração das decisões em que o Laya escolhe a opção indicada, comparada com a do gabarito (replica em descoberta, dev e teste; fonte: `REPORT.md` §4):

| hipótese | Laya | gabarito (teste) |
|---|---|---|
| `choice`: classe mais comum da pergunta | 17,7% | 41,8% |
| `noul`: responde `true` | 77,2% | 50,8% |
| `score`: nível mais alto | 17,6% | 26,9% |

- Em `choice`, o Laya escolhe o índice 3 em 47% das decisões (gabarito: 9%) e o índice 2 em 5% (gabarito: 45%).
- **Viés posicional (teste de rotação, §8):** nesta seção a ordem não era permutada; depois, rodamos o Laya nas k rotações cíclicas das opções. Em `choice` o maior desvio por posição é 3,2 p.p. (abaixo do limiar pré-fixado de 5 p.p.): o viés acima é de **conteúdo**. Em `score` o Laya quase nunca escolhe a primeira opção (6,8% contra 25%) e favorece a última (+11 p.p.): aí há viés posicional grande.

---

## 5. SAEs: o que foram treinados e se são confiáveis

**O que é um SAE aqui:** um autoencoder esparso TopK (k = 32 ou 64) com 16 vezes a largura do modelo, treinado em ativações de uma camada para decompô-las em "features" esparsas. Treinado com 50M de tokens (10% de prompts do próprio benchmark, 45% de texto em inglês e 45% em 12 outros idiomas).

| SAE | FVE (variância explicada) no dev | FVE só nos marcadores | efeito no Laya ao trocar a ativação pela reconstrução (acurácia / KL) |
|---|---|---|---|
| camada 8, k = 32 | 0,954 | 0,777 | +0,2 p.p. / 0,065 nat |
| camada 17, k = 32 | 0,900 | 0,813 | −0,7 p.p. / 0,013 nat |
| camada 8, k = 64 | 0,962 | 0,822 | −0,6 p.p. / 0,052 nat |
| camada 17, k = 64 | 0,917 | 0,833 | +0,6 p.p. / 0,006 nat |

- Como escala: trocar a ativação pela média dá KL ≈ 0,4. **Os SAEs são fiéis ao comportamento do modelo.** A diferença de acurácia (±1 p.p.) está dentro do ruído de 900 decisões.
- **Tokens massivos:** na camada 17, alguns tokens (o `[CLS]`, os `[SEP]` e ~5% dos tokens comuns) têm norma até 90 vezes a mediana. Sem tratamento, a FVE fica ~0,9997 "de graça". O SAE foi treinado só nos tokens não massivos. Esses tokens ficam **fora** do que as features enxergam. (Emenda 1.)
- Um sinalizador pré-declarado disparou na camada 8: a FVE só nos marcadores de opção é bem menor que a geral. A variante com 20% de domínio melhorou (0,861) sem custo no texto genérico.

---

## 6. Features que acompanham o erro (etapa 5)

**Método (estilo SAEfarer, arXiv:2609.21142, adaptado):** para cada decisão, a ativação de cada feature no marcador da opção prevista, no marcador da opção correta e (máximo) nos tokens do estado; teste exato de Fisher com correção de Benjamini-Hochberg, suporte mínimo de 20 decisões.

**O achado mais importante e mais fácil de errar.** O ranking ingênuo por "taxa de erro quando a feature ativa" é **dominado pela opção que o Laya escolheu**: várias das melhores features só dizem "este é o marcador da opção 3", e o Laya erra sempre que escolhe essa opção. Por isso usamos o teste de
Cochran-Mantel-Haenszel (CMH), que controla pergunta e identidade da opção:

| camada 8, descoberta | features com suporte | significativas no teste ingênuo | significativas no CMH (mais erro) |
|---|---|---|---|
| marcador da opção prevista | 543 | 364 | **9** |
| marcador da opção correta | 625 | 476 | 68 |
| tokens do estado | 10.213 | 3.226 | 324 |

- **O que sobrevive ao CMH replica em direção** (as 20 features de maior |z| têm 75–100% de concordância de sinal entre descoberta e dev/teste), **mas é fraco** e, nos exemplos, em geral lexical (tokens como ` river`, `_`, ` of`, ` over`). Há features duplicadas e features com poucas ativações e razões de chances enormes (instáveis).
- **Nenhuma feature foi rotulada**: os cartões (`feature_cards/`) têm o campo "Interpretação" em branco. Quem escrever o trabalho não deve inventar rótulos que os exemplos não sustentem.
- A escolha dos estratos do CMH por visão foi feita **depois de ver contagens agregadas** (sem escolher feature alguma): é análise exploratória e deve ser dita assim (Emenda 3).
- **Atribuição por gradiente** (efeito linear de primeira ordem de cada feature sobre a diferença de logit entre a resposta correta e a escolhida): explica **85–87%** do efeito total na camada 8, mas só **26–31%** na camada 17.

---

## 7. Ablação causal (a parte "causal", com cuidados)

**Teste:** zerar um conjunto de features no marcador da opção escolhida (variante principal) e medir a mudança nos logits do Laya. Listas escolhidas só na descoberta; dev para análise, teste para replicar; comparação com 50 conjuntos aleatórios pareados por frequência de ativação.

**Zerar as 20 features de maior atribuição negativa (K = 20), dev (teste entre parênteses):**

| camada | efeito na margem | % da margem dos erros | erros corrigidos | decisões certas quebradas | acurácia total |
|---|---|---|---|---|---|
| 8 | +0,040 (+0,033) | 1,9% | 0,8% | 2,4% | −0,2 (+0,2) p.p. |
| 17 | +0,421 (+0,433) | 20% | 7,6% | 17,7% | **−0,7 (−1,4) p.p.** |

- **Distinguível e replicado:** os efeitos ficam muito acima do nulo (p empírico no mínimo possível, 0,0196), o IC95% exclui zero, o controle positivo tem o sinal certo e a previsão de primeira ordem acerta o efeito de features individuais (Spearman 0,90–0,93).
- **Mas não relevante:** na camada 8 o efeito é ~2% da margem (viés distribuído em muitas features); na camada 17 é maior, mas **a acurácia total cai**: a intervenção quebra quase tantas decisões certas quanto conserta. Leitura consistente (não verificada): essas features codificam "a opção escolhida" em geral, não "o erro".
- Zerar em todos os tokens (variante de controle) tem efeito grande, e às vezes melhora a acurácia total, mas quebra 19–38% das decisões certas; **não deve ser apresentado como mecanismo isolado.**
- **Dois pontos de método a declarar:** (i) o desenho inicial da ablação estava errado (zerava também o marcador da opção correta e o controle positivo saía invertido); foi corrigido por emenda antes de rodar dev e teste. (ii) O limiar de "relevante" (≥10% da margem ou ≥1% dos erros corrigidos, com acurácia total sem queda) foi escrito **depois** de ver que o efeito na descoberta era de ~1%: não é neutro.

---

## 8. Rotação das opções: o viés é de posição ou de conteúdo? (a ponte com o viés posicional)

**O que foi feito:** cada decisão foi mostrada ao Laya multilingual nas k rotações cíclicas da ordem das opções (o gabarito não muda). Como cada opção passa por cada posição, a média separa **preferência por posição** de **preferência por conteúdo**. Treino e teste (números do teste; o treino repete o padrão). Fonte: `REPORT.md` §8.

| tipo de pergunta | desvio por posição em relação ao uniforme (teste) | leitura |
|---|---|---|
| `choice` (4 e 5 opções) | no máximo 3,2 p.p. (posição 2, 4 opções) | **abaixo do limiar pré-fixado de 5 p.p.**: o viés é de **conteúdo** |
| `noul` (sim/não) | ±2,1 p.p. | pequeno |
| `score` (4 opções) | primeira opção −18,2 p.p. (6,8% contra 25%); última +10,9 p.p. | **viés posicional grande** |

- A escolha muda com a rotação em 44% (4 opções) e 30% (5 opções) das decisões de `choice` e em 82–91% das de `score`; em `choice`, 56–70% escolhem sempre a mesma *opção* (conteúdo). O preferir "o índice 3" da §4 é preferir o conteúdo que ficava no índice 3.
- **Acurácia com a ordem balanceada:** média das rotações 0,334 (IC95% 0,317–0,353) contra 0,349 na ordem original e 0,318 de acaso esperado. **O IC inclui o acaso:** o pouco acima do acaso que o Laya tinha depende em parte da ordem original das opções. Não testei a diferença pareada.
- **O probe (§3) sob rotação** (probe treinado só na ordem original, decisões com k ≥ 4, acaso ≈ 0,24): `enc_08` 0,50 → 0,41 → 0,34 → **0,27**; `enc_17` 0,52 → 0,46 → 0,42 → **0,42**; Laya 0,29 / 0,30 / 0,25 / 0,24. **A hipótese "o probe é mais estável que o Laya" foi desmentida** (amplitude do probe 0,23 e 0,10 contra 0,05 do Laya). A camada 8 depende da ordem; a 17 guarda conteúdo (continua bem acima do acaso), mas também usa a posição.
- **Onde a posição vive:** η² (parcela da variância do vetor do marcador) de posição é 0,96 nas camadas 1–2 e 0,91 na 4, e cai para 0,13 na 8; o de conteúdo vai de 0,005 na camada 1 a 0,78 na 8 e decai a 0,42 no fim (figura `rotation/eta_by_layer.png`). A troca de posição por conteúdo acontece entre as camadas 4 e 6.
- **As features da ablação** (lista das 20 de maior atribuição negativa) são mais de conteúdo que o acaso nas duas camadas; em `enc_17` são também mais posicionais que o acaso. Entre todas as features com suporte, só 12% (camada 8) e 7% (camada 17) têm mais posição que conteúdo.
- **Limites:** rotações cíclicas não esgotam as k! permutações; só o multilíngue foi rotacionado (o ajustado em inglês não); em `score` os rótulos "level i" ficam no texto, então é um tipo de pergunta diferente dos demais.

## 9. Estabilidade entre sementes do SAE

Segunda semente (mesmos hiperparâmetros). Fonte: `REPORT.md` §9.
- **Qualidade equivalente:** FVE no dev 0,954 contra 0,954 (camada 8) e 0,899 contra 0,899 (camada 17).
- **Só ~50% das features têm correspondente na outra semente** (48,6% e 52,5%, pelo cosseno do decoder ≥ 0,7). A expectativa que eu tinha escrito (> 80%) foi desmentida; parte pode ser *feature splitting* (não verificado).
- **A lista das 20 features da ablação não é estável:** 15% (camada 8) e 35% (camada 17) delas reaparecem entre as 100 primeiras da outra semente; nenhum índice coincide.
- **O efeito agregado da ablação se repete com a semente 1** nas duas camadas (mesmo sinal, acima do nulo, mesma leitura: camada 8 desprezível; camada 17 grande mas não específico; tamanho em `enc_17` de 20% a 29% da margem).
- **Implicação para o texto:** descrever os resultados por feature como "candidatas desta semente", nunca como "a feature X"; o resultado agregado (um conjunto de features no marcador escolhido) é o que se reproduz.

---

## 10. Par em inglês: Laya base × ajustado ao benchmark (Parte C)

- **Entradas idênticas** nos dois modelos (mesmos tokens e papéis); a configuração comum de comprimento (1024/256) não distorce o base (acurácia 0,3625 contra 0,3630 com a configuração própria).
- **Acurácia:** base 0,363 no teste; **ajustado 0,7685** (o README do modelo diz 0,766). No treino e no dev o ajustado acerta ~85%, porque **foi treinado no benchmark**: só o teste vale para ele.
- **Probe no base em inglês:** melhor camada `enc_17` (0,647 no dev, 0,621 no teste, contra prior 0,483 e o próprio base 0,363).
- **SAE:** o SAE do base lê mal o ajustado (FVE 0,774 contra 0,912); a cópia ajustada nas ativações do ajustado recupera 0,915, e 91% das features ficam pareadas (cosseno mediano 0,94).
- **Features que mudam entre base e ajustado:** 915 têm intervalo de confiança do efeito fora de zero na descoberta (646 no teste), as maiores iguais nas duas passadas (então não é artefato do SAE), e as 50 maiores têm o mesmo sinal no teste. Isso mede que a ativação delas no marcador da opção correta, em relação às erradas, muda com o ajuste; **não diz o que elas representam.**
- **Marcadores massivos no base em inglês:** só na opção `true` das perguntas sim/não; quando esse marcador é massivo, o base responde `true` em 90–92% dos casos (contra 54–55% quando não é). No ajustado a relação some. É associação dentro de um tipo de pergunta, não causa.

---

## 11. O que se pode e o que não se pode dizer

**Pode dizer (com a evidência acima):**
- A resposta é legível linearmente nas camadas intermediárias do Laya, muito além do que o modelo entrega (57% contra 35%), e a cabeça de decisão perde parte disso.
- O Laya tem um viés de prior forte e replicável (foge da classe mais comum; responde `true` demais), e **em `choice` ele é de conteúdo, não de posição** (§8). Em `score` há viés posicional grande (quase nunca escolhe a primeira opção).
- A posição é codificada nas camadas 1 a 4 do encoder e o conteúdo da opção a partir da 5 (§8).
- SAEs de 16× com k = 32 reconstroem bem essas camadas e são fiéis ao comportamento.
- O erro **não** se concentra em poucas features: ablar até 100 features no marcador escolhido move a margem pouco na camada 8, e na camada 17 desloca a margem mas derruba decisões certas junto.
- O ranking ingênuo de features por erro é enganoso porque mede a opção escolhida; é preciso estratificar.

**Não pode dizer:**
- "Encontramos as features do erro" ou "features que causam o viés". O efeito causal medido é pequeno ou não específico.
- Qualquer rótulo para uma feature (nenhuma foi interpretada).
- Que o viés de `choice` é **posicional**: o teste de rotação o descarta (máximo 3,2 p.p.). Só em `score` há viés posicional grande.
- Que o Laya "acerta acima do acaso" quando a ordem das opções é balanceada: o IC95% da acurácia (0,317–0,353) inclui o acaso (0,318).
- Que o 0,574 do probe é "conteúdo puro": medido só na ordem original; com rotação o da camada 17 cai para ~0,42 e o da camada 8 para ~0,27.
- Que uma feature específica "é" a do erro: o ranking de features depende da semente do SAE (§9).
- Que o fine-tuning "corrige" features específicas: a comparação base × ajustado é descritiva e o ajustado foi treinado no benchmark.
- Resultados de acurácia do modelo ajustado fora do teste.

**Limitações para a seção correspondente do trabalho:** tudo é correlacional exceto a fidelidade a jusante e a ablação; duas sementes por SAE (a segunda só para estabilidade e ablação); a ablação só zerou features já ranqueadas por atribuição de primeira ordem; os tokens massivos ficam fora do SAE; os limiares de sinalização e de relevância são escolhas nossas, sem base prévia; o prior de 0,461 do enunciado não foi reproduzido.

---

## 12. Onde está cada coisa

| o quê | onde |
|---|---|
| Relatório técnico completo (todas as tabelas) | `results/sae_features/REPORT.md`, `REPORT_tables.md` |
| Critérios fixados antes de rodar e emendas | `results/sae_features/PROTOCOLO.md` |
| Probes por camada | `results/probe_typed/` (`REPORT.md`, `probe_by_layer.csv`, `per_decision.parquet`, `acc_by_layer.png`) |
| Qualidade dos SAEs | `results/sae_features/sae_quality.csv` e `sae_quality_flags_*.json` |
| Cartões de features | `results/sae_features/feature_cards/` (campo de interpretação em branco) |
| Ablação | `results/sae_features/ablation/<SAE>/{dev,test}/summary.json` |
| Rotação das opções (comportamento, probe, η²) | `results/sae_features/rotation/` (`behaviour.json`, `probe_stability.json`, `eta_by_layer.csv`, `feature_eta.json`, 2 figuras) |
| Estabilidade entre sementes | `results/sae_features/seed_stability/`, `results/sae_features/ablation/*_s1/` |
| Par em inglês | `results/sae_features/part_c/`, `part_c_sae_ft/` |
| Pesos dos 10 SAEs | pacote `saes_laya.zip` (com `LEIA-ME.md` e `SHA256SUMS.txt`); também em `checkpoints/sae/` **no Mac do Vinícius e no cluster (não estão no git)** |
| Estatísticas por feature (parquets), atribuições | no cluster e no Mac (não estão no git, ~270 MB) |
| Código | `run_*.py`, `src/`, `scripts/sauron/`; testes em `tests/` (`uv run pytest`) |

**Para reproduzir:** `uv sync` (Python 3.14, torch 2.15, `uv.lock` versionado) e os scripts `run_*.py` na ordem listada em `.claude/CLAUDE.md` (esse arquivo não é versionado; a ordem também aparece no `REPORT.md` §9). Os jobs rodaram no cluster Sauron (GPU B200); os IDs dos jobs estão em `REPORT.md` §9.

**Figuras (todas em inglês):** `results/probe_typed/acc_by_layer_en.png` (probe por camada; a `acc_by_layer.png` está em português), `results/sae_features/rotation/eta_by_layer.png` (posição × conteúdo por camada) e `results/sae_features/rotation/position_preference_choice.png` (preferência por posição em `choice` e `score`).

---

## 13. Glossário rápido

- **Probe linear:** classificador linear treinado sobre ativações congeladas de uma camada; mede o que está disponível, não o que o modelo usa.
- **`[MASK]` de opção:** o token especial que o Laya coloca antes de cada opção; é de onde ele tira a nota daquela opção.
- **SAE (autoencoder esparso):** rede que decompõe ativações em um grande número de "features" de ativação esparsa.
- **FVE:** fração da variância das ativações que a reconstrução do SAE explica.
- **KL (nat):** distância entre a saída do modelo original e a do modelo com a ativação trocada pela reconstrução.
- **CMH (Cochran-Mantel-Haenszel):** teste que combina tabelas 2×2 por estrato; aqui controla pergunta e identidade da opção.
- **Atribuição por gradiente:** efeito linear de primeira ordem de uma feature sobre a diferença de logit entre a resposta correta e a escolhida.
- **Ablação:** zerar features e observar o efeito no modelo; é o que dá caráter causal.
- **Margem:** logit da resposta correta menos o logit da resposta que o modelo escolheu sem intervenção.
