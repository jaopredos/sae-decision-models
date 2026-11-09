# REPORT.md — E0: Diagnóstico comportamental de vieses

## Fase A

**Protocolo**: 300 decisões por benchmark (seed fixa 42), estratificadas por classe
(Banking20/Newsgroups) ou por (workflow, tipo de pergunta) (Typed Decisions). Ordens testadas:
todas as rotações cíclicas (k=20 em Banking20/Newsgroups) ou todas as permutações (k≤4 em
Typed Decisions, ver `src/perturbations.py:generate_orders`). IC 95% por bootstrap (1000
reamostragens por exemplo). Dados brutos em `logs/phaseA_*.jsonl`, tabela agregada em
`results/phaseA_table.csv`, figuras em `figures/acc_by_gold_position.png` e
`figures/flip_rate_by_dataset.png`.

### A1 — Sanidade: Laya English vs. multilingual contra o paper Runoff

| Modelo | Dataset | Acc (nosso) | Acc (paper Laya/Native) | Δ | Flip (nosso, todas as rotações) | Flip (paper, 1 shuffle) |
|---|---|---|---|---|---|---|
| Laya English | Banking20 | 66.7 [61.0, 72.0] | 60.3 | +6.4 | 40.7 | 29.3 |
| Laya English | Newsgroups | 41.3 [35.3, 47.0] | 48.0 | −6.7 | 64.3 | 27.7 |
| Laya English | Typed Decisions | 40.0 [34.7, 45.7] | 31.0 | +9.0 | 69.5 | 26.7 |
| Laya multilingual | Banking20 | 61.0 [55.3, 66.7] | 60.3 | **+0.7** | 44.7 | 29.3 |
| Laya multilingual | Newsgroups | 28.3 [23.3, 33.3] | 48.0 | −19.7 | 68.3 | 27.7 |
| Laya multilingual | Typed Decisions | 33.0 [28.0, 38.0] | 31.0 | **+2.0** | 79.5 | 26.7 |

**Qual checkpoint o paper usou?** Não dá para afirmar com certeza (o paper não distingue
English/multilingual — ver NOTES.md). Mas o padrão é informativo: no **multilingual**, a
acurácia em Banking20 e Typed Decisions fica a menos de 2 pontos percentuais do número do
paper (0.7 e 2.0 pp), enquanto em Newsgroups ela despenca quase 20 pp abaixo. No **English**,
nenhum dataset chega tão perto, mas também nenhum diverge tanto — erro médio absoluto ~7.3 pp
em ambos os checkpoints, mas com perfis de erro opostos. A hipótese mais provável: o paper usou
(ou algo mais próximo de) o checkpoint **multilingual** para tarefas de classificação curta
(Banking20, Typed Decisions), e a queda em Newsgroups é um artefato da nossa aproximação do
dataset (posts de Newsgroups são textos técnicos longos em inglês, truncados em 3000
caracteres — ver NOTES.md — e as 20 categorias/descrições são as nossas, não as do paper, que
não publicou a lista). Não é possível descartar que o paper tenha usado o checkpoint English;
sinalizamos isso como uma divergência não resolvida — ver NOTES.md para o que falta para
reproduzir exatamente.

**Flip rate**: nosso protocolo (todas as k rotações, k=20) é estruturalmente mais exigente que
o do paper (1 shuffle por pergunta), então os números não são diretamente comparáveis em
magnitude absoluta — só confirmam, de forma mais rigorosa, que existe viés posicional
substancial nos dois checkpoints (bem acima do que uma comparação de 2 ordens revelaria).

### A2 — Flip rate e acurácia por posição (Laya multilingual × Julia-1)

| Modelo | Dataset | Acc | Flip rate |
|---|---|---|---|
| Laya multilingual | Banking20 | 61.0 [55.3, 66.7] | 44.7 [39.3, 50.7] |
| Laya multilingual | Newsgroups | 28.3 [23.3, 33.3] | 68.3 [63.0, 73.7] |
| Laya multilingual | Typed Decisions | 33.0 [28.0, 38.0] | 79.5 [74.0, 85.0] |
| Julia-1 | Banking20 | 84.3 [80.0, 88.0] | 29.7 [24.7, 34.7] |
| Julia-1 | Newsgroups | 31.3 [26.0, 36.7] | 78.0 [73.3, 82.7] |
| Julia-1 | Typed Decisions | 74.0 [69.0, 78.7] | 72.0 [65.5, 77.5] |

**O Julia-1 tem viés posicional? Mais ou menos que o Laya?** Depende do benchmark — não é uma
resposta única:
- **Banking20**: Julia-1 tem bem menos flip (29.7% vs. 44.7%) *e* muito mais acurácia (84.3%
  vs. 61.0%) — aqui o Julia-1 é claramente mais robusto à posição.
- **Newsgroups**: Julia-1 tem o **maior** flip rate dos três checkpoints (78.0%), apesar de
  acurácia comparável à do Laya multilingual (31.3% vs. 28.3%) — aqui o Julia-1 não é mais
  robusto, é tão ou mais sensível à posição.
- **Typed Decisions**: flip comparável entre os dois (72.0% Julia-1 vs. 79.5% Laya
  multilingual), mas o Julia-1 acerta muito mais (74.0% vs. 33.0%).

`figures/acc_by_gold_position.png` mostra o padrão por trás disso: em Banking20, a acurácia do
Laya (ambos os checkpoints) **sobe com a posição** (pior quando a opção correta vem primeiro,
melhor perto do fim) — viés de recência claro. Em Newsgroups, o Laya English tem um pico
isolado na posição 0 (57% vs. ~40% nas demais) — viés de primazia nesse checkpoint específico.
O Julia-1 oscila mais e tem baseline mais alto em Banking20/Typed Decisions, mas não é imune à
posição: em Typed Decisions ele também despenca na última posição testada (k=5, posição 4).

Em suma: **os dois modelos têm viés posicional mensurável em todos os três benchmarks**; o
Julia-1 não é uniformemente "menos enviesado" — ele é mais robusto (e mais acurado) em
Banking20, mas tão ou mais sensível à posição em Newsgroups. A pergunta "a posição, o id ou a
descrição" explicam o viés, e se corrigir prior/posição recupera acurácia, ficam para a Fase B
(B2/B4), assim como a análise de transferência entre domínios (B5) e a decomposição completa
por workflow/tipo de pergunta em Typed Decisions.

---

## Fase B

**Protocolo**: mesmos 300 exemplos por benchmark (seed 42, estratificados) da Fase A. B1–B6
rodados nos dois checkpoints principais (Laya multilingual, Julia-1); dados brutos em
`logs/phaseB_*.jsonl`, tabelas em `results/phaseB_*.csv`, figuras em `figures/b*.png`. Um bug
de agregação foi encontrado e corrigido durante a revisão: a primeira passada de
`analysis_phaseB.b6_analysis()` leu 0 registros dos logs de B6 (condição de corrida/E-S
transitória na leitura dos `.jsonl` recém-fechados pelo subprocesso do Julia-1, não reproduzida
em re-execuções isoladas da mesma função); todas as tabelas de Fase B foram regeneradas via
`--aggregate-only` a partir dos logs brutos (que sempre estiveram corretos, 100/100 registros)
e conferidas byte-a-byte contra uma segunda regeneração idêntica antes de serem usadas abaixo.

### B1 — Sensibilidade ao número de opções (k-sweep)

| Modelo | Dataset | k=2 | k=4 | k=8 | k=16 | k=20 |
|---|---|---|---|---|---|---|
| | | Acc / Flip | Acc / Flip | Acc / Flip | Acc / Flip | Acc / Flip |
| Laya multilingual | Banking20 | 93.2 / 1.7 | 84.6 / 16.0 | 76.8 / 16.7 | 66.0 / 38.3 | 62.1 / 43.0 |
| Laya multilingual | Newsgroups | 81.5 / 7.7 | 61.7 / 42.7 | 43.6 / 56.0 | 28.0 / 73.7 | 24.2 / 77.0 |
| Julia-1 | Banking20 | 98.0 / 1.3 | 97.6 / 7.0 | 94.1 / 11.7 | 86.7 / 27.0 | 84.2 / 32.3 |
| Julia-1 | Newsgroups | 76.5 / 16.3 | 65.2 / 59.0 | 52.9 / 53.7 | 39.2 / 77.3 | 33.8 / 81.7 |

(`figures/b1_flip_acc_vs_k.png`). A degradação é monotônica nos dois modelos e nos dois
benchmarks: mais opções → menos acurácia e mais flip, como esperado (mais chances de troca de
ordem, tarefa estruturalmente mais difícil). O Julia-1 começa e termina acima do Laya
multilingual em Banking20 (84.2% vs. 62.1% em k=20), mas a vantagem erode: em k=2 a diferença é
de 4.8 pp, em k=20 é de 22 pp — ou seja, o Julia-1 degrada proporcionalmente mais devagar nesse
benchmark. Em Newsgroups a ordem se inverte a partir de k=4 (Laya passa a ter acc. comparável
ou levemente melhor que o Julia-1, e o Julia-1 tem o flip mais alto do conjunto em quase todo o
sweep) — confirma o padrão já visto na Fase A de que o Julia-1 não é uniformemente mais robusto.

Um ponto que não aparecia na Fase A: `max_position_deviation_from_uniform_pct` (coluna de
`phaseB_b1.csv`) mede o quanto a distribuição de posições **escolhidas** (com conteúdo real)
se afasta do uniforme — e esse valor fica sempre abaixo de ~1.6 pp, mesmo em k=20 com flip rate
de 40-80%. Ou seja: **flip rate alto não implica viés posicional agregado consistente quando
há conteúdo real** — o modelo troca de resposta ao reordenar (instabilidade por exemplo), mas
qual posição "ganha" varia de exemplo para exemplo, sem uma posição fixa preferida no agregado.
Isso contrasta fortemente com o regime sem conteúdo de B3 (abaixo), onde a preferência
posicional é grande e consistente — sinal de que o viés posicional "puro" do modelo só domina
quando não há conteúdo informativo para disputar com ele.

### B2 — Desacoplando posição, id e descrição

| Modelo | Dataset | Acc original | Acc com ids renomeados (`opt_N`) | Segue descrição (desacoplado) | Segue id/slot original (desacoplado) |
|---|---|---|---|---|---|
| Laya multilingual | Banking20 | 61.3 | 71.7 | 9.2% | 65.8% |
| Julia-1 | Banking20 | 84.1 | 85.8 | 89.2% | 5.0% |

Newsgroups e Typed Decisions não geraram registros na condição "desacoplada" (`n_decoupled=0`
nas duas linhas correspondentes de `phaseB_b2.csv`): em Newsgroups o id **é** o nome da
categoria (idêntico à descrição mostrada), e em Typed Decisions os ids são literais
`true`/`false`/nomes de critério já idênticos ao texto — não há o que desacoplar nesses dois
benchmarks por construção dos próprios dados (documentado em `src/perturbations.py`). Banking20
é o único dos três onde id (`wrong_exchange_rate_for_cash_withdrawal`) e descrição em
linguagem natural são objetos distintos, então é o único onde B2 isola algo.

Dois achados em Banking20: (1) **renomear os ids para rótulos neutros (`opt_1`, `opt_2`, ...)
melhora a acurácia dos dois modelos** (Laya +10.4 pp, Julia-1 +1.7 pp) — os ids "semânticos"
originais (ex. `wrong_amount_of_cash_received`) parecem introduzir ruído/viés de prior lexical
que atrapalha mais do que ajuda, achado que conversa diretamente com B3 abaixo. (2) No teste de
desacoplamento (descrição de um id associada ao texto apresentado de outro), os dois modelos
têm **mecanismos opostos**: o Julia-1 segue a descrição em 89.2% dos casos (decide pelo
conteúdo, ignorando com qual id/posição original aquele texto veio) — mecanismo
predominantemente orientado a conteúdo; o Laya multilingual segue o id/posição original em
65.8% dos casos e a descrição em apenas 9.2% — mecanismo majoritariamente **não** orientado ao
conteúdo apresentado (resto dos casos cai numa terceira opção, nem id original nem a nova
descrição). Isso responde diretamente à pergunta 3 do enunciado para Banking20: no Julia-1 o
viés é dominado pela **descrição/conteúdo**; no Laya multilingual é dominado por
**id/posição**, não pela descrição.

### B3 — Prior de classe sem conteúdo (estado vazio)

Prior médio por id, Banking20 (top 3 de 20 classes, `phaseB_b3_prior_by_id.csv`; frequência
real de cada classe nos 300 exemplos = 5.0% uniforme, em `phaseB_b3_real_class_frequency.csv`
— a estratificação funcionou, então o desvio abaixo não vem dos dados):

| Modelo | Top-1 id | Top-1 | Top-2 id | Top-2 | Top-3 id | Top-3 | Soma top-3 |
|---|---|---|---|---|---|---|---|
| Laya multilingual | wrong_exchange_rate_for_cash_withdrawal | 37.6% | wrong_amount_of_cash_received | 23.2% | card_payment_wrong_exchange_rate | 21.8% | **82.6%** |
| Julia-1 | direct_debit_payment_not_recognised | 15.5% | wrong_exchange_rate_for_cash_withdrawal | 9.4% | why_verify_identity | 8.8% | 33.7% |

Em Newsgroups o padrão se repete e é ainda mais extremo no Julia-1: Laya concentra 37.3% em
`sci.electronics` sozinho; Julia-1 concentra **77.3%** em apenas 2 classes
(`comp.sys.mac.hardware` 42.6% + `soc.religion.christian` 32.4%), apesar do prior por posição
do Julia-1 em Banking20 ser mais disperso (33.7% no top-3) — ou seja, o viés de id/classe sem
conteúdo é forte nos dois modelos, mas a distribuição entre as classes "favoritas" não é
estável entre benchmarks: o Julia-1 é relativamente disperso em Banking20 e extremamente
concentrado em Newsgroups. Combinado com o B2 de Banking20 (renomear ids melhora acurácia),
isso sustenta a leitura de que boa parte do erro vem de um prior de classe aprendido sobre o
texto do id/categoria em si, independente do conteúdo da pergunta.

Prior por **posição** sem conteúdo (`figures/b3_prior_by_position.png`, estado vazio, k nativo
de cada dataset): também não-uniforme e estruturalmente diferente por modelo — Laya
multilingual favorece posições centrais em Banking20 (pico ~15% na posição 11, uniforme seria
5%) e tem primazia forte em Newsgroups (10% na posição 0, depois um segundo platô em 14-19);
Julia-1 é bimodal em Banking20 (picos perto da posição 1 e de 16-17) e tem recência quase
absoluta em Newsgroups — **nenhuma escolha registrada antes da posição 13** em 300 exemplos. Em
Typed Decisions (k≤5, permutações completas), os dois modelos convergem num padrão incomum:
quase nunca escolhem a **última** posição apresentada (Laya 0.8%, Julia-1 1.4%, ante ~20-25%
esperado uniforme), com Laya preferindo fortemente a penúltima posição (41.9%) e Julia-1
distribuindo-se quase uniformemente entre as 4 primeiras.

`phaseB_b3_noul_neutral_rate.csv`: perguntas tipo `noul` sem conteúdo (apenas critério,
estado vazio) — Laya multilingual responde "verdadeiro" com probabilidade média 0.745 e só fica
genuinamente neutro (perto de 50/50) em 11.0% dos casos; Julia-1 tem p(verdadeiro) médio mais
baixo (0.639) mas é **ainda menos neutro** (2.5%) — os dois modelos têm um prior de classe
default forte para `noul` mesmo sem nenhuma informação, com o Julia-1 sendo mais extremo/menos
incerto nesse default do que o Laya.

### B4 — Mitigações (correção de prior e rotation-averaging)

| Modelo | Dataset | Baseline (1 ordem)¹ | +correção de prior | +rotation-averaging | +ambos |
|---|---|---|---|---|---|
| Laya multilingual | Banking20 | 61.0 | 62.3 | 62.3 | **64.3** |
| Laya multilingual | Newsgroups | 28.3 | 31.7 | 28.3 | 31.7 |
| Laya multilingual | Typed Decisions | 24.0 | — ² | 20.5 | — ² |
| Julia-1 | Banking20 | 84.3 | 84.3 | 84.3 | 83.3 |
| Julia-1 | Newsgroups | 31.3 | 35.7 | 33.7 | **38.7** |
| Julia-1 | Typed Decisions | 70.5 | — ² | 65.5 | — ² |

¹ Acurácia na primeira ordem testada (`rot_0`/`perm_0`) por exemplo, excluindo `noul` — **não**
é a mesma métrica da tabela A2 (que faz a média sobre todas as ordens testadas por exemplo, com
`noul` incluído); por isso os números "baseline" aqui diferem dos de A2 para o mesmo
par modelo/dataset.
² Correção de prior exige um prior de classe por id estável entre exemplos (só existe para
Banking20/Newsgroups — ver B3); Typed Decisions não tem essa estrutura, então essa coluna fica
vazia por desenho, não por falha.

(`figures/b4_mitigations_before_after.png`). As duas mitigações **nunca pioram** o resultado em
Banking20/Newsgroups de forma relevante, e o ganho combinado é maior no Julia-1/Newsgroups
(+7.3 pp) — exatamente o par (modelo, dataset) onde B1 mostrou o pior flip rate do conjunto, o
que é consistente: rotation-averaging ajuda mais onde a instabilidade por ordem é maior. Em
Banking20 o Julia-1 já está perto do teto (84%) e a correção de prior não acrescenta nada (ele
já ignora bastante o prior de id, como B2 mostrou). Em Typed Decisions, a única mitigação
aplicável (rotation-averaging) **piora** a acurácia nos dois modelos (Laya −3.5 pp, Julia-1
−5.0 pp) — plausível porque B3 mostrou que o prior de posição em Typed Decisions é fortemente
não-uniforme por desenho da tarefa em si (penúltima posição favorecida), não um artefato a ser
cancelado por médias; forçar invariância de ordem aqui descarta sinal real, não ruído.

### B5 — Transferência entre domínios, calibração e "mecanismo único"

| Modelo | Dataset | Acc | Maj. baseline | Aleatório | Erro c/ conf.>0.9 | ECE (15 bins) |
|---|---|---|---|---|---|---|
| Laya multilingual | Banking20 | 61.0 | 5.0 | 5.0 | 33.3% | 0.338 |
| Laya multilingual | Newsgroups | 28.3 | 5.0 | 5.0 | 50.0% | 0.319 |
| Laya multilingual | Typed Decisions | 33.0 | 18.0 | 32.8 | 55.1% | 0.328 |
| Julia-1 | Banking20 | 84.3 | 5.0 | 5.0 | 8.7% | **0.081** |
| Julia-1 | Newsgroups | 31.3 | 5.0 | 5.0 | 44.9% | 0.367 |
| Julia-1 | Typed Decisions | 74.0 | 18.0 | 32.8 | 21.5% | 0.216 |

(`figures/b5_ece_by_benchmark.png`, `figures/b5_ece_by_workflow.png`). Os dois modelos batem
folgadamente o baseline de classe majoritária e o aleatório em todos os benchmarks — não há
colapso total no prior. Mas a calibração é ruim de forma generalizada: ECE entre 0.22 e 0.37 em
quase todas as células (só o Julia-1/Banking20 foge disso, com ECE de 0.08 — justamente onde
ele também tem a maior acurácia bruta). Em Typed Decisions por workflow, o Julia-1 é mais bem
calibrado que o Laya em 3 dos 4 workflows (exceção: `customer_service`), com a maior diferença
em `invoice_processing` (ECE 0.172 vs. 0.377). Por tipo de pergunta, `noul` é onde os dois
modelos mais superam o baseline de maioria (Laya 51.0% vs. 54.0% de maioria — na verdade fica
**abaixo** da maioria; Julia-1 81.0% vs. 54.0% — bem acima), e `score` é onde os dois têm a pior
taxa de erro com alta confiança (Laya 100%, Julia-1 29.3%) — nos dois modelos, quando erram em
perguntas `score` com confiança >0.9, erram **sempre** de forma mal calibrada no caso do Laya.

**Teste de mecanismo único** (`phaseB_b5_mechanism.csv`, acertos vs. erros): a entropia da
previsão é sempre maior nos erros que nos acertos (esperado — overconfidence errada é mais rara
que incerteza alta nos erros), mas o padrão de "colapso para o prior" **não é universal**:
- **Newsgroups**: forte e consistente nos dois modelos — a concordância do erro com o argmax do
  prior sem conteúdo (B3) mais que dobra nos erros vs. acertos (Laya: 12.9% → 37.2%; Julia-1:
  10.6% → 25.2%). Esse é o benchmark onde o mecanismo "erro = colapso no prior de classe" mais
  se sustenta nos dados.
- **Banking20**: só se sustenta no Laya (concordância com prior: 2.2% → 7.7% dos acertos para
  erros); no Julia-1 a concordância com o prior **cai** nos erros (5.9% → 2.1%) e a entropia
  sobe forte (0.18 → 0.68) — os erros do Julia-1 em Banking20 parecem mais "incerteza genuína"
  (ele já ignora bastante o prior de id, como visto em B2/B4) do que colapso previsível.
- **Typed Decisions**: sem evidência de colapso para posição preferida nos dois modelos (Laya
  20.8% → 20.4%, Julia-1 19.9% → 15.3% — se algo, cai levemente); a concordância com prior de id
  não se aplica aqui (sem estrutura de classe estável, mesma razão de B4).

### B6 — Sensibilidade à formulação noul (literal vs. descritivo)

| Modelo | n | Acc. descrição original | Acc. literal `true`/`false` | Queda |
|---|---|---|---|---|
| Laya multilingual | 100 | 51.0% | 47.0% | −4.0 pp |
| Julia-1 | 100 | 81.0% | 64.0% | −17.0 pp |

Referência do card oficial do Julia-1 (mesmo protocolo CPU/FP32, 600 exemplos — ver NOTES.md):
**483/600 (80.5%) → 391/600 (65.2%)**, queda de 15.3 pp. Nossa amostra de 100 exemplos do
Julia-1 reproduz essa queda de forma notavelmente próxima (81.0% → 64.0%, −17.0 pp vs. −15.3 pp
do card, dentro do ruído esperado para n=100 vs. n=600) — um segundo ponto de validação
independente de A1 de que nosso pipeline de inferência do Julia-1 está correto. O Laya
multilingual, que não tem número oficial publicado para essa ablação, é **muito menos sensível**
à troca (−4.0 pp vs. −17.0 pp) — mas também já parte de uma acurácia baixa (51.0%) nessa fatia,
então há menos "acerto sensível ao fraseado" para perder: o Julia-1 claramente depende mais da
descrição em linguagem natural do critério do que o Laya multilingual, que já parece sub-usar
essa informação mesmo na formulação original (consistente com B2, onde o Laya ignora a
descrição em 90%+ dos casos no teste de desacoplamento).

---

## Respostas às perguntas finais

**1. O Laya multilingual reproduz a magnitude de viés relatada no paper (Runoff)?**
Parcialmente. A acurácia bate com uma proximidade notável em Banking20 (+0.7 pp) e Typed
Decisions (+2.0 pp), mas diverge muito em Newsgroups (−19.7 pp) — provavelmente por diferenças
na nossa aproximação desse dataset (ver A1/NOTES.md), já que o paper não publica a lista exata
de categorias/descrições usadas. O flip rate não é diretamente comparável em magnitude (nosso
protocolo testa todas as rotações, o paper testa 1 shuffle), mas confirma qualitativamente a
existência de viés posicional substancial nos dois checkpoints testados. B6 fornece uma segunda
confirmação independente e mais direta: a queda 483→391/600 do card oficial do Julia-1 é
reproduzida com grande fidelidade na nossa amostra do Julia-1, o que aumenta a confiança de que
nosso pipeline de inferência está correto onde há número de referência exato disponível.

**2. O Julia-1 tem viés posicional — mais ou menos que o Laya?**
Não é uma resposta única entre benchmarks (ver A2 e B1). Em Banking20 o Julia-1 é claramente
mais robusto (flip e acurácia melhores em todo o k-sweep, embora a vantagem relativa encolha
conforme k cresce). Em Newsgroups o Julia-1 **não** é mais robusto — tem o maior flip rate do
conjunto (78-82%) com acurácia comparável à do Laya. Em Typed Decisions os dois têm flip
parecido, mas o Julia-1 acerta muito mais. B3 mostra que os dois modelos têm prior de
posição/id fortemente não-uniforme sem conteúdo — a diferença entre eles não é "um tem viés e o
outro não", é que a *forma* do viés (onde ele concentra, e o quanto o conteúdo real consegue
sobrepor esse viés) difere por benchmark e por modelo.

**3. O viés vem da posição, do id, ou da descrição?**
Depende do modelo, isolado em B2 (único benchmark — Banking20 — onde id e descrição são
separáveis): no **Julia-1** o mecanismo é majoritariamente orientado ao **conteúdo/descrição**
(89.2% de concordância no teste de desacoplamento) — ele decide pelo que está escrito, não por
qual id/posição originalmente carregava aquele texto. No **Laya multilingual** é o oposto: 65.8%
de concordância com id/posição original, só 9.2% com a descrição — o texto da opção pesa pouco,
o que "estava ali antes" pesa muito. B3 reforça isso: Laya tem o prior de id mais concentrado
em 1-2 classes (Banking20: top-1 sozinho = 37.6%), compatível com um modelo que "decora" certos
ids/posições independente do conteúdo.

**4. Corrigir o prior/posição recupera acurácia em cada modelo?**
Sim, mas de forma desigual e nunca de forma dramática (B4). Ganhos reais existem (Julia-1 em
Newsgroups: +7.3 pp combinando as duas correções; Laya em Banking20: +3.3 pp), mas no melhor
caso de prior puro o ganho fica abaixo de 5 pp, e em Typed Decisions a correção por
rotation-averaging **piora** os dois modelos — porque lá o desvio de uniformidade em B3 reflete
estrutura real da tarefa (penúltima posição genuinamente mais informativa), não ruído de ordem.
Onde o modelo já ignora bastante o prior (Julia-1 em Banking20, visto em B2), a correção
também não ajuda — não há viés de prior sobrando para corrigir.

**5. Os erros (por benchmark/workflow) colapsam para o prior/posição?**
Só de forma consistente em **Newsgroups**, nos dois modelos (B5): a concordância do erro com o
argmax do prior sem conteúdo mais que dobra em relação aos acertos. Em **Banking20** isso só se
sustenta no Laya; no Julia-1 os erros parecem mais "incerteza genuína" (entropia alta, sem
colapso num id específico) — consistente com B2/B4 mostrando que o Julia-1 já ignora boa parte
do prior de id nesse benchmark. Em **Typed Decisions**, nenhum dos dois modelos mostra colapso
para a posição preferida nos erros. Conclusão geral: existe um mecanismo de "erro = default de
prior" real e mensurável, mas benchmark-dependente — não é a explicação única nem universal
para todo erro dos dois modelos.
