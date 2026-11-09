# NOTES.md — Diagnóstico comportamental E0

Investigação do código de inferência dos dois pacotes, feita lendo o código-fonte instalado
localmente (não documentação externa). Citações `arquivo:linha` referem-se aos pacotes
instalados em `.venv/lib/python3.14/site-packages/laya/` (laya 0.4.30) e
`data/julia1_repo/julia/` (snapshot de `SupersonicLabs/Julia-1`).

## Ambiente de execução

- Máquina: MacBook Air M5, 24 GB RAM, **sem GPU CUDA** (`torch.cuda.is_available() == False`),
  com MPS disponível.
- **Conflito de dependências entre os dois pacotes**: `laya` (via `vegaml`) exige
  `transformers>=5.17`; o pacote da Julia-1 fixa `transformers>=5.0,<5.1` porque
  `julia/router/encoder.py` monkeypatcha um método privado do `ModernBertModel`
  (`_update_attention_mask`) que muda de assinatura em `transformers>=5.1` — confirmado
  empiricamente (`AttributeError` com transformers 5.22.2).
  **Consequência arquitetural**: os dois modelos não podem ser importados no mesmo processo
  Python. Solução adotada: dois venvs —
  - `.venv` (transformers 5.22.2): laya, orquestração, datasets, métricas, plots.
  - `.venv_julia` (transformers 5.0.0, isolado): só para o encoder/head da Julia-1.
  O wrapper comum (`src/models.py`) implementa a Julia-1 como cliente de um subprocesso
  persistente (`src/julia_worker.py`) rodando em `.venv_julia`, comunicando por JSON-lines via
  stdin/stdout. Do ponto de vista da API (`DecisionModel.predict(state, question_spec) ->
  {choice, probs}`), os dois modelos são intercambiáveis; a diferença de processo é um detalhe
  de implementação.

## Laya (`convaiinnovations/laya`)

### Carregamento / Router

- `laya.load(model_id_or_path="convaiinnovations/laya", subfolder="multilingual", ...)` →
  `Agent.__init__` (agent.py:548). Como `subfolder` é passado explicitamente, o bloco de
  `load()` que resolve aliases via `Router` é **pulado**
  (`if subfolder is None and not _is_local_checkpoint_arg(...)`, agent.py:2242) — nenhum
  roteamento por idioma ocorre; o checkpoint fica fixo em `multilingual` para toda chamada
  feita nesse `Agent`.
- O comportamento "texto em inglês vai para o checkpoint English" só existe em
  `Router.predict` (router.py:1463-1609), um objeto **separado** usado quando se carrega sem
  `subfolder` explícito. **Para forçar sempre multilingual**: usar `laya.load(...,
  subfolder="multilingual")` e chamar `agent.predict(...)` diretamente (nunca passar pelo
  `Router`/aliases).
- Download via `huggingface_hub.snapshot_download` restrito a
  `rl_agent_config.json`, `model.safetensors`, `tokenizer/*`, `encoder/*` dentro do subfolder
  (agent.py:649-665).
- Device: auto-detecta `cuda > mps > xpu > cpu` (agent.py:730-738) → nesta máquina, `mps`.
- `predict = system_one` (agent.py:2080) — `agent.predict(state, questions)` é literalmente
  `agent.system_one(state, questions)`.

### Serialização de entrada (common.py)

Sequência de tokens (`build_sequence`/`build_head`, common.py:170 e seguintes):

```
[CLS] "<type> question: <instructions>" [SEP] [MASK] <id0>: <desc0> [MASK] <id1>: <desc1> ... [SEP] <state tokens> [SEP]
```

- `<type>` é `choice`/`score`/`noul` como texto literal antes de "question:".
- Cada opção é renderizada `"<id>: <descrição>"` via `render_options` (common.py:128-154); a
  ordem de apresentação = ordem do dict `criteria` (ou lista `option_order` explícita dentro de
  `questions[qid]`, que permite reordenar sem reescrever o dict — **é o mecanismo usado para
  gerar as condições de rotação/permutação sem recriar o `criteria`**).
- `state` não-string é serializado com `json.dumps` (common.py:95-98).
- Truncamento: `state` é cortado silenciosamente para caber em `max_len - head_len - 1`
  (truncamento à direita, exceto listas de turnos de conversa → truncamento à esquerda). Se
  mesmo reduzindo o budget por opção os marcadores não couberem, levanta `ValueError`
  explícito (agent.py:1251). `return_truncation_stats=True` expõe estatísticas de truncamento.

### Token(s) de leitura do decision head

Cada opção recebe **um token `[MASK]` dedicado** imediatamente antes do seu texto
(`opt_ids.append([tok.mask_token_id] + opt_tokens)`, common.py:252). As posições desses
`[MASK]` (`marker_pos`) são extraídas do hidden state via `torch.gather`
(`DecisionModel.forward`, common.py:~809); um `scorer` (LayerNorm → Linear → GELU → Linear →
1 escalar) produz um logit por opção; softmax mascarado dá as probabilidades.
**Não é o `[CLS]`**: o `[CLS]` alimenta só o `act_head` (estimativa de confiança/roteamento),
não as probabilidades de opção.
→ **Para extração de ativações no SAE (fases futuras): usar os hidden states nas posições
`marker_pos` (uma por opção) e/ou a posição 0 (CLS) para o sinal de confiança.**

### Limite de contexto

- `max_len` padrão 512; `head_max_len` padrão 192 (configuráveis via `rl_agent_config.json` ou
  argumento de `predict`). Cada opção é truncada em até 48 tokens; se não couberem todas no
  `head_max_len`, o budget por opção é reduzido ainda mais (common.py:245-260).

### Dtype

- Pesos carregam em FP32 do `model.safetensors` e não são convertidos no `.to(device)`
  (agent.py:867). O "dtype" reduzido só afeta um **autocast opcional**: CUDA usa
  bf16/fp16 (default fp16); MPS usa fp16 mas só ativa quando o batch tem ≥
  `mps_amp_min_rows` linhas (padrão 5, env `LAYA_MPS_AMP_MIN_ROWS`).
  **Uma chamada `predict()` de item único (1 linha) roda em FP32 pleno mesmo em MPS/CUDA**,
  pois `dtype_for(rows=1)` retorna `torch.float32` (agent.py:1275-1283).
  → Para comparar BF16×FP32 de forma que exercite o autocast de fato, é preciso usar
  `predict_batch` com ≥5 estados (MPS) ou chamadas individuais (que já são FP32 "de fábrica").
  Testado ao vivo: `agent.device=mps`, `agent.dtype=torch.float16`.

### Formato de probabilidades de saída

`agent.predict(state, questions)` → `{"answers": {qid: {...}}, "usage": {...}}`. Por tipo
(agent.py:1409-1483, confirmado ao vivo):

- `choice`: `{"type":"choice","choice": <id>, "probabilities": {id: prob, ...} (soma 1,
  TODAS as opções), "confidence":..., "answer_confidence":...}`
- `score`: `{"type":"score","score": <valor esperado>, "probabilities": {"0":p0,...}}`
- `noul`: `{"type":"noul","noul": p(true), "confidence":...}`

Testado ao vivo: `agent.predict(state, {'q1': {...4 opções...}})['answers']['q1']['probabilities']`
devolve dict completo id→prob; invertendo a ordem das opções no dict `criteria`, a resposta e
as probabilidades seguem o id correto (não a posição) em exemplos fáceis.

### Viabilidade de acesso via `transformers` puro

Confirmada. `build_model` (common.py:871-894) usa `AutoConfig`/`AutoModel` padrão para o
encoder (ModernBERT/mmBERT, `attn_implementation="sdpa"`), carregado de `<model_dir>/encoder/`
(config HF padrão) — hidden states puros via
`agent.model.encoder(input_ids=..., attention_mask=...).last_hidden_state`. O head
customizado (2 camadas de `TransformerEncoder` + `type_emb` + `scorer` + `act_head`) não é
módulo padrão do `transformers`, mas está disponível em `laya.common.DecisionModel`, já
instanciado em `agent.model` — logits/probabilidades completas via chamada direta a
`agent.model.forward(...)` com tensores de `build_sequence`, que devem bater exatamente com
`agent.predict()` (mesmo `nn.Module`). **Teste de paridade em ~50 exemplos**: ver
`src/verify_parity.py` / resultado registrado na seção "Paridade de caminhos" abaixo.

## Julia-1 (`SupersonicLabs/Julia-1`)

### Instalação

- `huggingface_hub.snapshot_download(repo_id="SupersonicLabs/Julia-1", local_dir=...)` baixa
  46 arquivos incl. `model.safetensors` (~550 MiB FP32) e o código-fonte do pacote em `julia/`.
- Nome real do pacote PyPI: **`supersonic-julia`** (não existe `pip install julia` genérico —
  esse nome é da linguagem Julia). `pyproject.toml`: `requires-python>=3.11`,
  `torch>=2.6`, `transformers>=5.0,<5.1`, `safetensors>=0.5`, `numpy>=1.26`. CUDA é extra
  opcional (`[project.optional-dependencies] cuda = [bitsandbytes, accelerate]`).
- Instalado via `pip install -e data/julia1_repo` dentro de `.venv_julia` dedicado (ver seção
  "Ambiente de execução" acima). Também funciona sem instalar, só com `sys.path.insert`.

### Carregamento e dtype — **Julia-1 roda sem CUDA**

- `from julia import load_model; engine = load_model(checkpoint_dir, device='cpu',
  strict_encoding=True, max_length=8192, head_length=512)`. `load_model` (julia/inference.py:80)
  tem `device='cpu'` **por padrão**.
- `julia/cuda.py:configure()` só levanta erro se pedirmos `device='cuda'` sem CUDA disponível;
  CPU nunca exige CUDA. Autocast BF16 (`inference.py:57`) só liga quando `device.type ==
  'cuda'` — em CPU a inferência é sempre FP32 puro, sem autocast.
- `julia_config.json`: `"weight_dtype": "float32"` — o checkpoint distribuído já é FP32; não há
  BF16 em nenhum ponto da inferência em CPU.
- **A frase do prompt da tarefa ("CUDA exige GPU com BF16") não corresponde ao comportamento
  real do código**: CUDA é opcional, BF16 só ativa em CUDA, e o checkpoint é FP32. Testado e
  confirmado com predição real funcionando em CPU nesta máquina (sem SLURM, sem GPU):
  ```json
  {"answers": {"team": {"type": "choice", "probabilities":
    {"billing": 0.854, "shipping": 0.143, "access": 0.003}, "choice": "billing"}}}
  ```

### Serialização de entrada (julia/data.py:72-110, `sequence`)

```
[CLS] "<type> question: <instructions>" [SEP] [MASK]+desc_opt1 [MASK]+desc_opt2 ... [SEP] <state tokens> [SEP]
```

Estrutura idêntica à do Laya (mesma família mmBERT): 1 token `[MASK]` dedicado por opção
(budget 48 tokens/opção), `state` depois do segundo `[SEP]`. `head_length` default 256;
`max_length` default 8192 (`encoder/config.json: max_position_embeddings=8192`).
`strict_encoding=True` levanta exceção em truncamento em vez de truncar silenciosamente
(**usar sempre `strict_encoding=True` na avaliação**, para não mascarar truncamento de
opções/state). Limite nativo: 2–20 opções por chamada; `noul` usa exatamente 2 opções, ordem
fixa `[false, true]`.

### Token(s) de leitura do decision head (julia/model.py:36-73)

Mesmo mecanismo do Laya: hidden state no token `[MASK]` prefixado de cada opção
(`hidden.gather(1, positions)`, model.py:58-59), passado por um `scorer` (LayerNorm→Linear→
GELU→Linear→escalar); softmax mascarado sobre os scores válidos. `act_head` lê CLS + features
agregadas (prob top-1/top-2, entropia, nº opções) só para roteamento hierárquico agrupado —
irrelevante para E0 (usamos chamadas nativas com ≤20 opções, nunca o roteamento agrupado).

### Interface recomendada para probabilidades completas

- **Usar a API de perguntas nomeadas** (`engine.predict(state=..., questions={...})`,
  `julia/typed.py`) — softmax completo sem arredondamento.
- **Evitar a API legada** (`engine.predict(rows=[...])`): passa por `display_probabilities()`
  (julia/probabilities.py) que arredonda/zera probabilidades <1% e colapsa para one-hot quando
  o vencedor tem >95% e as demais <4.5% — destruiria os vetores de probabilidade completos
  exigidos para os pares contrastivos do SAE.
- Logits brutos pré-softmax: `engine.logits(rows)`.

### Limite de contexto

8.192 tokens combinados (state + head + opções) = `max_position_embeddings` do encoder
ModernBERT-small.

### Acesso via `transformers` puro

Encoder é `AutoModel`/`ModernBertModel` padrão do HF (julia/model.py:32) — hidden states puros
via `transformers` padrão. A classe completa `JuliaDecisionModel` (encoder + head 2 camadas +
type-embedding + scorer) não é um `AutoModel` registrado; precisa ser instanciada manualmente a
partir de `model.safetensors`. `julia/router/encoder.py` monkeypatcha
`ModernBertModel.forward` (`_decision_forward`) para pular cálculo não usado — é só uma
otimização de performance; o forward padrão do HF (sem monkeypatch) produz os mesmos hidden
states, mais lento. **Recomendação para extração de ativações do SAE (fases futuras): usar o
encoder sem esse monkeypatch**, para não depender de API privada instável (já quebrou entre
versões do `transformers`).

### B6 — `noul` literal vs. descritivo

Confirmado no próprio repositório (README + `metrics/typed-cpu-20260926.json` +
`scripts/reproduce_typed.py`): rodando as 2.000 decisões (400 casos) em CPU/FP32,
**426/600 choice, 542/800 score, 483/600 noul** com descrições booleanas originais; substituindo
por literais `false`/`true`: **391/600 noul**, mesmo protocolo/run. 400 das 600 perguntas
`noul` têm `criteria` descritivo nos dados de teste; a ablação troca só essas, mantendo as 200
que já eram literais. **Usar a comparação 483→391 (mesmo protocolo) para B6** — não misturar
com o número histórico "oficial" (BF16/CUDA) de 484/600, que vem de um run diferente.

## Dataset "Typed Decisions"

Localizado em `scripts/reproduce_typed.py` do próprio repo da Julia-1 (confirma a pesquisa
independente no repositório do Laya): dataset HF **`LocalLLaMA/typed-decisions`**, config
`"all"`, revisão pinada `c76749ec58bd8c3d2ea706b31c333a9059c38f90`,
arquivo `all/test-00000-of-00001.parquet` (SHA-256 documentado). Split **test = 400 casos**
(bate com BENCHMARKS.md do Laya: "400 cases, 2,000 decisions").

Estrutura de cada caso (colunas do parquet): `id`, `workflow` (um de `invoice_processing`,
`security_incidents`, `customer_service`, `agent_trace_observability`), `state` (JSON string),
`questions` (JSON: cada pergunta tem `type` ∈ {choice,score,noul}, `instructions`, `criteria`
id→descrição), `gold` (JSON: por pergunta, `label` + `probabilities`). 5 perguntas fixas por
caso × 400 casos = 2000 decisões: **600 `choice` + 800 `score` + 600 `noul`**.
Parsing exato replicado em `src/datasets.py::load_typed_decisions` seguindo a lógica de
`reproduce_typed.py` (linhas 52-68): para `noul` sem `criteria`, usa
`{'false': 'false', 'true': 'true'}`; `gold[question_id]['label']` é o rótulo correto.

**Checkpoint `laya-typed-decisions`**: segundo o próprio BENCHMARKS.md do Laya, foi avaliado
sobre o split de **treino**, não o de teste — por isso, conforme instrução do prompt, não é
usado como um dos modelos avaliados neste E0 (só os checkpoints `laya` multilingual/English e
`Julia-1` são avaliados).

## Datasets "Banking20" e "Newsgroups" — aproximação documentada

Paper fonte: **Huang, Teng, Deng, An — "Consider, Then Choose: Order-Invariant Decisions in
System One Models"**, arXiv:2610.18233 (texto completo lido em
`arxiv.org/html/2610.18233v1`). Achados:

- O paper cita Banking20 só como "a 20-intent subset of BANKING77 (Casanueva et al., 2020)" —
  **sem listar quais 20 intents nem critério de seleção**; idem para Newsgroups
  ("20 Newsgroups (Lang, 1995)", sem lista de categorias/descrições). **Nenhum repositório de
  código do próprio paper foi encontrado** — não há como reproduzir exatamente.
- Protocolo do paper: 300 perguntas em cada um de Banking20/Newsgroups (split não especificado);
  opções = "the gold label and the first K−1 of the other labels in a fixed random order";
  K testado ∈ {2,4,8,12,16,20}; **métrica "Flip" do paper = fração de perguntas cuja resposta
  muda entre a ordem original e **um único shuffle** (não todas as rotações)** — Seção 5.1,
  citação verbatim: "the flip rate under one shuffle per question".
- Tabela 9 (Apêndice D) confirma os números da tabela do prompt da tarefa (Laya/Native:
  Banking20 Acc=60.3/Flip=29.3; Newsgroups Acc=48.0/Flip=27.7; Typed Decisions
  Acc=31.0/Flip=26.7). O checkpoint "Laya" do paper é construído sobre ModernBERT-large e o
  paper **não distingue** English vs. multilingual — a dúvida sobre qual checkpoint foi usado
  permanece em aberto (ver A1 no REPORT.md).

**⚠️ Aproximação assumida e usada neste projeto** (sinalizada também no REPORT.md):
- **Banking20**: 20 classes do dataset `banking77` (HF `legacy-datasets/banking77` /
  `PolyAI-LM/banking77`) escolhidas por amostragem aleatória com seed fixa (42) dentre as 77,
  descrição de cada classe = nome do intent com underscores trocados por espaços (ex.
  `card_arrival` → "card arrival").
- **Newsgroups**: as 20 categorias nativas de `sklearn.datasets.fetch_20newsgroups`
  (já são exatamente 20 — não há amostragem aqui), descrição de cada classe = nome da categoria
  formatado (ex. `comp.sys.mac.hardware` → "computer discussion about mac hardware").
- **Métrica "Flip" do E0 é mais rigorosa que a do paper** (o E0 pede todas as rotações
  cíclicas, e todas as permutações para k≤4, não um único shuffle) — isso é uma divergência
  metodológica **deliberada** do protocolo do E0 em relação ao paper Runoff, registrada aqui
  para não ser confundida com erro na reprodução de A1.
- Em nenhum dos dois datasets o paper especifica split treino/teste; usamos o dataset completo
  disponível e amostramos 500 exemplos (ou 300 se custo computacional elevado) com seed fixa,
  estratificados por classe.

## Paridade de caminhos (pacote oficial × `transformers` puro)

**Método** (`src/verify_parity.py` para Laya, `src/verify_parity_julia.py` para Julia-1,
este sob `.venv_julia`): monkeypatch em `agent.model.forward` / `engine.model.forward` para
capturar os tensores exatos de entrada (`input_ids`, `attention_mask`, `marker_pos`,
`marker_mask`, `qtype`) e a saída oficial em ~50 chamadas reais de `agent.predict()` /
`engine.logits()`. Em seguida, carrega-se o encoder **uma segunda vez**, de forma
independente, via `AutoConfig.from_pretrained(encoder_dir)` + `AutoModel.from_config(...)`
(arquitetura pura do `transformers`, sem relação com o objeto já carregado), com os pesos do
encoder lidos diretamente do `model.safetensors` combinado (prefixo `"encoder."`) — exatamente
como `laya.common.build_model`/`julia.model.JuliaDecisionModel.from_pretrained` fazem por
baixo dos panos. Essa segunda instância do encoder processa os mesmos tensores capturados, e o
resultado passa pelo head/scorer/type_emb **já carregados** (reutilizados, não
reimplementados) para produzir uma segunda predição, sem nenhum atalho de otimização
(`_cpu_window_encoder`/`_cpu_head`/`marker_only_head`) que o caminho oficial possa usar.

**Resultado** — ambos os modelos têm o caminho `transformers` puro confirmado, com
concordância efetivamente perfeita com o pacote oficial:

| Modelo | n | max \|Δp\| | argmax concorda |
|---|---|---|---|
| Laya multilingual (`results/parity_check_laya.json`) | 50 | 0.0 | 100% |
| Julia-1 (`results/parity_check_julia.json`) | 50 | 7.45e-8 (ruído de float32) | 100% |

Confirma que os hidden states e as probabilidades finais obtidas via `transformers` puro
batem com as do pacote oficial nos dois modelos. A extração de ativações em si (quais camadas,
que granularidade) fica para a próxima fase do projeto (SAE), não faz parte do E0.

## Dtype — BF16/FP16 × FP32

**Laya** (`src/verify_dtype.py`, 100 exemplos de Banking20, `predict_batch` com 100 estados
para forçar o autocast — ver seção "Dtype" acima: chamadas de item único já são FP32 "de
fábrica"): no device desta máquina (`mps`, autocast reduzido = `float16`), a diferença máxima
de probabilidade entre o forward com autocast e o forward forçado em FP32
(`agent.amp_enabled=False`) foi **0.0044**, diferença média **0.00046**, e **0 mudanças de
escolha** em 100 exemplos (`results/dtype_check.json`). Efeito desprezível — **não foi
necessário forçar FP32** no pipeline principal; de todo modo, como `runner.py` sempre chama
`model.predict()` por exemplo único (nunca `predict_batch`), cada chamada já roda em FP32
puro por construção (ver "Dtype" acima), então esta diferença nem chega a se manifestar nos
resultados da Fase A/B.

**Julia-1**: não aplicável — em CPU a inferência é sempre FP32 (autocast BF16 só liga com
`device.type=='cuda'`, nunca usado neste projeto). Nenhuma comparação adicional é necessária.
