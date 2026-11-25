# Plano: o que causa a sensibilidade à posição nos modelos System One

Documento para a reunião de planejamento. Detalhes e critérios fixados antes de rodar: `PROTOCOLO.md`, emendas 6 e 7. Números já medidos: `REPORT.md` §8 e §9.

## A pergunta (reduzida)
**"O que acontece internamente nos modelos System One que causa sensibilidade à posição?"** Sensibilidade = a escolha muda quando só a ordem das opções muda. (A parte "como controlar" saiu do escopo.)

## O que já temos que responde à pergunta (nada a rodar)
| achado | número | de onde |
|---|---|---|
| A sensibilidade existe e varia muito por tipo de pergunta | a escolha muda em 82–91% das decisões `score`, 30–44% das `choice`, 6% das `noul` | `REPORT.md` §8 |
| Em `score` há um viés posicional grande | a 1ª opção é escolhida 6,8% das vezes (esperado 25%); a última +10,9 p.p. | §8 |
| Em `choice` o viés médio é de conteúdo, não de posição | maior desvio por posição 3,2 p.p. (limiar fixado: 5) | §8 |
| A posição é codificada por dentro nas camadas iniciais | η² de posição 0,91–0,96 nas camadas 1–4, 0,13 na camada 8, ~0,05 no fim; o conteúdo assume entre as camadas 4 e 6 | §8 |
| O probe da etapa 3 depende da ordem original | camada 8: 0,50 → 0,27 nas rotações; camada 17: 0,52 → 0,42 | §8 |
| Com a ordem balanceada o Laya fica no acaso | 0,334 (IC95% 0,317–0,353) contra 0,318 | §8 |
| As features do SAE ligadas ao erro são sobretudo de conteúdo | só 12% (camada 8) e 7% (camada 17) têm mais posição que conteúdo | §8 |

Isso descreve **onde** a posição aparece. Falta mostrar se ela **causa** a sensibilidade.

## O que falta (3 experimentos, todos reaproveitando dados e código existentes)
| # | experimento | pergunta | reaproveita | compute estimado* |
|---|---|---|---|---|
| E1 | Sondas por camada: posição, conteúdo e resposta (treinada nas rotações) | O que está legível em cada camada, sem a posição correlacionada com a resposta? | os vetores de rotação já extraídos; nenhuma passada nova do modelo | minutos |
| E2 | **Remover o subespaço de posição** de uma camada (só nela, ou dali em diante) | A informação de posição, numa camada, causa a sensibilidade? | vetores de rotação, o `Runner` e o nulo pareado da ablação | ~1 h por modo |
| E3 | Zerar as features do SAE de maior η² de posição | As features do SAE que carregam posição causam a sensibilidade? | SAEs, η² por feature, `SAEIntervention`, o nulo pareado | ~10 min por camada |

\* Estimativa sem medida; a fila do Sauron é compartilhada com outros projetos e pode atrasar.

## Os três desfechos possíveis, e a história de cada um
1. **E2 qualifica em alguma camada** (a remoção reduz a sensibilidade sem destruir a acurácia, e vence o nulo): "a posição codificada na camada L é causalmente necessária para parte da sensibilidade". É o resultado mais forte. Quanto da sensibilidade ela explica é a fração Δflip / flip de base.
2. **E2 não qualifica** (o que eu espero, ver `PROTOCOLO.md`): "a posição é legível por dentro, mas removê-la de uma camada não basta; a sensibilidade parece distribuída (recalculada pelas camadas seguintes)". É um resultado negativo honesto, e combinado com o E1 e o η² já descreve o mecanismo (onde a posição vive e que não é um ponto único).
3. **E2 só funciona no modo "dali em diante"**: a sensibilidade depende de a posição existir em várias camadas; quanto mais cedo a remoção, mais efeito.
Em todos os casos, `choice` e `noul` saem ao lado, e o foco é `score` (escolhido depois de ver o dado, e isso fica declarado).

## O que NÃO vai ser feito (trabalho futuro)
Activation patching entre rotações, ablação de cabeças de atenção, permutações completas, estado sem conteúdo, rotação nos modelos em inglês. Fica registrado porque revisor vai perguntar.

## Ressalvas que o texto precisa ter
- `score` é uma escala ordinal (Benign → High): rotacionar a embaralha, e o Laya evitar a 1ª opção pode ser um prior aprendido ("o primeiro item quase nunca é o certo"), não um mecanismo de posição. As duas leituras cabem.
- A escolha de focar em `score` e os limiares dos critérios são convenções nossas.
- Os experimentos novos não foram rodados no cluster ainda; só um teste de encanamento em 20 casos.
- O ranking de features individuais depende da semente do SAE (`REPORT.md` §9): o E3 só usa a semente 0.

## Como contar na segunda
- Resultado principal (já existente): sensibilidade grande em `score`, que é de posição; em `choice` é de conteúdo; onde ela vive por dentro (camadas 1–4).
- Resultado causal (E2/E3): a depender do desfecho acima.
- Duas correções honestas que tornam o artigo mais forte: o 0,574 do probe não é "conteúdo puro"; com a ordem balanceada o Laya está no acaso.
