# O que causa a sensibilidade à posição nos modelos System One: plano e resultados

Para a reunião. Detalhes: `REPORT.md` §8 a §10; critérios e expectativas fixados antes de rodar: `PROTOCOLO.md`, emendas 6 e 7.

## A pergunta (reduzida)
**"O que acontece internamente nos modelos System One que causa sensibilidade à posição?"** Sensibilidade = a escolha muda quando só a ordem das opções muda. Foco em `score` (escolhido depois de ver o dado; `choice` e `noul` saem ao lado).

## Resposta em uma frase
Em `score`, a sensibilidade à posição tem uma **causa interna localizada**: o sinal de posição que existe nos marcadores das opções nas camadas médias e tardias (14 e 22) e que a cabeça de decisão lê. Removê-lo reduz a troca da escolha em cerca de **um quinto** sem perder acurácia, e **5 a 20 features** do SAE da camada 17 carregam parte dele. Não é a única causa: ~65% das decisões continuam trocando.

## Os números (todos `score`, teste, salvo indicação)
| o que | resultado |
|---|---|
| Sensibilidade de base | a escolha muda em 83% das decisões de `score` (30–44% em `choice`, 6% em `noul`) |
| Viés posicional do `score` | a 1ª opção é escolhida 6,8% das vezes (esperado 25%); a última +10,9 p.p. |
| Onde a posição vive (variância) | domina nas camadas 1–4 (η² 0,91–0,96); o conteúdo assume entre a 4 e a 6 |
| Posição é **legível**? | sim, em todas as camadas do encoder (acurácia ~1,00; acaso 0,24); só cai na cabeça (0,64) |
| Há conteúdo da resposta sem a posição? | sim: sonda treinada nas rotações chega a 0,559 (IC95% 0,539–0,579; Laya 0,334) |
| **Causal:** remover o subespaço de posição na camada 22 | troca 0,829 → 0,665 (−16,4 p.p.), acurácia +1,6 p.p., controle de conteúdo −0,5; a aversão à 1ª opção some |
| **Causal:** idem na camada 14 | −17,9 p.p. (acurácia −1,5; um pouco menos específico) |
| **Causal:** zerar 5–20 features de maior η² de posição em `enc_17` | −10 a −11 p.p., acurácia −0,6, controle 0,0 |
| Mesma coisa em `enc_08` | nenhum efeito |
| Camadas 1–6 e 10 (remover a posição) | só perturba (a troca sobe 2–8 p.p.) |

## O que eu tinha escrito antes de rodar, e errei
Esperava **poucos ou nenhum** efeito causal (a posição seria recalculada pelas camadas seguintes). Erros: as camadas 14 e 22 reduzem a troca em 16–18 p.p.; 5–20 features de `enc_17` bastam. Acertei em `enc_08` (features sem efeito) e na legibilidade da posição. Tudo reportado como saiu.

## Como contar na segunda
1. **Fenômeno:** a sensibilidade é enorme em `score` (83%), moderada em `choice` e pequena em `noul`; só em `score` há viés posicional grande.
2. **Mecanismo, em três passos:** (a) a posição é codificada cedo (camadas 1–4) e fica **legível o tempo todo**; (b) o conteúdo está lá também (sonda 0,559); (c) **removê-la nas camadas 14 e 22 reduz a sensibilidade**, e uma meia dúzia de features do SAE de `enc_17` carrega parte do sinal.
3. **Limite honesto:** explica um quinto, não tudo; só `score`; só marcadores; um SAE.
4. **Duas correções que fortalecem o artigo:** o 0,574 do probe se apoiava na posição (mas o conteúdo está na representação, 0,559); com a ordem balanceada o Laya não passa do acaso (0,334 contra 0,318).

## Ressalvas que o texto precisa ter
- `score` é uma escala ordinal: o Laya evitar a 1ª opção pode ser um prior aprendido, não um mecanismo; posição e "ordem esperada da escala" não se separam por rotação.
- 26 células foram avaliadas no dev e 4 qualificaram, número que o acaso poderia produzir em 1 ou 2; três replicam no teste. O p empírico mínimo com 20 sorteios é 0,048.
- O modo "da camada L até a 22" **não soma** os efeitos (remover da 14 à 22 dá −2 p.p.): **não sei explicar**.
- O foco em `score` e os limiares dos critérios são convenções nossas.
- O ranking de features individuais depende da semente (`REPORT.md` §9); E3 só usa a semente 0.

## O que não foi feito (trabalho futuro)
Activation patching entre rotações, ablação de cabeças de atenção, permutações completas, estado sem conteúdo, rotação nos modelos em inglês, e os subespaços de `final_norm` e da cabeça de decisão (o efeito se concentra no fim). Ficam registrados porque revisor vai perguntar.

## Se der tempo (cada um exige protocolo escrito antes)
| próximo passo | pergunta | custo |
|---|---|---|
| Remover o subespaço em `final_norm` e nas camadas da cabeça | O efeito é maior depois da 22? | job curto |
| E3 com a semente 1 | A lista de 5–20 features é estável entre sementes? | ~15 min por camada |
| Remover o subespaço de posição **e** o de conteúdo juntos, ou posição em 14 + 22 | Os efeitos somam? (hoje não somam) | ~20 min |
| Rotação no Laya ajustado (inglês) | O fine-tuning reduz a sensibilidade? | extração curta |
