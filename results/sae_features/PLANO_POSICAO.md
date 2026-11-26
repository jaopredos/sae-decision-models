# O que causa a sensibilidade à posição nos modelos System One: plano e resultados

Para a reunião. Detalhes: `REPORT.md` §8 a §10; critérios e expectativas fixados antes de rodar: `PROTOCOLO.md`, emendas 6 e 7.

## A pergunta (reduzida)
**"O que acontece internamente nos modelos System One que causa sensibilidade à posição?"** Sensibilidade = a escolha muda quando só a ordem das opções muda. Foco em `score` (escolhido depois de ver o dado; `choice` e `noul` saem ao lado).

## Resposta em uma frase
Em `score`, a sensibilidade à posição tem uma **causa interna localizada**: o sinal de posição que existe nos marcadores das opções nas camadas médias e tardias (14 e 22) e que a cabeça de decisão lê. Removê-lo reduz a troca da escolha em cerca de **um quinto** (**um quarto** nas camadas 14 + 22 juntas) sem perder acurácia, e **5 a 20 features** do SAE da camada 17 carregam parte dele (o efeito se repete com uma segunda semente do SAE). Não é a única causa: ~60% das decisões continuam trocando. **E o fenômeno é específico do checkpoint multilíngue**: o Laya base em inglês quase não tem a sensibilidade (troca 0,31 contra 0,83).

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
| **Causal:** `final_norm` / cabeça / conjuntos | `final_norm` −15,0 p.p.; cabeça só perturba; **14 + 22: −20,5 p.p. (~25%)**, acurácia +2,4; efeito sub-aditivo |
| **Causal:** 5–20 features de `enc_17`, semente 1 | −7 a −8 p.p., acurácia intacta (repete a semente 0) |
| **Outros modelos** | troca em `score`: multilíngue 0,83; base (inglês) 0,31; ajustado 0,21; só o multilíngue tem a aversão à 1ª opção |

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
- O foco em `score` e os limiares dos critérios são convenções nossas; o critério do ajuste fino (−10 p.p.) **não foi atendido por 0,5 p.p.** e não foi retocado.
- O mecanismo causal só foi testado no multilíngue; **não sabemos o que torna o multilíngue tão sensível** (o treino multilíngue? a largura? outra coisa), e esse é o ponto que um revisor vai atacar primeiro.
- O ranking de features individuais depende da semente (`REPORT.md` §9); E3 só usa a semente 0.

## O que não foi feito (trabalho futuro)
Activation patching entre rotações, ablação de cabeças de atenção, **permutações completas** de `score` (separariam posição de "ordem esperada da escala", a ressalva mais provável de revisor), estado sem conteúdo, e os experimentos causais nos modelos em inglês (o base nem é sensível). Ficam registrados porque revisor vai perguntar.

## Se der tempo (cada um exige protocolo escrito antes)
| próximo passo | pergunta | custo |
|---|---|---|
| **Permutações completas só de `score`** (24 ordens de 4 níveis) | A aversão à 1ª opção é posição ou prior sobre a ordem da escala? | extração de ~10 min + análise |
| Outro checkpoint multilíngue, se existir | O que torna o multilíngue tão sensível? | depende de haver o modelo |
| Remover posição + conteúdo juntos | O resto da sensibilidade (~60%) é interação posição × conteúdo? | ~20 min |
