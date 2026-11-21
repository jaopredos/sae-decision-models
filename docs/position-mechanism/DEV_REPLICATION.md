# Replicação no desenvolvimento do efeito da ordem

Protocolo congelado após a descoberta e antes de inferência no desenvolvimento.
A descoberta foi coletada no commit 68d7fab, job 4056. Nenhum caso de desenvolvimento
ou teste é usado para escolher este contraste. A hipótese é específica do checkpoint
multilíngue do Laya e das perguntas choice do Typed Decisions.

## Hipótese primária

A transferência dos marcadores entre duas ordens transmite mais da mudança de logits
em enc_18 do que em enc_17. O contraste primário é a projeção agregada em enc_18 menos
a projeção agregada em enc_17, usando os mesmos casos, perguntas e direções.

A projeção por camada é soma dos produtos internos dividida pela soma dos quadrados
do contraste doador menos receptor, depois de centrar logits por decisão. Sua diferença
não é uma fração de mediação causal. Ela dá maior peso a decisões cujo contraste de
ordem tem maior magnitude.

Critério operacional de replicação: diferença positiva, com limite inferior do IC95%
bootstrap pareado por caso acima de zero. Um único contraste primário; as demais
camadas, métricas e controles são referências secundárias/exploratórias. Não escolher
um novo contraste após observar desenvolvimento e chamá-lo de confirmação.

## População e controles

Comportamento: todos os 180 casos de desenvolvimento definidos no projeto, apenas choice.
Patching: todos os choice dos primeiros 64 desses casos pelo hash case-order-v1, sem
seleção por erro, flip ou tamanho de efeito. Par de ordens fixo: canônica e rotação de um,
ambas as direções. Mesmos orçamentos, FP32 e checagens da descoberta.

Manter varredura das 25 camadas para comparabilidade, mas sem tratar a curva inteira
como 25 hipóteses confirmatórias. Mesmas intervenções: todos os marcadores, um marcador
escolhido por hash, identidades erradas, quatro ruídos de norma pareada e self patch.
Exigir erro zero até a tolerância predefinida no self patch e controle final positivo.

Comparar enc_17 e enc_18 com seus ruídos como apoio de especificidade; quatro direções
aleatórias não permitem p-valores precisos. enc_08 é uma referência precoce e enc_22,
head_1 e head_2 referências tardias. O controle final é trivial por construção.

## Análise

Reamostrar os 64 casos com reposição, mantendo juntas suas perguntas e duas direções;
em cada uma de 2.000 reamostras, recalcular ambas as projeções e sua diferença. Semente 0.
Reportar estimativa, IC95%, projeções por camada, efeito absoluto, efeito ortogonal,
medianas e quantis individuais. Incluir todos os pares, mesmo sem flip, e relatar a
recuperação de escolha separadamente nos pares com flip, agrupando por caso.

No comportamento, reportar flip em qualquer ordem, desacordo com a ordem canônica,
variação de probabilidades e entropia. Comparar original, vazio e IDs renomeados de
forma pareada por caso. O controle de estado trocado apenas permuta o conjunto de
estados dentro da mesma especificação de perguntas: a distribuição marginal agregada
é igual por construção. Seus contrastes entre casos são descritivos, sem IC ingênuo por
receptor. Não calcular acurácia de estados vazios/trocados contra o gold do receptor.

Se replicar, o achado localiza a concentração/transmissão do efeito nos marcadores do
encoder tardio. Não atribui sua origem ao bloco 18. Próximo experimento: patch fatorial
de marcadores, demais tokens e ambos, nas camadas 17/18, seguido por intervenções nas
ramificações de atenção/MLP com residual preservado. Se não replicar, registrar o
resultado negativo e revisar a hipótese antes de qualquer uso do teste.

O teste de 400 casos permanece reservado para uma confirmação final com protocolo
próprio. Não submeter inferência no teste neste job.
