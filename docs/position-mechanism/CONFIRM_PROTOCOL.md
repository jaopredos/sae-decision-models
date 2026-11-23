# Confirmação prospectiva do caminho no teste

Congelado após a descoberta de caminho no job 4288 e antes de qualquer execução desse
experimento no teste. O benchmark já foi examinado em estudos anteriores do grupo:
esta é uma confirmação prospectiva desta hipótese, não um teste nunca visto pelo grupo.

## População e hipótese primária única

Todos os 400 casos e 600 decisões choice do split test na revisão fixada do Typed Decisions.
Nenhuma seleção por erro, flip, workflow, denominador ou tamanho de efeito. Mesmas duas
ordens (canônica e rotação de uma), ambas as direções, mesmo Laya multilíngue, sem treino.

A hipótese primária é que o caminho nonmarkers(enc17) -> markers(enc18) transmite uma
mudança de pontuações na direção da execução doadora maior que a média dos quatro
deltas isotrópicos de mesma norma por token.

Para cada decisão e direção, calcular o produto interno entre a mudança de logits centrados
do caminho e o contraste doador menos receptor. Subtrair a média dos quatro numeradores
nulos. Dividir a soma dessas diferenças pela soma dos denominadores comuns, isto é, normas
quadráticas dos contrastes doadores menos receptores. Manter perguntas, direções, caminho
e ruídos juntos; reamostrar os 400 casos com reposição, 2.000 vezes, semente 0.

Critério operacional de sucesso: estimativa positiva e limite inferior do IC95% pareado
por caso acima de zero. Se não atender, registrar ausência de confirmação deste contraste;
não substituí-lo por outro achado secundário. O coeficiente não é porcentagem de mediação.

## Resultados secundários congelados

Reportar caminho isolado, ruídos, transmissão sem restauração, caminho pelos demais tokens
e contraste caminho menos caminho com MLP restaurada. O contraste da MLP é secundário,
sem virar novo critério de sucesso. Restaurar atenção nos marcadores é um controle estrutural:
os marcadores entram limpos e a MLP é por token, de modo que restaurar atenção deve devolver
o receptor. Esse resultado não identifica a origem do viés nem uma cabeça específica.

Reportar efeitos absolutos e ortogonais, medianas/quantis por decisão, peso máximo por caso,
e recuperação de escolha separadamente nos pares com flip. Nenhuma direção/permutações
do mesmo caso será tratada como observação independente. Razões individuais com denominador
<= 1e-8 são indefinidas; os casos permanecem nos efeitos absolutos e na projeção agregada.

## Execução e checagens

FP32, sem TF32, caminho instrumentado e não instrumentado em paridade; MHA fast path desativado.
Semente global 0 e ruídos com a mesma regra de hash do estudo de descoberta. Alinhamento
bijetivo por papel/identidade/deslocamento, IDs iguais, padding preservado. Orçamentos 1024/256
e rejeição de truncamento. Mesmas condições de PATH_PROTOCOL.md.

Self patch e restauração total em enc18: erro máximo 1e-5 logit. Adicionar em cada caso
e direção o controle de todos os tokens em enc_final, que deve reproduzir o doador com
tolerância 1e-4 logit. A paridade oficial usa tolerância 0,00011 de probabilidade e mesma
escolha. Falhas de implementação interrompem a coleta, não contam como resultado negativo.

Antes do teste, piloto nos primeiros dois casos de descoberta pelo hash, sem usar casos de
teste para ajustar código. Esse piloto deve reproduzir as condições não aleatórias do
job 4288 em até 1e-4 logit. A rotina compartilhada recebe apenas a opção de controle final;
as intervenções científicas permanecem iguais.

Fixar uma thread de operações CPU e interop, registrando a configuração, para evitar
possível sobrescrição das quatro CPUs do contêiner. Isso é uma escolha operacional:
o piloto verifica que ela preserva as intervenções determinísticas. Os ruídos mantêm
a regra de geração e a norma pareada; não se exige identidade bit a bit dos controles
aleatórios entre bibliotecas ou configurações de redução.

Limite do job: cinco horas, com checkpoints atômicos por decisão. O dimensionamento
conservador usa os cerca de 20 s por decisão da descoberta (600 decisões ≈3h20) e margem;
o ajuste de threads pode reduzir esse tempo, mas não é assumido como ganho medido.
Retomar somente com manifesto, código, argumentos e protocolo idênticos. Não inspecionar
saídas parciais para redefinir hipóteses ou interromper por resultados científicos.

## Limite da conclusão

Sucesso confirma um caminho de transmissão sob intervenções em entradas híbridas e acima
do controle isotrópico especificado. Não demonstra necessidade no processamento natural,
a origem inicial da sensibilidade, a separação entre RoPE e máscaras/janelas de atenção,
uma cabeça isolada, nem generalização para todos os modelos System One.
