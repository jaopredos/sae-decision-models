# Contribuição dos marcadores e dos demais tokens

Estudo exploratório após a replicação do contraste enc_18 menos enc_17 no desenvolvimento.
Usa somente os mesmos 64 casos de descoberta do job 4056, incluindo todos os seus choice,
sem selecionar por erro, flip ou tamanho de efeito. Nenhuma inferência no teste ou ajuste
do contraste confirmado no desenvolvimento. Modelo e dados permanecem nas revisões fixadas.

## Pergunta

Quanto da mudança de pontuações entre duas ordens é transferido pelas representações dos
marcadores, pelas representações dos demais tokens e pela transferência conjunta?
Se os demais tokens contribuírem, a baixa transferência precoce dos marcadores pode
coexistir com informação já disponível no restante da sequência. Isso não identifica
onde a dependência da ordem se originou.

## Desenho

Mesma pergunta e estado; ordem canônica e rotação de uma posição, ambas as direções.
Capturar cópias FP32 independentes da sequência em enc_17, enc_18 e enc_final. Construir
um alinhamento bijetivo por papel, identidade da alternativa e deslocamento no segmento:
prefixo da pergunta, cada segmento MASK+descrição, separador e estado. Conferir os IDs
dos tokens e a cobertura exata dos tokens válidos; tokens repetidos não definem o mapa.
Padding fica inalterado. Não editar position_ids nem os mecanismos RoPE posteriores.

Em enc_17 e enc_18, transferir: (1) marcadores; (2) todos os demais tokens; (3) ambos.
Para cada grupo, quatro ruídos isotrópicos com a mesma norma da mudança por token.
Registrar número de tokens endereçados e norma total; os grupos têm tamanhos e energias
diferentes. Não interpretar maior efeito dos demais tokens como mecanismo mais específico.
Conservar todos os vetores de logits, numeradores, denominadores e efeitos absolutos.

## Verificações obrigatórias

- Paridade com o caminho oficial e tolerâncias da coleta anterior.
- Self patch em cada camada: erro máximo 1e-5 logit.
- Transferir apenas marcadores deve reproduzir a intervenção da descoberta em cada caso,
  camada 17/18 e direção, com erro máximo 1e-4 logit.
- Transferir todos os tokens alinhados em enc_final deve reproduzir o doador com erro
  máximo 1e-4 logit. A cabeça sem posição é equivariante à permutação dos vetores; esse
  é um controle de alinhamento, não uma descoberta de mecanismo.
- O piloto de dois casos deve passar antes dos 64 casos completos.

Uma transferência completa em enc_17/18 não precisa recuperar exatamente o doador: as
camadas restantes continuam usando a geometria posicional da execução receptora.

## Análise posterior

Comparar a projeção agregada dos três grupos com seus controles, além do efeito absoluto,
componentes ortogonais e recuperação de escolha. Reamostrar casos mantendo perguntas,
direções, grupos e camadas juntos. Usar 2.000 reamostras, semente 0. As análises são
exploratórias; qualquer hipótese de confirmação posterior deve ser congelada antes de
nova avaliação independente.

Calcular a interação vetorial centrada:
Y(ambos) - Y(marcadores) - Y(demais) + Y(original).
Projetá-la sobre o contraste doador menos receptor usando o mesmo denominador.
Interação significa não aditividade sob estas intervenções, não prova de um circuito.

Se houver contribuição adicional dos demais tokens, planejar intervenções nas saídas
das ramificações de atenção e MLP, com residual preservado, para distinguir caminhos.
Não atribuir automaticamente o efeito a uma cabeça nem ao bloco 18 inteiro.
