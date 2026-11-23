# Caminho dos demais tokens para os marcadores no bloco 18

Exploração no mesmo conjunto de 64 casos/94 decisões de descoberta, após o fatorial.
Nenhum novo caso de desenvolvimento nem teste. Modelo, dados, ordens e FP32 inalterados.
O fatorial mostrou contribuição dos demais tokens em enc_17, mas não estabeleceu que
essa contribuição chega aos marcadores dentro do bloco 18. Este desenho testa esse caminho.

## Intervenções

Para cada caso, usar ordem canônica e rotação de uma posição, em ambas as direções.
Transplantar somente os demais tokens na saída de enc_17, mantendo os marcadores receptores.
Na saída de enc_18:

- Sem restauração: reproduzir o efeito não-marcador do fatorial anterior.
- Restaurar demais tokens receptores: deixar passar apenas os marcadores resultantes
  do processamento do bloco 18, testando o caminho nonmarkers17 -> markers18.
- Restaurar marcadores receptores: deixar passar somente os demais tokens resultantes.
- Restaurar todos os tokens receptores: controle de retorno ao baseline.

Na condição que deixa passar apenas marcadores, fazer duas intervenções adicionais:
restaurar, nos marcadores, a saída receptora original da ramificação de atenção ou da MLP
do bloco 18. Preservar o residual receptor; não substituir toda a camada por uma ramificação.
A atenção é a operação entre tokens desse bloco; a MLP e as normalizações são por token.
Clampar atenção pode também alterar a MLP subsequente, logo seu efeito não é uma decomposição
aditiva independente de atenção e MLP. Não atribuir automaticamente o caminho a uma cabeça.
Restaurar a atenção nos marcadores deve recuperar o receptor por construção: o residual
dos marcadores entra limpo e a MLP é local por token. Essa condição é um controle estrutural,
não uma descoberta independente sobre a origem do efeito.

Usar quatro deltas aleatórios de norma pareada por token em nonmarkers17, com restauração
dos demais tokens em enc_18, como controle do caminho que passa pelos marcadores. Há muito
mais tokens fora dos marcadores; não interpretar efeito amplo como especificidade de conceito.

## Verificações

Alinhamento bijetivo por papel/identidade/deslocamento, IDs iguais; padding preservado.
Paridade com o pacote oficial. Self patch e restauração completa devem retornar ao receptor
com erro máximo 1e-5 logit. A condição sem restauração deve reproduzir a transferência dos
demais tokens em enc_17 do fatorial, nas duas direções, com tolerância de 1e-4 logit.
Conferir o uso da camada encoder.layers[17] e módulos attn/mlp do ModernBertEncoderLayer.
Piloto de dois casos obrigatório antes dos 64. Gravação atômica por decisão e retomada.

## Análise planejada

Salvar todos os logits. Calcular projeção na mudança doador menos receptor após centrar
logits, efeitos absolutos/ortogonais e recuperação de escolhas. Reportar a condição de
caminho somente-marcadores e seus ruídos; contrastar com a condição sem restauração e
com a restauração da atenção ou da MLP. Reamostrar casos, mantendo juntas perguntas,
direções e condições; 2.000 reamostras e semente 0. Contrastes exploratórios, sem novos
p-valores confirmatórios nem seleção de casos por flip.

Se sobreviver efeito após restaurar os demais tokens em enc_18, a intervenção demonstra
um caminho causal através dos marcadores atualizados no bloco 18. Se restaurar a atenção
nesse caminho remover o efeito, isso satisfaz o controle estrutural esperado, sem localizar
a origem inicial do viés. Não estabelece uma cabeça específica,
nem a equivalência entre estas entradas híbridas e mudanças naturais de posição.
