# Features do contraste de ordem — protocolo exploratório B/C v1

Congelado após a auditoria A (job4507) e antes da primeira intervenção de features.
A regra e o código de seleção foram definidos antes de observar resultados causais
por feature. Somente descoberta; desenvolvimento e teste não serão acessados.

## Decisão após A

enc17 k32 passa o gate funcional: replace bilateral demais tokens P=0,8951,
erro relativo=0,2462. Delta reconstruído no caminho P=0,3205, acima da média dos
quatro ruídos por 0,3062 [0,2699;0,3366]. O delta geométrico tem erro relativo
0,7049: não confundir fidelidade funcional específica com reconstrução perfeita.
Enc08 demais tokens falha e não será alvo primário de atribuição. O efeito isolado
fora do domínio é pequeno neste caminho; não treinar um SAE novo nesta etapa.

## Seleção congelada, sem alvo de erro

Usar os mesmos 64 casos/94 choice e as duas direções canônica/rotação. O |delta z|
é igual nas duas direções e entra uma vez por pergunta, sem duplicar observações.
Domínio E é interseção não massiva limpa nas duas ordens; somente nonmarkers enc17.

Quatro folds por caso: ordenar casos dentro de cada workflow pelo stable_seed
sae-feature-fold-v1; distribuir ciclicamente. Cada avaliação usa ranking dos outros
folds. Ranking = média entre casos da média entre perguntas de soma_t |delta z_i|/n_E.
Suporte mínimo: ativa em ao menos 20 casos de seleção. K=32 fixo como orçamento de
edição, não estimativa do número de features verdadeiras. Não usar gabarito, erro,
flip, gradiente nem logits no ranking. Dicionário treinado sem rótulos no universo
de descoberta: cross-fitting avalia a regra de seleção, não independência do SAE.

A lista calculada na descoberta completa fica congelada para futura avaliação no
desenvolvimento. Arquivos: features/selection.json, feature_ranks.csv e control_balance.csv.
Não ajustar K, suporte ou ranking aos efeitos C. Estabilidade entre folds é descritiva.

## Controles congelados

Oito conjuntos de K=32, sem features selecionadas nem repetição entre os oito conjuntos
de um fold. Pareamento em quintis de frequência e mudança média por token ativo,
medidas apenas nos casos de seleção. Sorteio entre os oito vizinhos mais próximos
no espaço log padronizado; semente sae-matched-v1. Se não houver candidato na célula,
usar vizinho global e registrar fallback/distância. A lista congelada teve três
fallbacks em 1280 pareamentos (inclui lista completa futura); não ocultar isso.

Além do balanceamento de seleção, registrar norma efetiva da edição por caso e
controle. Frequência/magnitude pareadas não garantem dose idêntica no hidden ou
mesmo papel semântico. Não alegar especificidade se balanceamento for inadequado.
Quatro ruídos isotrópicos com norma por token igual à edição selected-only.

## Intervenções

Preservar o residual receptor. selected-only troca zR por zD apenas em S:
  h' = hR + (zD_S-zR_S) Wdec_S / scale.
É equivalente a hR+decode(z_edit)-decode(zR); o viés cancela. Complemento drop-S:
partir do delta reconstruído completo e retirar apenas o delta de S. Isso testa
perda do efeito transplantável, não necessidade do conceito em execução natural.
Executar only/drop também para os oito conjuntos pareados. Referências: self,
todas as features e transferência direta das ativações elegíveis. Em enc18,
restaurar todos os nonmarkers receptores em todas as condições de caminho;
controle adicional restaura todos os tokens e deve recuperar o baseline.

Todas as condições usam o mesmo caso, par de ordens, domínio E e contraste limpo.
Reproduzir as referências all-features/eligible do job4507 (1e-4 logit), self e
reset-all18 (1e-5). Testar unidades, identidade e complemento em tensores sintéticos.
Piloto de dois casos obrigatório antes de nova coleta dos 64; não sobrescrever A.

## Previsão e exemplos

Calcular gradiente no baseline receptor, replay dos inputs oficiais FP32 com modelo
congelado, folha na saída enc17 e restauração de nonmarkers enc18. Alinhar logits
canônicos. Alvo escalar = produto interno dos logits com contraste limpo centrado
(doador-receptor), sem gabarito. Previsão por feature = soma_t delta z_ti *
((Wdec_i/scale) dot grad_t). Gradiente não seleciona candidatos.

O replay deve reproduzir os logits limpos em até 1e-4; diferenças finitas validam a
implementação em modelo sintético. Comparar numeradores previstos e medidos e
reportar falhas da aproximação linear. Guardar dois maiores |delta z| por feature
selecionada/caso/direção com contexto, token e posição absoluta; exemplos são
exploratórios, sem nomear conceitos pela associação a erro.

## Análise e interpretação

Primários exploratórios: selected-only menos média dos oito matched-only e a perda
all-features menos drop-selected comparada à média das perdas drop-matched. Média
sobre controles dentro de pergunta/direção ANTES do bootstrap pareado por caso.
Comparar selected-only com média de quatro ruídos de norma pareada; reportar também
all-features, eligible, efeitos absolutos/ortogonais, distribuição e recuperação
de escolha. Usar 2000 reamostras dos casos, seed0, preservando perguntas/direções.

Evidência de transmissão seletiva exige contraste positivo e IC inferior >0;
sem isso, resultado negativo para essa regra, orçamento, dicionário e intervenção.
Como escala convencional, projeção selected-only >=0,05 será descrita como efeito
material; sempre mostrar valores contínuos. A validação prospectiva futura no dev
requer congelar sua hipótese/lista antes de acessar os resultados e reportar A/C.

Entropia, top2, KL e acurácia são efeitos colaterais, não alvos. Não chamar destruição
ou uniformização de melhoria. A identificação de transmissores pode ocorrer mesmo
sem mudança de acurácia; não é evidência de correção de viés nem origem inicial.

A maioria do peso agregado de A está em customer_service (77,2%); reportar por
workflow, direção e retirada do maior caso. Deltas/atribuições podem interagir.
Não tratar features, tokens, permutações ou direções como casos independentes.

Esta rodada ainda não separa ordinal, posição absoluta, distâncias, conteúdo e
vizinhança. Variações de IDs e múltiplas permutações cruzadas serão necessárias
para qualquer rótulo de seletividade posicional. Robustez k64/seed1 permanece
planejada; índices de features não correspondem entre dicionários. Nenhuma
conclusão sobre todos os System One ou explicação integral será extraída desta rodada.

## Operação

FP32, AMP/TF32/MHA fastpath off, 1 thread CPU/interop, ambiente/caches anteriores
reutilizados sem instalar ou sincronizar. GPU job isolado, 8 GB RAM e VRAM mínima
8 GB; piloto limite10min. Recursos/tempo da coleta serão definidos após piloto.
Manifestos incluem seleção, código, pesos/revisões e protocolo; arquivos atômicos
por decisão, hashes finais e cópia externa verificada. Nenhum arquivo de job ativo
é alterado; cada rodada usa diretório novo.