# SAE e transmissão do contraste de ordem — protocolo v1

Registrado antes da primeira inferência SAE desta investigação (2026-11-25).
Base: experiment/position-mechanism @2317b25. Nenhuma nova hipótese confirmatória
é avaliada no teste nesta etapa. O teste já foi usado na investigação anterior.

## Pergunta e etapas

O alvo é a mudança de preferência entre ordens semanticamente alinhadas, não o erro.
A: auditar compatibilidade e preservação do contraste; B: selecionar features somente
na descoberta; C: transplantar valores e controles; D: decidir sobre variantes/treino;
E: comparar com intervenções nas ativações. Este primeiro runner implementa A e
coleta estatísticas exploratórias para B. Não implementa ainda seleção causal,
atribuição por gradiente nem a separação experimental dos fatores posicionais.

## Piloto e população

Dois primeiros casos de descoberta pelo hash case-order-v1; todos os seus choice,
canônica versus rotação de uma posição, ambas as direções. Não selecionar por erro,
flip nem tamanho do efeito. Após auditoria do piloto e dimensionamento: os mesmos
64 casos/94 decisões da investigação anterior. O piloto terá 1 ruído por condição;
a rodada completa terá 4. Piloto não sustenta conclusões científicas de estabilidade.
Todos os prompts pertencem ao universo de treino sem rótulos do dicionário SAE;
a fidelidade é no domínio. Desenvolvimento permanece fora da seleção de features.

## Compatibilidade

Conferir SHA256 dos dois checkpoints principais contra SHA256SUMS.txt do fornecedor;
metadados e state_dict estritos; modelo/revisão/subfolder/camada/seed/dimensão/k,
escala global, threshold e capture encoder block output antes de final_norm.
Registrar checkpoint extra e metadados completos. Os SAEs foram treinados com o
forward bf16 anterior; esta investigação usa FP32. A diferença de precisão deve
constar como mudança de distribuição e ser avaliada empiricamente, não presumida
irrelevante. encode aplica a escala e decode retorna unidades do modelo.

## Domínio e intervenções

Alinhar tokens por papel/identidade/deslocamento usando token_alignment existente,
conferindo bijeção e IDs. Runner rejeita truncamento e colapso de alternativas.
E = tokens não massivos nas duas ordens limpas; O = complemento no grupo.
Não codificar O nem extrapolar reconstrução a O. Reportar cobertura E/O por grupo.

Para a=decode(encode(h)) e e=h-a, em E: Δh=Δa+Δe. Nos grupos marcadores,
demais tokens e ambos, em enc_17 e separadamente enc_08:
self; Δh completo; Δh em E; Δa em E; Δe em E; Δh em O;
Δa+Δe em E + Δh em O; ruídos isotrópicos de mesma norma por token que Δa.
Cada edição é hR+delta; reconstruído preserva o residual receptor. Reconstrução
substitutiva (h'=a) em ambas as ordens, com o mesmo E alinhado, é diagnóstico
separado. Registrar logits das duas ordens e preservar o contraste limpo como eixo.

Para demais tokens enc_17, repetir intervenções restaurando os demais tokens
receptores em enc_18, deixando passar somente marcadores atualizados. Restaurar
TODOS em enc_18 é controle adicional. Reproduzir o direto e o caminho de path_v1.
Não presumir aditividade de logits a partir da identidade vetorial de ativações.
O desenho canônica/rotação confunde ordinal, posição absoluta, distâncias e vizinhança;
nenhuma feature recebe rótulo de posição apenas com essa associação.

## Gates técnicos e decisão científica

Self e restauração total: erro máximo 1e-5 logits. Alinhamento final, reprodução
das referências e recombinação: 1e-4. Identidade algébrica em ativações: 1e-3 absoluto
(necessário por cancelamento FP32 dos vetores massivos). Testes sintéticos verificam
unidades, simetria do domínio, sinais e identidade sem edição. Falha técnica interrompe.

Não confundir gate técnico com resultado científico. Sinalizar como fidelidade
insuficiente para atribuição primária se a projeção do contraste sob substituição
em ambas as ordens se afastar de 1 por mais de 0,20 ou se erro L2 relativo do
contraste superar 0,50. Limiares convencionais, não escolhidos por resultados.
Comparar transferência reconstruída com E direto e ruído, com IC pareado por caso;
a escolha de features só prossegue após examinar essas medidas e o papel de O.
Se O importar, auditar primeiro o SAE literal existente e as interações E/O;
nenhum treino novo é automaticamente autorizado por uma FVE baixa.

## Análise e próximos protocolos

Salvar momentos para FVE agrupada, SSE do próprio delta, produto interno/normas,
features |Δz|/Δz/frequência por grupo, posição ordinal dos marcadores, coordenadas
absolutas, comprimentos e IDs de tokens alinhados. Estatísticas de features desta
etapa são candidatos exploratórios sem causalidade ou rótulo interpretativo.

Usar contraste centrado limpo doador-receptor; projeção agregada = soma dos produtos
internos / soma das normas quadráticas, não média de razões nem porcentagem explicada.
Bootstrap 2.000 por caso, seed 0, mantendo perguntas, direções, condições e camadas.
Reportar denominadores <=1e-8, distribuição de efeitos individuais, normas absolutas,
ortogonais, maior peso de caso e flip estratificado sem tratar pares como independentes.
Acurácia, entropia, margem top-2 e KL devem ser reportados como efeitos colaterais.
Uniformizar as decisões não conta como explicar/remover o mecanismo.

Congelar critérios/listas/K/controles pareados por frequência e magnitude antes
avaliar transplantes de features no desenvolvimento. Mais permutações e manipulações
cruzadas de conteúdo/ID/posição são necessárias antes de alegar seletividade posicional.
Os testes comparativos futuros no teste usado são pareados e têm essa limitação;
confirmação genuinamente nova exige um conjunto adicional definido antes da observação.

## Operação e integridade

Somente jobs GPU; 1 thread CPU e interop; FP32 explícito, AMP/TF32/MHA fastpath off.
Workspace novo, código congelado por commit/archive, ambiente existente usado sem
sincronizar/reinstalar pacotes; HF_HOME/TORCH_HOME preservados. Nenhum arquivo de
outro job é alterado. Checkpoints JSON atômicos por decisão/camada, manifesto dos
arquivos/pesos/referência e hashes finais de decisões; recusar retomada incompatível.
Piloto: GPU mínima 20 GB (classe disponível 89), 8 GB RAM, 2 CPUs, 15 min. Reavaliar
recursos e tempo observado antes da coleta completa. Backups externos com SHA256.
### Adendo operacional antes do primeiro job

O CLI instalado em 2026-11-25 rejeita submit -c antes de criar o job.
Sua ajuda documenta submit -f com command: ou script:. Este piloto usa
command: bash scripts/sauron/position_sae_pilot.sauron, validado por --dry-run.
Nenhuma mudança do desenho científico ou observação de resultados ocorreu.

O job 4504 falhou no parser Bash antes dos testes/inferência: git archive no
Windows exportou CRLF por core.autocrlf=true. Correção de empacotamento:
git -c core.autocrlf=false archive, atributo LF no script e validação Bash
antes de submeter novamente. Nenhum resultado científico foi observado.
