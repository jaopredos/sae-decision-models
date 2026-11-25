# Execução e retomada

## Rodada atual

- Job **4507**, ID `a069279a17e148a799f96f3b1ef3e39f`.
- Submetido em 2026-11-25. Estado após submissão não acompanhado continuamente.
- Código/preparação: `e28c74b`; código científico idêntico ao piloto `728eeed`.
- Objetivo: etapa A, auditoria de fidelidade do contraste de ordem; estatísticas Δz
  exploratórias para B, sem seleção causal de features nem treino novo.
- 64 casos/94 decisões choice de descoberta; canônica/rotação, ambas as direções;
  SAEs principais enc_17 e enc_08; quatro ruídos por condição.
- Workspace `~/workspace/sae-position-sae-mechanism-v1`.
- Saída `results/position_sae_mechanism/discovery_audit_v1` nesse workspace.
- Recursos: 1 GPU, mínimo solicitado 8 GB VRAM (classe disponível 89 GB), 8 GB RAM,
  2 CPUs, 1 thread CPU/interop, limite 15 min. FP32/AMP off/TF32 off/MHA fastpath off.

```powershell
ssh sauron "sauron status 4507"
ssh sauron "sauron logs 4507 -f"
```

## Próxima retomada

1. Conferir status e logs, COMPLETE, recursos e integridade; falha técnica não é
   resultado científico negativo. Não modificar arquivos do job enquanto ativo.
2. Compactar resultados, baixar para results/position_sae_mechanism/backups e
   conferir SHA256 nas duas pontas antes de analisar.
3. Rodar offline `python analyze_position_sae.py --input <diretorio> --out <analise>`.
   Auditar denominadores/exclusões, FVE/delta por grupo/posição, fidelidade de
   contraste, efeitos colaterais, E/O e distribuição; adicionar contrastes pareados
   reconstruído versus E direto e ruído. Não interpretar intervalos por direção
   como independentes e não chamar projeções de porcentagens explicadas.
4. Decidir se a fidelidade permite B/C. O piloto já sinalizou perda de contraste
   nos demais tokens: não ignorar esse sinal por boa FVE ou acurácia individual.
5. Se necessário, auditar variantes já existentes (literal/k64/seed1) antes de
   cogitar treino. Separar diagnóstico daquela configuração de juízo geral sobre SAE.
6. Congelar seleção, mais ordens/manipulações e controles pareados antes de
   validação causal em desenvolvimento; não selecionar features pelo erro do modelo.
7. Comparativo final e revisão metodológica continuam pendentes. O teste existente
   já foi usado. Confirmação genuinamente nova exige conjunto adicional pré-definido.

O piloto 4505 passou e foi preservado com hashes em REPORT.md. Job 4504 falhou
antes da inferência por CRLF; preservado como incidente operacional.

## PR recomendado

Base: `experiment/position-mechanism`; head: `experiment/position-sae-mechanism`.
Não foi criado PR, merge ou force-push. Relatórios parciais distinguem o que passou
tecnicamente, o alerta de fidelidade e as etapas científicas ainda não realizadas.