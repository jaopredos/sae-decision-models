# Execução e retomada

## A concluída e auditada

Job4507 concluiu 64 casos/94 choice, enc17+enc08, quatro ruídos, ambas as direções.
Backup externo SHA256 `4265a098dd4f6c946b3e9653e3467e22fa0b7416bfb799962f11d2b673182989`.
Manifestos, hashes e condições auditados; resultados em REPORT.md e discovery/.
Enc17 passou gate funcional; enc08 demais tokens falhou. Nenhum SAE novo treinado.

## Rodada atual B/C — piloto

- Job **4511**, ID `87527809a46d42359d09762ad9a9e244`.
- Código/protocolo/seleção de inferência congelados em `aab0d30`.
- Submetido 2026-11-25; fila por cota livre zero, estimativa inicial ~18min, máximo2h.
- Não acompanhado continuamente, sem alterar os demais jobs.
- Dois casos/duas decisões de descoberta, enc17 k32, ambas as direções.
- Quatro folds por caso; top32 de |delta z| por token, sem alvo de erro, logits ou flip.
- Oito controles disjuntos pareados por frequência/magnitude; quatro ruídos de mesma
  norma por token que selected-only. Transplante selected-only e drop-selected.
- Inclui previsão por gradiente do contraste, exemplos de tokens e reprodução de A.
- Workspace `~/workspace/sae-position-sae-mechanism-v1`.
- Saída `results/position_sae_mechanism/features_pilot_v1` nesse workspace.
- 1 GPU (mínimo8GB VRAM, classe89GB), RAM8GB, 2CPUs, limite10min, 1thread CPU/interop.

```powershell
ssh sauron "sauron status 4511"
ssh sauron "sauron logs 4511 -f"
```

## Gates antes da coleta completa

1. Piloto precisa passar testes, replay de gradiente FP32 e referências all-features /
   eligible do4507 (erro<=1e-4 logit); self e reset-all18 <=1e-5.
2. Conferir COMPLETE, cobertura, hashes, recursos e preservação da lista/folds.
3. Compactar piloto, baixar e conferir SHA256 nas duas pontas; rodar offline
   analyze_position_sae_features.py. Auditor local validado ponta a ponta com modelo
   sintético e teste que detecta adulteração de uma métrica salva.
4. Passando gates técnicos, dimensionar e submeter os mesmos64/94 com código e
   seleção idênticos. Ainda não foi submetida a coleta completa de features.
5. Não adaptar K/listas/matching aos efeitos do piloto. Seus resultados causais
   serão exploratórios, e não evidência populacional por dois casos.
6. Dados completos: contrastes selected-minus-matched e perda drop-selected versus
   média de perdas dos controles; médias dentro da pergunta/direção antes do
   bootstrap por caso. Reportar dose, balanceamento, primeira ordem, efeitos
   colaterais e estabilidade por workflow/direção.
7. Depois: múltiplas ordens e controles de conteúdo/IDs/posição; robustez entre
   dicionários e validação prospectiva da lista congelada no dev. O teste já foi
   usado. Confirmação genuinamente nova requer conjunto adicional pré-definido.

Enquanto4511 estiver na fila/rodando, NÃO atualizar os arquivos remotos usados por
seu script. Analisadores e testes adicionais locais posteriores ao aab0d30 não
foram enviados ao workspace do piloto em fila. Nenhum ambiente de colega alterado.

## PR recomendado

Base `experiment/position-mechanism`; head `experiment/position-sae-mechanism`.
Não houve PR, merge ou force-push. Relatório comparativo final permanece pendente.