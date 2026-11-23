# Sensibilidade à ordem no checkpoint multilíngue do Laya

A confirmação prospectiva do caminho demais tokens17 → marcadores18 superou o controle
de ruído nos 400 casos/600 decisões choice do teste: diferença de projeção 0,3754,
IC95% por caso [0,3629; 0,3867]. Isso identifica transmissão sob intervenção, não a
origem inicial da sensibilidade nem uma causa universal em modelos System One.

## Relatórios

- [Conclusão e confirmação no teste](confirmation/REPORT.md)
- [Caminho isolado na descoberta](path/REPORT.md)
- [Grupos de tokens na descoberta](factorial/REPORT.md)
- [Replicação do contraste entre camadas no desenvolvimento](development/REPORT.md)
- [Descoberta e localização inicial](discovery/REPORT.md)

## Protocolos

- [Descoberta](PROTOCOL.md)
- [Replicação no desenvolvimento](DEV_REPLICATION.md)
- [Fatorial por grupo de tokens](FACTORIAL_PROTOCOL.md)
- [Caminho pelo bloco 18](PATH_PROTOCOL.md)
- [Confirmação prospectiva](CONFIRM_PROTOCOL.md)

## Reprodução

As inferências rodam em jobs do Sauron com revisões fixas, FP32, manifestações e gravação
atômica por decisão. Scripts em `scripts/sauron/position*.sauron`, configurações em
`scripts/sauron/jobs/position-*.yaml`. A confirmação usa `position-confirm.yaml`.

As análises são offline, sem carregar modelos:

```bash
python analyze_position.py --input results/position_mechanism/collection_v1/discovery --out analysis/discovery
python analyze_position_factorial.py --input results/position_mechanism/factorial_v1 --marker-reference results/position_mechanism/collection_v1/discovery --out analysis/factorial
python analyze_position_path.py --input results/position_mechanism/path_v1 --factorial-reference results/position_mechanism/factorial_v1 --out analysis/path
python analyze_position_path.py --input results/position_mechanism/path_confirm_v1 --out analysis/confirmation
python -m unittest discover -s tests -p 'test_position*.py' -v
```

Os arquivos volumosos por decisão, ativações e ambientes não são versionados. Relatórios
e tabelas agregadas são versionados. Cópias compactadas verificadas por SHA256 estão
preservadas fora do cluster; hashes e commits constam em cada relatório.
