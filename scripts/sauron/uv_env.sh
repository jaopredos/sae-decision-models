#!/bin/bash
# uv_env.sh — ambiente uv dentro do container do Sauron. Use com `source` no corpo do job.
#
# Por que assim (ver o guia do Sauron e o env.sh dos jobs de exemplo):
#   * a imagem pytorch/pytorch não traz uv nem ensurepip, e o Python dela é PEP 668. O uv entra
#     por `pip install --user` num PYTHONUSERBASE dentro de /workspace/cache, como no env.sh;
#   * o venv do projeto é do uv (`uv sync --frozen`, a partir do uv.lock versionado), isolado da
#     imagem: o torch é o 2.15.0+cu132 do lock, o mesmo do Mac, e não o da imagem. Ele fica em
#     .venv-sauron, na pasta do projeto, e é reaproveitado entre jobs;
#   * HF_HOME/TORCH_HOME NÃO são tocados: o guia pede para não trocar, e o Sauron já os aponta
#     para o cache do nó (/cache/huggingface). Só o cache do uv vai para /workspace/cache/uv.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${PWD}}"
# o Sauron monta ~/workspace como /workspace e o job começa em /workspace/<projeto>
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"
CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_ROOT}/cache}"
UV_VERSION="0.13.7"   # a mesma do Mac

export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_PYTHON_INSTALL_DIR="${CACHE_ROOT}/uv-python"
export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.venv-sauron"
# cache e venv podem estar em sistemas de arquivos diferentes; hardlink falharia
export UV_LINK_MODE=copy
export PYTHONUSERBASE="${CACHE_ROOT}/pyuser-uv"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"

log() { echo "[probe $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "[probe] ERRO: $*" >&2; exit 1; }

if ! command -v uv >/dev/null 2>&1 || [[ "$(uv --version | awk '{print $2}')" != "${UV_VERSION}" ]]; then
    log "instalando uv ${UV_VERSION} em ${PYTHONUSERBASE}"
    mkdir -p "${PYTHONUSERBASE}"
    # --user exige o user site ligado, então PYTHONNOUSERSITE só é exportado depois
    PIP_BREAK_SYSTEM_PACKAGES=1 PIP_CACHE_DIR="${PIP_CACHE_DIR:-/cache/pip}" \
        python -m pip install --quiet --user "uv==${UV_VERSION}"
fi
export PYTHONNOUSERSITE=1
log "uv: $(uv --version)"

log "uv sync --frozen -> ${UV_PROJECT_ENVIRONMENT}"
uv sync --frozen

# Qual commit está rodando. A imagem não traz o binário do git, então o SHA é lido do .git
# (HEAD -> ref solto ou packed-refs), como no env.sh dos exemplos.
_commit_sem_git() {
    local g="${PROJECT_ROOT}/.git" head ref linha sha
    [[ -f "${g}/HEAD" ]] || return 1
    head=$(<"${g}/HEAD")
    if [[ "${head}" != ref:* ]]; then echo "${head:0:12} (HEAD solto)"; return 0; fi
    ref="${head#ref: }"
    if [[ -f "${g}/${ref}" ]]; then sha=$(<"${g}/${ref}"); echo "${sha:0:12} (${ref#refs/heads/})"; return 0; fi
    if [[ -f "${g}/packed-refs" ]]; then
        while IFS= read -r linha || [[ -n "${linha}" ]]; do
            [[ "${linha}" == "${linha#\#}" && "${linha}" == *" ${ref}" ]] || continue
            sha="${linha%% *}"; echo "${sha:0:12} (${ref#refs/heads/}, packed)"; return 0
        done < "${g}/packed-refs"
    fi
    return 1
}
log "commit         : $(_commit_sem_git || echo indisponível)"
log "projeto        : ${PROJECT_ROOT}"
log "HF_HOME        : ${HF_HOME:-<não definido>} (do Sauron, não alterado)"
log "UV_CACHE_DIR   : ${UV_CACHE_DIR}"
log "variáveis do Sauron: $(env | grep -E '^SAURON_' | tr '\n' ' ' || true)"

uv run --frozen python - <<'PYEOF'
import sys, torch
if not torch.cuda.is_available():
    print("[ERRO] torch.cuda.is_available() == False:", torch.__version__, "cuda", torch.version.cuda)
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
a = torch.randn(1024, 1024, device="cuda")
ok = torch.allclose((a @ a).cpu(), a.cpu() @ a.cpu(), atol=1e-2, rtol=1e-3)
print(f"[OK] python {sys.version.split()[0]} | torch {torch.__version__} (CUDA {torch.version.cuda}) | "
      f"{p.name} | {p.total_memory / 2**30:.0f} GiB | cc {p.major}.{p.minor} | matmul ok={ok}")
if not ok:
    sys.exit(1)
PYEOF

mkdir -p results/probe_typed
uv pip freeze --python "${UV_PROJECT_ENVIRONMENT}/bin/python" > results/probe_typed/pip_freeze.txt

# pico de RAM do cgroup ao sair (inclusive em falha), para calibrar `memory:` no yaml
relatorio_memoria() {
    local f pico=""
    for f in /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
        [[ -r "${f}" ]] && { pico="$(cat "${f}" 2>/dev/null)"; break; }
    done
    [[ "${pico}" =~ ^[0-9]+$ ]] && log "RAM do job: pico $(awk -v p="${pico}" 'BEGIN{printf "%.1f", p/1024^3}') GiB" \
        || log "RAM do job: o cgroup não expôs o pico"
}
trap relatorio_memoria EXIT
