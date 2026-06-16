#!/usr/bin/env bash
# Regenerate all figure data and compile the figures PDF.
#
# Usage:
#   bash figures/scripts/build.sh [--no-data] [--no-pdf] [--latexmk]
#
# --no-data   Skip the Python data-generation step.
# --no-pdf    Regenerate .dat files but don't compile.
# --latexmk   Try latexmk instead of plain 3-pass xelatex. Default is plain
#             xelatex because Rocky 10's minimal Perl doesn't ship
#             Time::HiRes (XS module, can't be shimmed with a .pm file),
#             so latexmk errors out at startup here.
#
# Output:
#   figures/data/*.dat         (regenerated unless --no-data)
#   figures/sageeval-figures.pdf (regenerated unless --no-pdf, 7-page PDF)
#
# HPC notes:
#   - user-local TeX Live at /scratch/ah7660/texlive (scheme-medium)
#   - pgfplots added manually to ~/texmf from CTAN (tlmgr was too slow)
#   - a File::Copy shim at ~/perl5/lib/perl5 is loaded via PERL5LIB in
#     case latexmk is requested; Time::HiRes is still missing, so --latexmk
#     will fail until somebody installs it (cpanm Time::HiRes user-local).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FIG_DIR="$ROOT/figures"
TEXLIVE_BIN="/scratch/ah7660/texlive/bin/x86_64-linux"
PERL_SHIM="$HOME/perl5/lib/perl5"

GEN=1
PDF=1
USE_LATEXMK=0
for arg in "$@"; do
  case "$arg" in
    --no-data) GEN=0 ;;
    --no-pdf)  PDF=0 ;;
    --latexmk) USE_LATEXMK=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$GEN" == 1 ]]; then
  echo "[build] regenerating .dat files"
  (cd "$ROOT" && uv run python figures/scripts/generate_figure_data.py)
fi

if [[ "$PDF" == 0 ]]; then
  echo "[build] --no-pdf set, done"
  exit 0
fi

if [[ -d "$TEXLIVE_BIN" ]]; then
  export PATH="$TEXLIVE_BIN:$PATH"
fi
if [[ -d "$PERL_SHIM" ]]; then
  export PERL5LIB="$PERL_SHIM${PERL5LIB:+:$PERL5LIB}"
fi

cd "$FIG_DIR"

if [[ "$USE_LATEXMK" == 1 ]] && command -v latexmk >/dev/null 2>&1; then
  echo "[build] running latexmk -xelatex"
  latexmk -xelatex -interaction=nonstopmode -halt-on-error sageeval-figures.tex
else
  echo "[build] running xelatex x3 (plain mode)"
  for i in 1 2 3; do
    echo "[build] pass $i"
    xelatex -interaction=nonstopmode -halt-on-error sageeval-figures.tex
  done
fi

echo "[build] done → $FIG_DIR/sageeval-figures.pdf"
