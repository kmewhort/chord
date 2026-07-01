#!/usr/bin/env bash
#
# Compile the CHORD whitepaper (Markdown) to a nicely typeset PDF.
#
# Format: the source is plain Markdown with LaTeX math ($...$ / $$...$$). We keep
# it that way — it renders on GitHub, diffs cleanly, and is easy to edit — and use
# pandoc + a real LaTeX engine (xelatex) purely to typeset the math and layout for
# the PDF. (This is the same TeX path Quarto/pandoc use for equations, without the
# rest of the Quarto machinery.)
#
# Requirements (install locally if not present):
#   - pandoc         https://pandoc.org/installing.html
#   - a LaTeX engine with xelatex. The lightest is TinyTeX:
#       https://yihui.org/tinytex/   (or `quarto install tinytex`, or texlive-xetex)
#
# Usage:
#   ./compile.sh            # -> CHORD-whitepaper.pdf
#   ./compile.sh out.pdf    # custom output path
#
set -euo pipefail

cd "$(dirname "$0")"

SRC="CHORD-whitepaper.md"
OUT="${1:-CHORD-whitepaper.pdf}"

command -v pandoc >/dev/null 2>&1 || {
  echo "error: pandoc not found on PATH. See https://pandoc.org/installing.html" >&2
  exit 1
}
# Prefer a system xelatex; fall back to tectonic (a self-contained XeTeX engine,
# `brew install tectonic` / `cargo install tectonic`) which also handles the
# fontspec + fvextra preamble and pulls TeX packages on demand. Both are XeTeX so
# the header.tex font/unicode setup renders identically.
if command -v xelatex >/dev/null 2>&1; then
  PDF_ENGINE=xelatex
elif command -v tectonic >/dev/null 2>&1; then
  PDF_ENGINE=tectonic
else
  echo "error: no XeTeX engine on PATH. Install TinyTeX (https://yihui.org/tinytex/)," >&2
  echo "       a texlive distribution providing xelatex, or tectonic (brew install tectonic)." >&2
  exit 1
fi

echo "Rendering $SRC -> $OUT (engine: $PDF_ENGINE) ..."
pandoc "$SRC" \
  --output="$OUT" \
  --pdf-engine="$PDF_ENGINE" \
  --from=markdown+tex_math_dollars+pipe_tables+backtick_code_blocks \
  --toc --toc-depth=2 \
  --include-in-header=header.tex \
  --lua-filter=fit-codeblocks.lua \
  --highlight-style=tango \
  -V documentclass=article \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V linkcolor=RoyalBlue \
  -V urlcolor=RoyalBlue \
  -V toccolor=black \
  -V colorlinks=true \
  -V title-meta="CHORD whitepaper"

echo "Wrote $OUT"
