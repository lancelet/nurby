#!/usr/bin/env bash
#
# This runs `cargo doc`, but supplies the command-line arguments necessary to
# include an HTML header that itself includes MathJax for inline mathematics.
# 
# Examples:
#   $ ./make-docs.sh          # Runs documentation generation
#   $ ./make-docs.sh --open   # Runs doc generation and then opens them

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MATHJAX_HEADER="${SCRIPT_DIR}/mathjax-doc-header.html"
export RUSTDOCFLAGS="--cfg docsrs --html-in-header ${MATHJAX_HEADER}"

cargo doc "$@"