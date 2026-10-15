#!/usr/bin/env bash
set -eo pipefail
mkdir -p references
if [ ! -d "references/semantic-router" ]; then
  git clone --depth 1 https://github.com/aurelio-labs/semantic-router.git references/semantic-router
fi
if [ ! -d "references/setfit" ]; then
  git clone --depth 1 https://github.com/huggingface/setfit.git references/setfit
fi
