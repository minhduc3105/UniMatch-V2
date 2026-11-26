#!/usr/bin/env bash
set -euo pipefail

output="${1:-pretrained/dinov2_small.pth}"
mkdir -p "$(dirname "$output")"
curl --fail --location --retry 3 \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
    --output "$output"
echo "Saved $output"
