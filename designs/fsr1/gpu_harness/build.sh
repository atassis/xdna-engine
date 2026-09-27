#!/bin/bash
# Build the standalone Vulkan compute harness + compile its shaders.
set -euo pipefail
cd "$(dirname "$0")"
glslangValidator -V shaders/easu.comp -o shaders/easu.spv
glslangValidator -V shaders/rcas.comp -o shaders/rcas.spv
gcc -O2 -o harness harness.c -lvulkan -lm
echo "built: harness, shaders/easu.spv, shaders/rcas.spv"
