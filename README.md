# llama-diffusion-gemma

> **Fix for:** `500 Internal Server Error: unknown model architecture: 'diffusion-gemma'`

[DiffusionGemma](https://deepmind.google/models/diffusion-gemma/) uses a block-diffusion architecture not yet merged into llama.cpp mainline. This repo tracks the custom build from [PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423) that adds support.

## Why This Exists

The standard `llama-server` (Homebrew `llama.cpp`) crashes with:
```
error loading model: unknown model architecture: 'diffusion-gemma'
```
DiffusionGemma requires the special `llama-diffusion-cli` runner from PR #24423.

## Build From Source

```bash
git clone --depth=1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch origin pull/24423/head:diffusion-gemma
git checkout diffusion-gemma
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu) --target llama-diffusion-cli
```

Binary will be at `build/bin/llama-diffusion-cli`.

## Usage

```bash
./build/bin/llama-diffusion-cli -m /path/to/diffusiongemma.gguf -p "Your prompt here"
```

> **Note:** Use `llama-diffusion-cli`, not `llama-server`. The diffusion architecture is not autoregressive and requires this dedicated runner.

## Status

- [ ] PR #24423 merged into llama.cpp main
- [ ] Homebrew `llama.cpp` updated with support

Track merge status: https://github.com/ggml-org/llama.cpp/pull/24423
