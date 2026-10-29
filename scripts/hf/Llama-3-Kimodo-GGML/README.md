---
license: other
library_name: ggml
tags: [gguf, ggml, llama-3, text-embeddings]
---

# Llama-3-Kimodo-GGML

Native GGML/GGUF text encoder used by Kimodo. This is the
reusable LLM2Vec encoder only; download a matching Kimodo diffusion model
separately, for example
[`Kimodo-SMPLX-RP-v1-GGML`](https://huggingface.co/LocalAI-io/Kimodo-SMPLX-RP-v1-GGML).

From a kimodo.cpp checkout with the Hugging Face CLI installed, install both with:

```sh
scripts/download_gguf_weights.sh --output "$PWD"
```

The downloader installs the recommended Q8_0 variant by default. Select a
different one with `--text-quantization bf16|q8_0|q6_k|q5_k|q4_k|q4_k_m`.
Every variant is one monolithic weight GGUF and uses the same `tokenizer.gguf`.
Kimodo reads individual tensor ranges from the monolith, so streaming one layer
at a time does not require separate per-layer files.

| Variant | Weight file | Size (GB) | Intended use |
| --- | --- | ---: | --- |
| BF16 | `Llama-3-Kimodo-BF16.gguf` | 15.18 | Reference |
| Q8_0 | `Llama-3-Kimodo-Q8_0.gguf` | 8.14 | Recommended |
| Q6_K | `Llama-3-Kimodo-Q6_K.gguf` | 6.32 | Experimental |
| Q5_K | `Llama-3-Kimodo-Q5_K.gguf` | 5.33 | Experimental |
| Q4_K | `Llama-3-Kimodo-Q4_K.gguf` | 4.39 | Experimental |
| Q4_K_M | `Llama-3-Kimodo-Q4_K_M.gguf` | 5.06 | Preferred low-bit experiment |

The published distribution contains only monolithic weight variants and one
shared tokenizer. Split component files are local conversion intermediates.

## Provenance and licence

The bundle is converted from Meta Llama-3-8B-Instruct and the MIT-licensed
McGill LLM2Vec MNTP and supervised adapters. **Built with Meta Llama 3.**

`LICENSE-META-LLAMA-3.txt` and `NOTICE` accompany this distribution. Review
the [Meta Llama 3 Community License](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
before use or redistribution. `MANIFEST.json` records the exact source commits
and SHA-256 of every published artifact, plus the weight/tokenizer mapping for
each quantization.
