<div align="center" style="line-height: 1;">
<h1>LLM-jp-4-VL</h1>


  |
  <a href="https://huggingface.co/llm-jp/llm-jp-4-vl-9b" target="_blank">🤗 Model</a>
  &nbsp;|
  <a href="https://llm-jp.nii.ac.jp/blog/llm-jp-4-vl-9b/" target="_blank">📄 Blog</a>
  &nbsp;|
  <a href="https://github.com/llm-jp/llm-jp-4-vl" target="_blank">🧑‍💻 Code</a>
  &nbsp;|

  <br/>
</div>

LLM-jp-4-VL is a series of vision-language models developed by LLM-jp.

This repository provides sample code for running inference with the LLM-jp-4-VL models.


## Models

| Model | Type | Inference |
|---|---|---|
| [`llm-jp/llm-jp-4-vl-9b`](https://huggingface.co/llm-jp/llm-jp-4-vl-9b) | reasoning (Harmony: analysis + final channel) | Transformers, SGLang |
| [`llm-jp/llm-jp-4-vl-9B-beta`](https://huggingface.co/llm-jp/llm-jp-4-vl-9B-beta) | non-reasoning | Transformers |

## Inference with Transformers

```bash
git clone https://github.com/llm-jp/llm-jp-4-vl && cd llm-jp-4-vl
uv sync                              # torch 2.8, transformers 4.57, flash-attn 2.8.3 (Python >= 3.13)
uv run python cookbooks/basic.py     # text-only, single-image, multi-image and multi-turn examples
```

[`cookbooks/basic.py`](cookbooks/basic.py) loads the model with `trust_remote_code=True`, applies the chat template (`reasoning_effort`: `low` / `medium` / `high`) and keeps only the final channel of the answer. Switch models with `model_id` at the top of the file.

## Inference with SGLang (OpenAI-compatible server)

For evaluation or many requests, serve the model with [SGLang](https://github.com/sgl-project/sglang). SGLang has no built-in support for LLM-jp-VL; [`inference/sglang/`](inference/sglang/) adds it as a plugin that is loaded at startup, without modifying SGLang:

```bash
bash inference/sglang/install.sh                                         # inference/sglang/.venv: SGLang 0.5.23 (Python 3.11)
bash inference/sglang/launch.sh llm-jp/llm-jp-4-vl-9b 30000 --dp-size 8  # MODEL PORT [sglang.launch_server args]
python inference/sglang/client.py --image assets/Shiba_inu.jpg --prompt "この画像を説明してください。"
```

- `client.py` (standard library only) sends an OpenAI chat request with `chat_template_kwargs` (`reasoning_effort`, and `conversation_start_date` for templates with a date line) and `skip_special_tokens: false`, then splits the Harmony output into reasoning and answer. Any OpenAI-compatible client works the same way.
- The plugin (`inference/sglang/llmjpvl_sglang/`) provides the model (SigLIP2 vision tower → pixel shuffle → MLP → Llama), the image processor (the same tiling and `<|image_pad|>` expansion as `processing_llmjpvl.py`) and two fixes to SGLang's chat path: it keeps images for chat templates that render plain-string content, and it removes the spaces the tokenizer's decode adds after special tokens, so that the prompt SGLang encodes is the one the template rendered.
- Use the cuDNN bundled with torch: loading an older system cuDNN (e.g. an HPC `cudnn` module) breaks the SigLIP convolution (`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`).

SGLang matches the Transformers implementation (greedy, `reasoning_effort=medium`, one attempt; scores from [simple-evals-mm](https://github.com/llm-jp/simple-evals-mm), GPT-6 Luna as judge):

| `llm-jp/llm-jp-4-vl-9b` | Prompt tokens equal | Transformers | SGLang | Same verdict |
|---|---:|---:|---:|---:|
| CVQA-JA-Refined | 200/200 | 68.0 | 69.0 | 178/200 |
| JGraphQA-Refined | 196/196 | 90.8 | 91.3 | 187/196 |
| CC-OCR-JA-Refined | 145/145 | 69.7 | 68.3 | 141/145 |

The encoded prompts (text and image tokens) are identical; the remaining differences come from bf16 kernels in long greedy generations (output budget 65,536 tokens).

## Evaluation Reproduction
To reproduce the evaluation results reported in our blog post, please refer to [simple-evals-mm](https://github.com/llm-jp/simple-evals-mm), our VLM evaluation framework.

## LICENSE
This code is released under the Apache 2.0 license.

## Citation
If you find our work useful, please consider citing the following papers:
```bibtex
@misc{sugiura2026jaglebuildinglargescalejapanese,
      title={Jagle: Building a Large-Scale Japanese Multimodal Post-Training Dataset for Vision-Language Models},
      author={Issa Sugiura and Keito Sasagawa and Keisuke Nakao and Koki Maeda and Ziqi Yin and Zhishen Yang and Shuhei Kurita and Yusuke Oda and Ryoko Tokuhisa and Daisuke Kawahara and Naoaki Okazaki},
      year={2026},
      eprint={2604.02048},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.02048},
}

@misc{sugiura2026jammevalrefinedcollectionjapanese,
      title={JAMMEval: A Refined Collection of Japanese Benchmarks for Reliable VLM Evaluation},
      author={Issa Sugiura and Koki Maeda and Shuhei Kurita and Yusuke Oda and Daisuke Kawahara and Naoaki Okazaki},
      year={2026},
      eprint={2604.00909},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.00909},
}
```
