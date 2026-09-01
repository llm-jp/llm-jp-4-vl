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


## Usage
Install dependencies:
```bash
uv sync
```

See [`cookbooks/basic.py`](cookbooks/basic.py) for a runnable inference example covering text-only, single-image, multi-image, and multi-turn inputs. It works with both `llm-jp/llm-jp-4-vl-9b` (reasoning) and `llm-jp/llm-jp-4-vl-9B-beta` (non-reasoning); switch models by editing `model_id` at the top of the file.

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
