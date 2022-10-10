# Mixture of Attention Heads

This repository contains the code used for WMT14 translation experiments in 
[Mixture of Attention Heads: Selecting Attention Heads Per Token](https://arxiv.org/) paper.
<!-- If you use this code or our results in your research, we'd appreciate if you cite our paper as following:

```
@article{shen2018ordered,
  title={Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks},
  author={Shen, Yikang and Tan, Shawn and Sordoni, Alessandro and Courville, Aaron},
  journal={arXiv preprint arXiv:1810.09536},
  year={2018}
}
``` -->

## Software Requirements
Python 3, fairseq and PyTorch are required for the current codebase.

## Steps

1. Install PyTorch and fairseq

2. Generate WMT14 translation dataset with [Transformer Clinic](https://github.com/LiyuanLucasLiu/Transformer-Clinic).

3. Scripts and commands

  	+  Train Language Modeling
  	```sh run.sh /path/to/your/data```

  	+ Test Unsupervised Parsing
    ```sh test.sh /path/to/checkpoint```
    
    The default setting in MoA achieves a BLEU of approximately `28.4` on WMT14 EN-DE test set.