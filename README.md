# efficientnet-pytorch-scratch
## Description
I rewrote efficientnet from scratch for learning purposes. For not hobbyist, actual implementation check:
1. https://github.com/lukemelas/EfficientNet-PyTorch (my inspo)
2. https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

## Results (EfficientNet-B0 on Imagenette dataset)
![EfficientNet-B0 on Imagenette dataset](/EfficientNet-B0_Imagenette_20_epoch.png)

## TODOs that I will probably not do
1. Normal resolution scaling
2. Better results visualization
3. More tests and more metrics
4. Better script.ipynb that is easy to follow

## Citation

```bibtex
@misc{tan2020efficientnetrethinkingmodelscaling,
      title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks}, 
      author={Mingxing Tan and Quoc V. Le},
      year={2020},
      eprint={1905.11946},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1905.11946}, 
}