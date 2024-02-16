# WarCraft-ShortestPath

**WORK IN PROGRESS**

Demonstrating how to embed a combinatorial solver into a deep learning model.

Recreating results from Vlastelica, Marin, et al. "Differentiation of Blackbox Combinatorial Solvers" ICLR 2020 [paper](https://arxiv.org/abs/1912.02175).

This implementation uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [Weights and Biases](https://wandb.ai).

---
### Warcraft Data

The warcraft dataset can be found [here](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S). You need to download the `warcraft_maps.tar.gz` file.

Once it's downloaded (and un-tarred) you can use my PyTorch DataLoader Class:

```python
import torchvision.transforms as transforms
import pylab as pl

from WarCraft import Warcraft12x12

train_loader = Warcraft12x12('path/to/tarball', train=True, transform=transforms.ToTensor())

image, weights, path = next(iter(train_loader))

f, axarr = pl.subplots(1,3, dpi=300)
axarr[0].imshow(np.array(image).transpose(1,2,0))
axarr[1].imshow(weights)
axarr[2].imshow(path)
axarr[1].set_yticks([])  
axarr[2].set_yticks([])  
axarr[0].set_title("Image")
axarr[1].set_title("Vertex Weights")
axarr[2].set_title("Shortest Path")
pl.show()

```

![alt text](https://github.com/as595/WarCraft-ShortestPath/blob/d49c465fab5691eadfd752be189cbd8b9e265ea7/figures/warcraft.png)

---
### Baseline Model

The baseline model uses [PyTorch ResNet18](https://pytorch.org/hub/pytorch_vision_resnet/), which is trained from scratch to predict the shortest path directly from the input WarCraft Images (i.e. not from the weights). This model should train to 100% (training) accuracy in a few epochs. The parameters are specified in a `config` file: `configs/baseline.cfg`. 

To run:

```python
python main.py --config ./configs/baseline.cfg
```

Results: 

<ins>Perfect Match Accuracy</ins>

|  | 12x12 | 18x18 | 24x24 | 30x30 |
| :---:   | :---: | :---: | :---: | :---: |
| Training accuracy | 100.0&pm;0.0%   | 100.0&pm;0.0%   | 95.8&pm;1.6%   | 83.4&pm;1.9%   |
| Test accuracy | 38.7&pm;2.7%   |  6.0&pm;0.4%  | 0.3&pm;0.1%   | 0.0&pm;0.0%   |

<ins>Cost Match Accuracy</ins>

|  | 12x12 | 18x18 | 24x24 | 30x30 |
| :---:   | :---: | :---: | :---: | :---: |
| Training accuracy | 100.0&pm;0.0%   | 100.0&pm;0.0%   | 100.0&pm;0.0%   | 100.0&pm;0.0%   |
| Test accuracy | 38.7&pm;2.7%   |  9.5&pm;0.5%  | 1.4&pm;0.3%   | 0.4&pm;0.2%   |

---
### Combinatorial Model

The combinatorial model uses a modified version of [PyTorch ResNet18](https://pytorch.org/hub/pytorch_vision_resnet/), which outputs a set of learned pixel weights for the input WarCraft Images. These weights are then used as input to a combinatorial solver (Dijkstra) to calculate the shortest path. The predicted shortest path is then compared to the true shortest path in order to calculate the loss. The shortest path calculation using the combinatorial solver step is written as a [custom autograd function for PyTorch](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html) that implements the loss smoothing from [Vlastelica, Marin, et al.](https://arxiv.org/abs/1912.02175). 

*Note: the custom autograd function implemented here uses a static method, as the approach in Vlastelica's original code is now deprecated.*

This model should train to 100% (training) accuracy in a few epochs. The parameters are specified in a `config` file: `configs/combinatorial.cfg`.

To run:

```python
python main.py --config ./configs/combinatorial.cfg
```

Results: 

<ins>Perfect Match Accuracy</ins>

|  | 12x12 | 18x18 | 24x24 | 30x30 |
| :---:   | :---: | :---: | :---: | :---: |
| Training accuracy | 100.0&pm;0.0%   | 99.9&pm;0.1%   | 95.8&pm;1.6%   | 100.0&pm;0.0%   |
| Test accuracy | 85.8&pm;0.5   |  75.0&pm;0.8%  | 0.3&pm;0.1%   | 283   |

<ins>Cost Match Accuracy</ins>

|  | 12x12 | 18x18 | 24x24 | 30x30 |
| :---:   | :---: | :---: | :---: | :---: |
| Training accuracy | 100.0&pm;0.0%   | 100.0&pm;0.0%   | 100.0&pm;0.0%   | 100.0&pm;0.0%   |
| Test accuracy | 97.1&pm;0.4%   |  97.5&pm;0.3%  | 1.4&pm;0.3%   | 0.4&pm;0.2%   |

---
### Training details

Models are trained on 10k input images of each type and tested on a reserved test set of 1k images. Training parameters are as described in [Vlastelica, Marin, et al.](https://arxiv.org/abs/1912.02175) and can be found in the `baseline.cfg` file. *z*-score normalisation is applied to all inputs using the mean and standard deviation of the 10k training images, values for these parameters can be found in the `baseline.cfg` file.

The Baseline training takes approximately 7 (12x12) - 14 (30x30) sec/epoch on an A100, and ~7 min/epoch on a CPU.

The Combinatorial training takes approximately 50 (12x12) - ?? sec/epoch on an A100.
