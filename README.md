# WarCraft-ShortestPath

**WORK IN PROGRESS**

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

The baseline model uses ResNet18, which is trained to predict the shortest path directly from the input WarCraft Images (i.e. not from the weights). This model should train to 100% (training) accuracy in a few epochs. The parameters are specified in a `config` file: `configs/baseline.cfg`.

To run:

```python
python main.py --config ./configs/baseline.cfg
```

---
### Combinatorial Model
