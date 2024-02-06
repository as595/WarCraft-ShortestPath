# WarCraft-ShortestPath

Recreating results from Vlastelica, Marin, et al. "Differentiation of Blackbox Combinatorial Solvers" ICLR 2020 [paper](https://arxiv.org/abs/1912.02175).

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

![alt text](http://url/to/img.png)
