## Configuration Files

```python
[top level]
model_dir: '/share/nas2/ascaife/warcraft/models/'    # directory to save trained model weights

[data]
dataset: 'Warcraft12x12'                             # data set  
datadir: '/share/nas2/ascaife/warcraft/data/'        # path to data 
datamean: 0.                                         # data mean for normalisation transform
datastd: 1.                                          # data std for normalisation transform
ntrain: 10000                                        # number of training inputs
ntest: 1000                                          # number of test inputs

[training]
num_epochs: 150                                      # number of training epochs
batch_size: 70                                       # batch size
l1_regconst: 0.0                                     # co-efficient for L1 regularisation
lambda_val: 20.0                                     # smoothing parameter for loss approximation
neighbourhood_fn: '8-grid'                           # neighbourhood function for dijkstra [if used]

[optimizer]
optimizer_name: 'Adam'                               # optimizer
lr: 0.0005                                           # learning rate

[model]
model_name: 'Baseline'                               # type of model ['Baseline', 'Combinatorial']
arch: 'ResNet18'                                     # CNN base architecture
```
