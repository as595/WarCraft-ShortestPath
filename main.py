import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import numpy as np
import random
import csv
import os, sys
from PIL import Image
import psutil

from utils import *
from models import Baseline
from WarCraft import Warcraft12x12

quiet = False

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# extract information from config file:

vars = parse_args()
config_dict, config = parse_config(vars['config'])

model_dir     	= config_dict['top level']['model_dir']
seed       		= config_dict['top level']['seed']

num_epochs   	= config_dict['top level']['num_epochs']
evaluate_every  = config_dict['top level']['evaluate_every']

# optional parameters:
use_ray 					= config_dict['optional']['use_ray']
fast_forward_training 		= config_dict['optional']['fast_forward_training']

save_visualizations	= config_dict['top level']['save_visualizations']

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
# -----------------------------------------------------------------------------
# lightning stuff

	weight_decay = 0.05  # decreased
	batch_size = 32
	dropout = 0.5
	lr_decay = 0.75
	learning_rate = 5e-4

	config = {
			'weight_decay': weight_decay,
			'dropout': dropout,
			'lr_decay': lr_decay,
			'learning_rate': learning_rate,
			'batch_size': batch_size
			}

	# initialise the wandb logger and name your wandb project
	wandb_logger = pl.loggers.WandbLogger(project='warcraft', log_model=True, config=config)

	# wandb will record config
	wandb.init(project='warcraft', config=config)  # args will be ignored by existing logger
	wandb_config = wandb.config

# -----------------------------------------------------------------------------

	os.makedirs(model_dir, exist_ok=True)

	num_cpus = psutil.cpu_count(logical=True)
	#print(num_cpus)
	if use_ray:
		ray.init(
			num_cpus=num_cpus,
			logging_level=WARNING,
			ignore_reinit_error=True,
			redis_max_memory=10 ** 9,
			log_to_driver=False,
			**ray_params)

# -----------------------------------------------------------------------------

	# data transforms
	totensor = transforms.ToTensor()
	normalise= transforms.Normalize((config_dict['data']['datamean'],), (config_dict['data']['datastd'],))

	transform = transforms.Compose([
		totensor, 
		normalise,
		])

	print(config_dict['data']['datadir'])
	train_data = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], train=True, transform=transform)
	
	# split into train:validation
	n_train = 10000
	indices = list(range(len(train_data)))
	train_indices, val_indices = indices[:n_train], indices[n_train:]
	
	train_sampler = Subset(train_data, train_indices)
	valid_sampler = Subset(train_data, val_indices)

	# specify data loaders for training and validation:
	train_loader = torch.utils.data.DataLoader(train_sampler, 
												batch_size=config_dict['training']['batch_size'], 
												shuffle=True, 
												num_workers=num_cpus-1,
												persistent_workers=True
												)

	test_loader = torch.utils.data.DataLoader(valid_sampler, batch_size=config_dict['training']['batch_size'], shuffle=True)

# -----------------------------------------------------------------------------

	
	model = locals()[config_dict['model']['model_name']](
														train_data.metadata["output_features"], 
														train_data.metadata["num_channels"],
														config_dict['optimizer']['lr']
														).to(device)

	#if not quiet:
	#    summary(model, (train_data.metadata["num_channels"], train_data.metadata["input_image_size"], train_data.metadata["input_image_size"]))

# -----------------------------------------------------------------------------

	# pass wandb_logger to the Trainer 
	trainer = pl.Trainer(logger=wandb_logger)

	# train the model
	trainer.fit(model, train_loader)

# -----------------------------------------------------------------------------

	if use_ray:
		ray.shutdown()

