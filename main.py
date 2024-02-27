import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import numpy as np
import random
import csv
import os, sys
from PIL import Image
import psutil

from utils import *
from models import Baseline, Combinatorial
from WarCraft import Warcraft12x12, Warcraft18x18, Warcraft24x24, Warcraft30x30

import platform

if platform.system()=='Darwin':
	os.environ["GLOO_SOCKET_IFNAME"] = "en0"

quiet = False

torch.set_float32_matmul_precision('medium')


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if torch.cuda.is_available():
	device='cuda'
#elif torch.backends.mps.is_built():
#	device='mps'
else:
	device='cpu'

# -----------------------------------------------------------------------------

if __name__ == "__main__":

# -----------------------------------------------------------------------------
# extract config info & set the random seed

	args = parse_args()
	
	random_state = args['random_state']
	pl.seed_everything(random_state)

	config_dict, config = parse_config(args['config'])
	model_dir = config_dict['top level']['model_dir']

# -----------------------------------------------------------------------------
# lightning stuff

	weight_decay = config_dict['training']['l1_regconst'] 
	batch_size = config_dict['training']['batch_size']
	learning_rate = config_dict['optimizer']['lr']

	config = {
			'weight_decay': weight_decay,
			'learning_rate': learning_rate,
			'batch_size': batch_size,
			'seed': random_state
			}

	# initialise the wandb logger and name your wandb project
	wandb_logger = pl.loggers.WandbLogger(project='warcraft', log_model=True, config=config)
	wandb_config = wandb.config

# -----------------------------------------------------------------------------

	os.makedirs(model_dir, exist_ok=True)
	num_cpus = psutil.cpu_count(logical=True)
	
# -----------------------------------------------------------------------------

	# data transforms
	totensor = transforms.ToTensor()
	normalise= transforms.Normalize(config_dict['data']['datamean'], config_dict['data']['datastd'])
	
	transform = transforms.Compose([
		totensor, 
		normalise,
		])

	print("Data: {}".format(config_dict['data']['dataset']))
	train_data = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], train=True, transform=transform)
	test_data = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], train=False, transform=transform)
	
	# take 10k samples for training; 1k samples for test
	n_train = config_dict['data']['ntrain']
	n_test = config_dict['data']['ntest']
	indices = list(range(len(train_data)))
	
	train_sampler = Subset(train_data, indices[:n_train]) # 10k samples
	test_sampler = Subset(test_data, indices[:n_test])    # 1k samples

	# specify data loaders for training and validation:
	train_loader = torch.utils.data.DataLoader(train_sampler, 
												batch_size=config_dict['training']['batch_size'], 
												shuffle=True, 
												num_workers=num_cpus-1,
												persistent_workers=True
												)

	test_loader = torch.utils.data.DataLoader(test_sampler, 
												batch_size=len(test_sampler), 
												shuffle=False, 
												num_workers=num_cpus-1,
												persistent_workers=True
												)

# -----------------------------------------------------------------------------

	print("Model: {} ({})".format(config_dict['model']['model_name'], device))
	model = locals()[config_dict['model']['model_name']](
														train_data.metadata["output_features"], 
														train_data.metadata["num_channels"],
														config_dict['optimizer']['lr'],
														config_dict['training']['l1_regconst'],
														config_dict['training']['lambda_val'],
														config_dict['training']['neighbourhood_fn']
														).to(device)

	#if not quiet:
	#    summary(model, (train_data.metadata["num_channels"], train_data.metadata["input_image_size"], train_data.metadata["input_image_size"]))

# -----------------------------------------------------------------------------


	if config_dict['model']['model_name']=='Combinatorial':
		ddp_strategy = 'ddp_find_unused_parameters_true' # strategy flag required for custom autograd fnc
	else:
		ddp_strategy = 'ddp' # default

	lr_monitor = LearningRateMonitor(logging_interval='epoch')

	trainer = pl.Trainer(max_epochs=config_dict['training']['num_epochs'], 
						 strategy=ddp_strategy,
						 callbacks=[lr_monitor],
						 num_sanity_val_steps=0, # 0 : turn off validation sanity check  
						 accelerator=device, 
						 devices=1,
						 logger=wandb_logger) 

	# train the model
	trainer.fit(model, train_loader)
	
# -----------------------------------------------------------------------------


	trainer.test(model, test_loader, ckpt_path=None) # test final epoch model

# -----------------------------------------------------------------------------

	wandb.finish()

