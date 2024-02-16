from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from dijkstra import ShortestPath, HammingLoss
from utils import exact_match_accuracy, exact_cost_accuracy


# -----------------------------------------------------------------------------

class Baseline(pl.LightningModule):

	"""lightning module to reproduce resnet18 baseline"""

	def __init__(self, out_features, in_channels, lr, l1_regconst, lambda_val, neighbourhood_fn):

		super().__init__()
        
		#l1_regconst, lambda_val : not used

		self.encoder = torchvision.models.resnet18(weights=None, num_classes=out_features)
		del self.encoder.conv1
		self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
		self.lr = lr

	def training_step(self, batch, batch_idx):
        
		x_train, y_train, z_train = batch
        
		z_pred = self.encoder(x_train)
		z_pred = torch.sigmoid(z_pred)

		flat_target = z_train.view(z_train.size()[0], -1)
        
		criterion = torch.nn.BCELoss()
		loss = criterion(z_pred, flat_target.to(dtype=torch.float)).mean()
		accuracy = (z_pred.round() * flat_target).sum() / flat_target.sum() # perfect match accuracy

		suggested_path = z_pred.view(z_train.shape).round()
		last_suggestion = {"vertex_costs": None, "suggested_path": suggested_path}

		self.log("train_loss", loss)
		self.log("train_accuracy", accuracy)

		return loss

	def test_step(self, batch, batch_idx):

		x_test, y_test, z_test = batch
        
		suggested_paths = self.encoder(x_test)
		suggested_paths = torch.sigmoid(suggested_paths).round()

		true_paths = z_test.view(z_test.size()[0], -1)

		accuracy = exact_match_accuracy(true_paths, suggested_paths)
		self.log('exact match accuracy [test]', accuracy)

		true_weights = y_test.view(y_test.size()[0], -1)
		accuracy = exact_cost_accuracy(true_paths, suggested_paths, true_weights)
		self.log('exact cost accuracy [test]', accuracy)

		return


	def configure_optimizers(self):

		# should update this at some point to take optimizer from config file
		optimizer    = torch.optim.Adam(self.parameters(), lr=self.lr)

		# learning rate steps specified in https://arxiv.org/pdf/1912.02175.pdf (A.3.1)
		lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40], gamma=0.1)

		return [optimizer], [lr_scheduler]


# -----------------------------------------------------------------------------

class Combinatorial(pl.LightningModule):

	"""lightning module to reproduce resnet18+dijkstra baseline"""

	def __init__(self, out_features, in_channels, lr, l1_regconst, lambda_val, neighbourhood_fn):

		super().__init__()
        
		self.neighbourhood_fn = neighbourhood_fn
		self.lambda_val = lambda_val
		self.l1_regconst = l1_regconst

		self.encoder = CombResNet18(out_features, in_channels)
		self.solver = ShortestPath.apply

		self.lr = lr

	def training_step(self, batch, batch_idx):
        
		x_train, true_weights, true_shortest_paths = batch
        
        # get the output from the CNN:
		output = self.encoder(x_train)
		output = torch.abs(output)

		weights = output.reshape(-1, output.shape[-1], output.shape[-1]) # reshape to match the path maps
		assert len(weights.shape) == 3, f"{str(weights.shape)}" # double check dimensions
		
		# pass the predicted weights through the dijkstra algorithm:
		predicted_paths = self.solver(weights, self.lambda_val, self.neighbourhood_fn) # only positional arguments allowed (no keywords)
		
		# calculate the Hammingloss
		criterion = HammingLoss()
		loss = criterion(predicted_paths, true_shortest_paths)
		
		# calculate the regularisation:
		l1reg = self.l1_regconst * torch.mean(output)
		loss += l1reg
		
		# calculate the accuracy:
		accuracy = (torch.abs(predicted_paths - true_shortest_paths) < 0.5).to(torch.float32).mean()

		last_suggestion = {
            "suggested_weights": weights,
            "suggested_path": predicted_paths
        }

		self.log("train_loss", loss)
		self.log("train_accuracy", accuracy)

		return loss


	def test_step(self, batch, batch_idx):

		x_test, y_test, z_test = batch
        
		# get the output from the CNN:
		output = self.encoder(x_test)
		output = torch.abs(output)

		weights = output.reshape(-1, output.shape[-1], output.shape[-1]) # reshape to match the path maps
		assert len(weights.shape) == 3, f"{str(weights.shape)}" # double check dimensions
		
		# pass the predicted weights through the dijkstra algorithm:
		suggested_paths = self.solver(weights, self.lambda_val, self.neighbourhood_fn) # only positional arguments allowed (no keywords)
        
		true_paths = z_test.view(z_test.size()[0], -1)
		print(true_paths.shape, suggested_paths.shape)
		
		accuracy = exact_match_accuracy(true_paths, suggested_paths)
		self.log('exact match accuracy [test]', accuracy)

		true_weights = y_test.view(y_test.size()[0], -1)
		accuracy = exact_cost_accuracy(true_paths, suggested_paths, true_weights)
		self.log('exact cost accuracy [test]', accuracy)

		return


	def configure_optimizers(self):

		# should update this at some point to take optimizer from config file
		optimizer    = torch.optim.Adam(self.parameters(), lr=self.lr)

		# learning rate steps specified in https://arxiv.org/pdf/1912.02175.pdf (A.3.1)
		lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40], gamma=0.1)

		return [optimizer], [lr_scheduler]


# -----------------------------------------------------------------------------

class CombResNet18(nn.Module):

	def __init__(self, out_features, in_channels):
		super().__init__()
		self.resnet_model = torchvision.models.resnet18(weights=None, num_classes=out_features)
		del self.resnet_model.conv1
		self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
		output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
		self.pool = nn.AdaptiveMaxPool2d(output_shape)
		#self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)

	def forward(self, x):
		x = self.resnet_model.conv1(x)
		x = self.resnet_model.bn1(x)
		x = self.resnet_model.relu(x)
		x = self.resnet_model.maxpool(x)
		x = self.resnet_model.layer1(x)
		#x = self.resnet_model.layer2(x)
		#x = self.resnet_model.layer3(x)
		#x = self.last_conv(x)
		x = self.pool(x)
		x = x.mean(dim=1)
		return x


# -----------------------------------------------------------------------------
