from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# -----------------------------------------------------------------------------

class Baseline(pl.LightningModule):

	"""lighning module to reproduce resnet18 baseline"""

	def __init__(self, out_features, in_channels, lr):

		super().__init__()
        
		self.encoder = torchvision.models.resnet18(pretrained=False, num_classes=out_features)
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
		accuracy = (z_pred.round() * flat_target).sum() / flat_target.sum()

		suggested_path = z_pred.view(z_train.shape).round()
		last_suggestion = {"vertex_costs": None, "suggested_path": suggested_path}

		self.log("train_loss", loss)
		self.log("train_accuracy", accuracy)

		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
		return optimizer



# -----------------------------------------------------------------------------

class CombResNet18(nn.Module):

	def __init__(self, out_features, in_channels):
		super().__init__()
		self.resnet_model = torchvision.models.resnet18(pretrained=False, num_classes=out_features)
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

class Dijkstra():
	def __init__(self, l1_regconst, lambda_val, **kwargs):
		super().__init__(**kwargs)
		self.l1_regconst = l1_regconst
		self.lambda_val = lambda_val
		self.solver = ShortestPath(lambda_val=lambda_val, neighbourhood_fn=self.neighbourhood_fn)
		self.loss_fn = HammingLoss()

		print("META:", self.metadata)
	def build_model(self, model_name, arch_params):
		self.model = get_model(
			model_name, out_features=self.metadata["output_features"], in_channels=self.metadata["num_channels"], arch_params=arch_params
			)

	def forward_pass(self, input, true_shortest_paths, train, i):
		output = self.model(input)
		# make grid weights positive
		output = torch.abs(output)
		weights = output.reshape(-1, output.shape[-1], output.shape[-1])

		if i == 0 and not train:
			print(output[0])
		assert len(weights.shape) == 3, f"{str(weights.shape)}"
		shortest_paths = self.solver(weights)

		loss = self.loss_fn(shortest_paths, true_shortest_paths)

		logger = self.train_logger if train else self.val_logger

		last_suggestion = {
				"suggested_weights": weights,
				"suggested_path": shortest_paths
				}

		accuracy = (torch.abs(shortest_paths - true_shortest_paths) < 0.5).to(torch.float32).mean()
		extra_loss = self.l1_regconst * torch.mean(output)
		loss += extra_loss

		return loss, accuracy, last_suggestion