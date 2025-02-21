import torch
import torch.nn as nn


class PressureNet(nn.Module):
	def __init__(self, in_channels=3, num_classes=88, use_relu=False):
		super(PressureNet, self).__init__()

		self.features = nn.Sequential(
			nn.Conv2d(in_channels, 192, kernel_size=7, stride=2, padding=3),
			nn.ReLU(inplace=True) if use_relu else nn.Tanh(),
			nn.Dropout(p=0.1),
			nn.MaxPool2d(3, stride=2),
			nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
			nn.ReLU(inplace=True) if use_relu else nn.Tanh(),
			nn.Dropout(p=0.1),
			nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),
			nn.ReLU(inplace=True) if use_relu else nn.Tanh(),
			nn.Dropout(p=0.1),
			nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
			nn.ReLU(inplace=True) if use_relu else nn.Tanh(),
			nn.Dropout(p=0.1),
		)

		self.output_layer = nn.Sequential(
			nn.Linear(67200, num_classes)
		)

	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.output_layer(x)
		return x