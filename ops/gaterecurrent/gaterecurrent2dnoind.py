# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import gaterecurrent2dnoind_cuda

class GateRecurrent2dnoindFunction(Function):
		
	@staticmethod
	@torch.cuda.amp.custom_fwd
	def forward(ctx, X, B, G1, G2, G3, items_each_chunk):
		num, channels, height, width = X.size()
		output = torch.zeros(num, channels, height, width, device=X.device, dtype=X.dtype)

		ctx.hiddensize = X.size()
		ctx.items_each_chunk = items_each_chunk

		if X.is_cuda:
			gaterecurrent2dnoind_cuda.forward(items_each_chunk, X, B, G1, G2, G3, output)
		else:
			raise NotImplementedError

		ctx.save_for_backward(X, B, G1, G2, G3, output)

		return output

	@staticmethod
	@torch.cuda.amp.custom_bwd
	@once_differentiable
	def backward(ctx, grad_output):
		grad_output = grad_output.contiguous()
		hiddensize = ctx.hiddensize
		items_each_chunk = ctx.items_each_chunk
		X, B, G1, G2, G3, output = ctx.saved_tensors
		
		assert (hiddensize is not None and grad_output.is_cuda)
		num, channels, height, width = hiddensize

		grad_X = torch.zeros_like(X)
		grad_B = torch.zeros_like(B)
		grad_G1 = torch.zeros_like(G1)
		grad_G2 = torch.zeros_like(G2)
		grad_G3 = torch.zeros_like(G3)

		gaterecurrent2dnoind_cuda.backward(items_each_chunk, output, grad_output, 
										 X, B, G1, G2, G3, 
										 grad_X, grad_B, grad_G1, grad_G2, grad_G3)

		return grad_X, grad_B, grad_G1, grad_G2, grad_G3, None

gaterecurrent = GateRecurrent2dnoindFunction.apply

def gaterecurrent2dnoind_pytorch(X, B, G1, G2, G3):
	"""PyTorch implementation of GateRecurrent2dnoind"""
	batch_size, channels, height, width = X.size()
	H = torch.zeros_like(X).clone()
	
	# Forward pass from left to right
	for w in range(width):
		for h in range(height):
			# Get current inputs
			x_t = X[..., h, w]
			b_t = B[..., h, w]
			
			# Calculate gated connections from previous positions
			if w > 0:
				# Top-left connection (h-1, w-1)
				if h > 0:
					h1_prev = H[..., h-1, w-1].clone()
					g1 = G1[..., h, w]  # Gate from current position
					h1_gated = g1 * h1_prev
				else:
					h1_gated = 0
				
				# Left connection (h, w-1)
				h2_prev = H[..., h, w-1].clone()
				g2 = G2[..., h, w]  # Gate from current position
				h2_gated = g2 * h2_prev
				
				# Bottom-left connection (h+1, w-1)
				if h < height-1:
					h3_prev = H[..., h+1, w-1].clone()
					g3 = G3[..., h, w]  # Gate from current position
					h3_gated = g3 * h3_prev
				else:
					h3_gated = 0
				
				# Combine all gated connections
				h_sum = h1_gated + h2_gated + h3_gated
			else:
				h_sum = 0
			
			# Update current hidden state
			H[..., h, w] = b_t * x_t + h_sum
	
	return H


class GateRecurrent2dnoind(nn.Module):
	def __init__(self, items_each_chunk_, backend='cuda'):
		super(GateRecurrent2dnoind, self).__init__()
		self.items_each_chunk = items_each_chunk_
		assert backend in ['cuda', 'pytorch'], f"Backend {backend} not supported"
		self.backend = backend

	def forward(self, X, B, G1, G2, G3):
		if self.backend == 'pytorch':
			return gaterecurrent2dnoind_pytorch(X, B, G1, G2, G3)
		else:  # cuda backend
			return gaterecurrent(X, B, G1, G2, G3, self.items_each_chunk)

	def __repr__(self):
		return f"{self.__class__.__name__}(backend={self.backend})"
