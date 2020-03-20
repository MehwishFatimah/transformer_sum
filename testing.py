#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:50:39 2019

@author: fatimamh
"""
import argparse
import sys
import os
from os import path
import time
import numpy as np
import pandas as pd
from csv import writer
import torch
from torch.autograd import Variable

from utils.loader_utils import get_data
from model.model_helper import *
from model.model import *
from model.mask_utils import * 

from utils.lang_utils import load_vocab
from utils.lang_utils import tensor_to_text

import model_config as config

parser = argparse.ArgumentParser(description = 'Help for the main module')
parser.add_argument('--f', type = str,   default = None,   help = 'To resume training, provide model')

'''==============================================================================
'''
class Test(object):
	def __init__(self, device):
		self.device = device
		self.model = make_model(device)
		self.vocab = load_vocab(config.sum_vocab_c)
		self.file  = config.s_summaries
	
	'''==============================================================================
	'''
	def load_model(self, file=None):
		print('-------------Start: load_model-------------')
		if file is not None and path.exists(file):
			state = torch.load(file, map_location= lambda storage, location: storage)
			self.model.load_state_dict(state["model_state"])
			print('-------------End: load_model-------------')
	
	'''==============================================================================
	'''
	def build_row(self, *args):
		row =[]
		for i in range(len(args)):
			row.append(args[i])
		return row
	'''==============================================================================
	'''
	def write_csv(self, file, content):
		with open(file, 'a+', newline='') as obj:
			headers = ['reference', 'system']
			csv_writer = writer(obj)
			is_empty = os.stat(file).st_size == 0
			if is_empty:
				csv_writer.writerow(headers)
			csv_writer.writerow(content)

	'''==============================================================================
	'''
	def get_text(self, reference, system):
		
		reference = reference.tolist()
		reference = [int(i) for i in reference]
		reference = tensor_to_text(self.vocab, reference)
		print('reference: {}'.format(reference))
		print()
		#system = system.tolist()
		system  = [int(i) for i in system]
		system = tensor_to_text(self.vocab, system)	
		print('system: {}'.format(system))
		print()
		
		row = self.build_row(reference, system)
		print('row: {}\n'.format(row))
		self.write_csv(self.file, row)

	'''==============================================================================
	'''
	def greedy_decode(self, src, src_mask, max_len= config.max_sum, start_symbol= config.SP_index):
		
		memory = self.model.encode(src, src_mask)
		ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
		for i in range(max_len-1):
			out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
			
			prob = model.generator(out[:, -1])
			_, next_word = torch.max(prob, dim = 1)
			next_word = next_word.data[0]
			ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
		return ys

	'''==============================================================================
	'''
	def test_a_batch(self, input_tensor, target_tensor, input_lens, output_lens):

		'''------------------------------------------------------------
		1: Setup tensors
		------------------------------------------------------------'''
		input_tensor = input_tensor.squeeze()#.t()
		target_tensor = target_tensor.squeeze()#.t()
		#print('input_tensor: {}'.format(input_tensor.shape))
		#print('target_tensor: {}'.format(target_tensor.shape))

		input_tensor, target_tensor = input_tensor.to(self.device), target_tensor.to(self.device)
		batch_size = input_tensor.size()[0]
		for idx in range(batch_size): # take 1 example from batch
			
			input_mask = (input_tensor[idx] != config.PAD_index).unsqueeze(-2)
			input_mask = input_mask.to(self.device)
                        
			out = beam_search(input_tensor[idx], input_mask)			
			print('out: {}'.format(out.shape))
			self.get_text(target_tensor[idx].squeeze(), out.squeeze())
	
	'''==============================================================================
	'''
	def test_model(self, file=None):
		'''-----------------------------------------------
		Step 1: Get model from file
		-----------------------------------------------'''
		# todo: best model
		if file is None:
			file = "train" + "_e_{}_".format(config.epochs) + "all" + ".pth.tar"
			folder = os.path.join(config.log_folder, 'train')
			if os.path.exists(folder):
				file = os.path.join(folder, file)
			else:
				print('Train dir or file {} doesn\'t exist'.format(folder))
				sys.exit()	
		else:
			file = file
		
		if os.path.exists(file):
			self.load_model(file)
		else:
			print('Train file doesn\'t exist... {}'.format(file))
			sys.exit()
		
		'''-----------------------------------------------
		Step 2: Get data_loaders
		-----------------------------------------------'''
		start = time.time()
		test_loader = get_data("test")
		self.model.eval()
		batch_idx = 1
		total_batch = len(test_loader)
		'''------------------------------------------------------------
		3: Get batches from loader and pass to evaluate             
		------------------------------------------------------------'''
		for input_tensor, target_tensor, input_lens, output_lens in test_loader:
			print('\t---Test Batch: {}/{}---'.format(batch_idx, total_batch))
			self.test_a_batch(input_tensor, target_tensor, input_lens, output_lens)
			batch_idx +=1
		
		return True
'''	
if __name__ == '__main__':
	args = parser.parse_args()
	file = args.f 
	#print (os.getcwd())
	torch.cuda.empty_cache()
	device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device: {}'.format(device))
	test = Test(device)
	final = test.test_model(file)
	print(final)
'''
