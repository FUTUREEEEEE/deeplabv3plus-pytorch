# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
from tensorboardX import SummaryWriter
from config import cfg
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader

tblogger = SummaryWriter(cfg.LOG_DIR+'/output/')
def test_net():
	flag=0
	dataset = generate_dataset(cfg.DATA_NAME, cfg, 'val')
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TEST_BATCHES, 
				shuffle=False, 
				num_workers=cfg.DATA_WORKERS)
	
	net = generate_net(cfg)
	print('net initialize')
	if cfg.TEST_CKPT is None:
		raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
	

	print('Use %d GPU'%cfg.TEST_GPUS)
	device = torch.device('cuda')
	if cfg.TEST_GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
	net.to(device)

	print('start loading model %s'%cfg.TEST_CKPT)
	model_dict = torch.load(cfg.TEST_CKPT,map_location=device)
	net.load_state_dict(model_dict)
	
	net.eval()	
	result_list = []
	with torch.no_grad():
		for i_batch, sample_batched in enumerate(dataloader):
			name_batched = sample_batched['name']
			row_batched = sample_batched['row']
			col_batched = sample_batched['col']

			[batch, channel, height, width] = sample_batched['image'].size()
			multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).to(0)
			for rate in cfg.TEST_MULTISCALE:
				inputs_batched = sample_batched['image_%f'%rate]
				inputs_batched=inputs_batched.float().to(0)
				predicts = net(inputs_batched).to(0)
				predicts_batched = predicts.clone()
				del predicts
				if cfg.TEST_FLIP:
					inputs_batched_flip = torch.flip(inputs_batched,[3]) 
					predicts_flip = torch.flip(net(inputs_batched_flip),[3]).to(0)
					predicts_batched_flip = predicts_flip.clone()
					del predicts_flip
					predicts_batched = (predicts_batched + predicts_batched_flip) / 2.0
			
				predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1/rate, mode='bilinear', align_corners=True)
				multi_avg = multi_avg + predicts_batched
				del predicts_batched
			
			multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
			#print(multi_avg.size())
			result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)
			result_color=np.zeros(shape=(batch,3,result.shape[-1],result.shape[-1]))
			#print(result[0].shape)
			if cfg.IF_SAVE_TEST_OUT:
				for i in range(batch):
					result_color[i]=dataset.label2colormap(result[i]).transpose((2,0,1))
					tblogger.add_image('Output', result_color[i], flag)
					tblogger.add_image('Input', inputs_batched[i], flag)	
					flag=flag+1
			
			for i in range(batch):
				row = row_batched[i]
				col = col_batched[i]
			#	max_edge = max(row,col)
			#	rate = cfg.DATA_RESCALE / max_edge
			#	new_row = row*rate
			#	new_col = col*rate
			#	s_row = (cfg.DATA_RESCALE-new_row)//2
			
	 
			#	p = predicts_batched[i, s_row:s_row+new_row, s_col:s_col+new_col]
				p = result[i,:,:]
				p = cv2.resize(p, dsize=(col,row), interpolation=cv2.INTER_NEAREST)
				result_list.append({'predict':p, 'name':name_batched[i]})

			print('%d/%d'%(i_batch,len(dataloader)))
	dataset.save_result(result_list, cfg.MODEL_NAME)
	dataset.do_python_eval(cfg.MODEL_NAME)
	print('Test finished')

if __name__ == '__main__':
	test_net()


