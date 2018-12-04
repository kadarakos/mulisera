import numpy as np
from data2 import get_loaders
from model import VSE
from torch.autograd import Variable
import torch
import pickle
import os
from evaluation import encode_data
model_path ='/roaming/u1257964/vsepp_distant/deenm30k+coco_409/409/model_best.pth.tar'
print("Loading model")
checkpoint = torch.load(model_path)

opt = checkpoint['opt']
vocab = pickle.load(open(os.path.join(opt.logger_name, 'vocab.pkl'), 'rb'))
model = VSE(opt)
model.load_state_dict(checkpoint['model'])
datasets = opt.data_set.split('-')
loaders = get_loaders(datasets, val_sets=['m30ken'], lang_prefix=False, 
                      downsample=False, batch_size=128, path='.', 
                      shuffle_train=True)

for dataset in datasets:
    print("Encoding {}".format(dataset))
    loader = loaders.get_trainloader(dataset)
    loader.dataset.vocab = vocab
    img_emb, cap_emb, _ = encode_data(model, loader)
    np.save(os.path.join(opt.logger_name, '{}_img_emb'.format(dataset)), img_emb)
    np.save(os.path.join(opt.logger_name, '{}_cap_emb'.format(dataset)), cap_emb)

