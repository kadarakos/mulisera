import os
import pickle
import argparse
import numpy as np
from data2 import get_loaders, load_data
from model import VSE
import torch
from evaluation import encode_data
device = torch.device("cuda") 


def compute_embeddings(model_path):
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    vocab = pickle.load(open(os.path.join(opt.logger_name, 'vocab.pkl'), 'rb'))
    model = VSE(opt)
    model.load_state_dict(checkpoint['model'])
    model_datasets = opt.data_set.split('-')
    loaders = get_loaders(model_datasets, val_sets=['m30ken'], lang_prefix=False, 
                          downsample=False, batch_size=128, path='.', 
                          shuffle_train=True)

    for dataset in model_datasets:
        if dataset not in model_datasets:
            print("Model wasn't trained on {}".format(dataset))
            raise NotImplementedError
        print("Encoding {}".format(dataset))
        loader = loaders.get_trainloader(dataset)
        loader.dataset.vocab = vocab
        img_emb, cap_emb, _ = encode_data(model, loader)
        np.save(os.path.join(opt.logger_name, '{}_img_emb'.format(dataset)), img_emb)
        np.save(os.path.join(opt.logger_name, '{}_cap_emb'.format(dataset)), cap_emb)
     

def synthetic_captions(emb1, emb2, caps, chunk_size=1000):
    """
    emb1 : embedded captions from data set 1.
    emb2 : embedded captions from data set 2.
    caps : captions from data set 2.
    
    Returns for each caption in emb1 the most similar caption from cap.
    """
    sim_caps = []
    emb2 = torch.from_numpy(emb2.T).to(device)
    print("Finding most similar captions")
    for i in range(0, len(emb1), chunk_size):
        top = i + chunk_size
        print("{}/{}".format(i, len(emb1)))
        A = torch.from_numpy(emb1[i:top]).to(device)
        sims = torch.mm(A, emb2)
        ranks = torch.argmax(sims, dim=1)
        print("adding to list")
        for j in ranks:
            sim_caps.append(caps[j])
    return sim_caps


def create_synthetic_dataset(model_path, dataset1, dataset2):
    """
    Takes captions from dataset1 and pairs them with images from dataset2.
    
    dataset1: name of the first data set
    dataset2: name of the second data set
    """
    img, caps = load_data(dataset2, 'train', False)
    del img
    #compute_embeddings(model_path)
    emb1_path = os.path.join(os.path.dirname(model_path), '{}_cap_emb.npy'.format(dataset1))
    emb2_path = os.path.join(os.path.dirname(model_path), '{}_cap_emb.npy'.format(dataset2))
    print("Loading {}".format(emb1_path))
    emb1 = np.load(emb1_path)
    print("Loading {}".format(emb2_path))
    emb2 = np.load(emb2_path)
    new_caps = synthetic_captions(emb1, emb2, caps)
    print(new_caps)
    print(len(new_caps))
    return new_caps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", type=str,
                        help='Data sets to merge. Annotate the images of the 2nd dataset with the cations of the 1st.')
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint.")
    args = parser.parse_args()
    datasets = args.data_set.split("-")
    if len(datasets) != 2:
        print("Need to specifiy 2 datasets")
        raise NotImplementedError
    else:
        d1, d2 = datasets
        caps = create_synthetic_dataset(args.model_path, d1, d2)
        out_path = os.path.join(os.path.dirname(args.model_path), "{}_synthetic_cap.txt".format(datasets[0]))
        print("Writing caps in {}".format(out_path))
        with open(out_path, 'w') as f:
            for line in caps:
                f.write("%s\n" % line)



