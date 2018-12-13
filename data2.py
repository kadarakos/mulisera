import os 
import re
import pickle
import random
from collections import Counter
import numpy as np  
import nltk
from torch.utils.data import DataLoader, Dataset
import torch
from vocab import Vocabulary

COCO_PATH = '/roaming/u1257964/coco_mulisera_2'
M30K_PATH = '/roaming/u1257964/multi30k-dataset/'

def build_vocabulary(captions, path='.', threshold=4):
    """
    Build a simple vocabulary wrapper.
    """
    print("Building vocabulary")
    counter = Counter()
    for i, caption in enumerate(captions):
        counter.update(caption)

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    print('Num words:', vocab.idx)
    path = os.path.join(path, 'vocab.pkl')
    with open(path, 'w') as f:
        pickle.dump(vocab, f,
                    pickle.HIGHEST_PROTOCOL)
    return vocab


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


# prefix = lambda x: " ".join(map(lambda y: l+"_"+y, x.split()))
def tokenize(s):
    """
    Remove non-alphanumeric characters, then tokenize.
    
    s : str
        String to tokenize.
    """
    s = re.sub(r'([^\s\w]|_)+', '', s)
    tokens = nltk.tokenize.word_tokenize(s.lower().decode('utf-8'))
    return tokens


def read_m30K(data_path, lang, split, lang_prefix=False):
    """
    Reads data from Multi30K task 2 comparable.
    
    data_path : str
        Root of the Multi30K data folder.
    split : str
        train, val or test.
    lang_prefix : bool
        Place en_ or de_ prefix infront of each token.
    """
    if split == 'test':
        split = 'test_2016'
    img_path = data_path + '/data/imgfeats/'
    imgpath = os.path.join(img_path, split +'-resnet50-avgpool.npy')
    image_vectors = np.load(imgpath).astype("float32")
    images = []
    caps = []
    img_ids = []
    for i in range(1, 6):
        text = '{}.lc.norm.tok.{}.{}'.format(split, i, lang)
        path = os.path.join('/data/task2/tok/', text)
        # Add language prefix to each word in all captions like 'en_woman en_sits en_on en_the en_bench.'
        with open(data_path + path) as f:
            t = f.read().split('\n')
            #TODO implement lang prefix
            if lang_prefix:
                pass
            caps.append(t[:-1])
    caps = zip(*caps)
    captions = [val for tup in caps for val in tup]
    images = np.repeat(image_vectors, 5, axis=0)
    return images, captions   


def read_coco(data_path, split, lang_prefix=False, downsample=False):
    """
    Reads data from Multi30K task 2 comparable.
    
    data_path : str
        Root of the coco_mulisra directory created with coco_process.py
    split : str
        train, val or test.
    lang_prefix : bool
        Place en_ prefix infront of each token.
    downsample : int
        Number of images to keep.
    """
    img_path = os.path.join(data_path, 'imgfeats')
    img_path = os.path.join(img_path, split +'-resnet50-avgpool.npy')
    image_vectors = np.load(img_path).astype("float32")
    caps = []
    text = '{}_captions.txt'.format(split)
    path = os.path.join(data_path, text)
    with open(path) as f:
        t = f.read().split('\n')
        #TODO implement lang_prefix
        if lang_prefix:
            pass
        caps.append(t[:-1])
    captions = np.array([y.split('\t')[0] for x in caps for y in x])
    if downsample:
        #Get indices for a random subsample for the image vectors
        a = np.arange(image_vectors.shape[0] - 1)
        np.random.shuffle(a)
        img_inds = a[:downsample]
        #Generate indices for the corresponding captions
        cap_inds = [np.arange(x*5, (x*5)+5) for x in img_inds]
        cap_inds = [y for x in cap_inds for y in x]
        #Pick the samples
        image_vectors = image_vectors[img_inds]
        captions = captions[cap_inds]
        print(image_vectors.shape)
        print(len(captions))
    #Repeast each image 5 times
    images = np.repeat(image_vectors, 5, axis=0)
    return images, captions


def read_synthetic(dataname, model_path, lang_prefix=False):
    """
    Reads synthetic captions from model_path.
    
    data_path : str
        Root of the coco_mulisra directory created with coco_process.py
    split : str
        train, val or test.
    lang_prefix : bool
        Place en_ prefix infront of each token.
    downsample : int
        Number of images to keep.
    """
    caps = []
    model_parent = model_path
    for f in os.listdir(model_parent):
        if f == "{}_synthetic_cap.txt".format(dataname):
            cap_file = f
    path = os.path.join(model_parent, cap_file)
    with open(path) as f:
        t = f.read().split('\n')
        #TODO implement lang_prefix
        if lang_prefix:
            pass
        caps = t[:-1]
    return caps


def load_data(name, split, lang_prefix, downsample=False):
    print("Loading {}, split {}".format(name, split))
    if name == 'coco':
        # Downsample coco valset, because of no reason
        path = COCO_PATH
        img, cap = read_coco(path, split, lang_prefix, downsample)
        # Add restval to train for now
        #TODO make restval thing optional
        if split == 'train':
            img2, cap2 = read_coco(path, 'restval', lang_prefix, downsample)
            img = np.vstack([img, img2])
            cap = np.concatenate([cap, cap2], axis=0)
    elif name == 'm30ken':
        path = M30K_PATH
        img, cap = read_m30K(path, 'en', split, lang_prefix)
    elif name == 'm30kde':
        path = M30K_PATH
        img, cap = read_m30K(path, 'de', split, lang_prefix)
    else:
        raise NotImplementedError
    print("N images {}, N captions {}".format(len(img), len(cap)))
    return img, cap
        
        

class ImageCaptionDataset(Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, captions, images, vocab=None):
        # Captions
        self.captions = captions
        self.images = images
        self.length = len(self.captions)
        self.vocab = vocab
        print("Tokenizing")
        self.tokenized_captions = [tokenize(x) for x in captions]
        print(vocab)

    def __getitem__(self, index):
        image = torch.Tensor(self.images[index])
        tokens = self.tokenized_captions[index]
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, index

    def __len__(self):
        return self.length

class DatasetCollection():
    
    def __init__(self):
        self.data_loaders = {}
        self.data_iterators = {}
        self.data_sets = {}
        self.val_loaders = {}
        self.val_sets = {}
        self.vocab = None

    def add_trainset(self, name, dset, batch_size, shuffle=True):
        data_loader = DataLoader(dataset=dset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 pin_memory=True,
                                 collate_fn=collate_fn)
        self.data_sets[name] = dset
        self.data_loaders[name] = data_loader
        self.data_iterators[name] = iter(data_loader)

    def add_valset(self, name, dset, batch_size, shuffle=False):
        data_loader = DataLoader(dataset=dset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=collate_fn)
        self.val_sets[name] = dset
        self.val_loaders[name] = data_loader
 
    def get_valloader(self, name):
        return self.val_loaders[name]
    
    def get_trainloader(self, name):
        return self.data_loaders[name]

    def compute_joint_vocab(self, path):
        """Join the captions of all data sets and recompute the vocabulary."""
        caps = [v.tokenized_captions for k, v in self.data_sets.items()]
        caps = [y for x in caps for y in x]
        vocab = build_vocabulary(caps, path)
        self.vocab = vocab
        for i in self.data_sets:
            self.data_sets[i].vocab = self.vocab
            self.data_loaders[i].dataset.vocab = self.vocab
        for j in self.val_sets:
            self.val_sets[j].vocab = self.vocab
            self.val_loaders[j].dataset.vocab = self.vocab

    def __iter__(self):
        return self

    def next(self):
        """Pick a data loader, either yield next batch or if ran out re-init and yield."""
        k = random.choice(self.data_loaders.keys())
        loader = self.data_iterators[k]
        try:
            images, targets, lengths, ids = next(loader)
        except StopIteration:
            self.data_iterators[k] = iter(self.data_loaders[k])
            loader = self.data_iterators[k]
            images, targets, lengths, ids = next(loader)
        return images, targets, lengths, ids


def get_loaders(data_sets, val_sets, lang_prefix, downsample, 
                path, batch_size, synth_path=None, shuffle_train=True):
    data_loaders = DatasetCollection()
    synthcaps = []
    synthnames = []
    print("Loading training sets.")
    for name in data_sets:
        if "synthetic" in name:
            synthset = name.split("_")[1]
            synthnames.append(synthset)
            #HACK new logger name eds with _2 and original is upt to that 
            synthcap = read_synthetic(synthset, synth_path, lang_prefix)
            synthcaps.append(synthcap)
        else:
            train_img, train_cap = load_data(name, 'train', lang_prefix, downsample)
            trainset = ImageCaptionDataset(train_cap, train_img)
            data_loaders.add_trainset(name, trainset, batch_size, shuffle_train)
    print("Adding synthetic data sets")
    for name, cap in zip(synthnames, synthcaps):
        img = data_loaders.data_loaders[name].dataset.images
        n = "synth"+name
        s = ImageCaptionDataset(cap, img)
        data_loaders.add_trainset(n, s, batch_size, shuffle_train)
    print("Loading validation sets.")
    for name in val_sets:
        val_img, val_cap = load_data(name, 'val', lang_prefix, downsample)
        valset = ImageCaptionDataset(val_cap, val_img) 
        data_loaders.add_valset(name, valset, batch_size)
    data_loaders.compute_joint_vocab(path)
    return data_loaders 

def get_test_loader(name, split, batch_size, lang_prefix, downsample=False):
    img, cap = load_data(name, split, lang_prefix, downsample)
    valset = ImageCaptionDataset(cap, img)
    loader = DataLoader(dataset=valset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=True,
                        collate_fn=collate_fn)
    return loader
