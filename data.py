import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
#from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
from collections import Counter
from vocab import Vocabulary
import sys
import random 
import itertools 



class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, vocab, char_level=False):
        self.vocab = vocab
        loc = data_path + '/'
        self.char_level = char_level
        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())
        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000
        # Re-write dictionary to character level.
        if self.char_level:
            chars = set(list("".join(vocab.idx2word.values())))
            self.vocab = Vocabulary()
            self.vocab.add_word('<pad>')
            self.vocab.add_word('<start>')
            self.vocab.add_word('<end>')
            self.vocab.add_word('<unk>')
            for c in chars:
                self.vocab.add_word(c)

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        if self.char_level:
            tokens = list(" ".join(tokens))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length


class Multi30KDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, 
                 vocab, lang, undersample=False, log_path=None,
                 half=False, disalligned=False):
        self.lang = lang.split("-")
        self.data_split = data_split
        self.vocab = vocab
        self.img_path = data_path + '/data/imgfeats/'
        self.undersample = undersample
        self.log_path = log_path
        self.half = half
        self.disalligned = disalligned
        #Captions
        self.captions = []
        l_stack = []
        # First let's do langauges in task 2
        self.imgpath = os.path.join(self.img_path, self.data_split +'-resnet50-avgpool.npy')
        self.image_vectors = np.load(self.imgpath).astype("float32")
        self.images = []
        # When halving task 2 dataset
        if self.half and self.data_split == 'train':
            ids = np.arange(0, 29000)
            np.random.shuffle(ids)
            # Separating task 2. data to non-overlapping pairs. 
            if self.disalligned:
                self.en_ids, self.de_ids = ids[:14500], ids[14500:]
            # Overlapping pairs.
            else:
                self.en_ids, self.de_ids = ids[:14500], ids[:14500]
        for l in self.lang:
            caps = []
            # Czech and French are only in task 1.
            if l in ["cs", "fr"]:
                l_stack.append(l)
                continue
            # en1 and de1 means that they should be chosen from task 1
            elif l.endswith('1'):
                l_stack.append(l[:-1])
                continue
            print("Running {} from task 2".format(l))
            for i in range(1, 6):
                if self.data_split == 'test':
                    split = 'test_2016'
                else:
                    split = data_split
                f = open(data_path+'/data/task2/tok/'+'{}.lc.norm.tok.{}.{}'.format(split, i, l), 'rb').read().split("\n")
                caps.append(f[:-1])
            # Pick one caption per image.
            if self.undersample:
                captions_candidate = [random.choice(tup) for tup in zip(*caps)]
            # Pick all captions per image.
            else:            
                captions_candidate =  zip(*caps)
            # When running the half-task2 experiments, filter enlgish, german caps and images based on imgids.
            if self.half and self.data_split == 'train':
                if l == 'en':
                    captions_candidate = [captions_candidate[x] for x in self.en_ids]
                    image_vectors = self.image_vectors[self.en_ids]
                elif l == 'de':
                    captions_candidate = [captions_candidate[x] for x in self.de_ids]
                    image_vectors = self.image_vectors[self.de_ids]
            else:
                image_vectors = self.image_vectors
            # For task 2 we replicate images 5 times when not undersampling.
            if not self.undersample:
                images = np.repeat(image_vectors, 5, axis=0)
                captions_candidate = [val for tup in captions_candidate for val in tup]
            else:    
                images = image_vectors
            # Otherwise take the vectors as they are
            self.captions += captions_candidate
            self.images.append(images)
            print(l, len(captions_candidate))

        # If there were languages that should be taken from task 1 collect them.
        if len(l_stack) > 0:
            for l in l_stack:
                # Add the image vetors again for each language
                self.images.append(self.image_vectors)
                print("Running {} from task 1".format(l))
                if self.data_split == 'test':
                    split = 'test_2016_flickr'
                else:
                    split = self.data_split
                with open(data_path +'/data/task1/tok/' + '{}.lc.norm.tok.{}'.format(split, l), 'rb') as f:
                    for line in f:
                        c = line.strip()
                        self.captions.append(c)
            
        self.images = np.concatenate(self.images, axis=0)
        if vocab is not None:
            self.vocab = vocab
        # Build potentiall multilingual vocab.
        else:
            self.vocab = self.build_vocabulary()
        # Image features
        self.length = len(self.captions)
        print('LANG:', self.lang, 'SPLIT:', self.data_split, 'LENGTH:', self.length)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        # TODO super inefficient For task 2 repeat the images 5 times for each caption
        # Re-write dictionary to character level.

    def build_vocabulary(self, threshold=4):
        """
        Build a simple vocabulary wrapper.
        """
        print("Building vocabulary")
        counter = Counter()
        for i, caption in enumerate(self.captions):
            tokens = nltk.tokenize.word_tokenize(
                caption.lower().decode('utf-8'))
            counter.update(tokens)
            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(self.captions)))

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
        pickle.dump(vocab, 
                open(os.path.join(self.log_path, 'vocab.pkl'), 'w'), 
                    pickle.HIGHEST_PROTOCOL)
        return vocab


    def __getitem__(self, index):
        #TODO not replicate image vectotrs a billion times
        img_id = index
        image = torch.Tensor(self.images[img_id])
        tokens = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        caption = []
        tokenized = nltk.tokenize.word_tokenize(
            str(tokens).lower().decode('utf-8'))
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokenized])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length


class M30KSentencePairDataset():
    """
    Data iterator to randomly pick batches of sentence pairs 
    from M30K belonging to the same image.
    Doesnt implement __len__ or __getitem__ its just infinite.
    """

    def __init__(self, data_path, data_split, batch_size,
                 vocab, lang, undersample=False,
                 char_level=False, langid=False):
        self.lang = lang.split("-")
        self.langid = langid
        self.data_split = data_split
        self.batch_size = batch_size
        self.vocab = vocab
        self.char_level = char_level
        self.vocab = vocab
        self.undersample = undersample
        # Captions
        self.captions = []
        l_stack = []
        # boundaries between languages in self.captions.
        self.boundaries = []
        # First let's do langauges in task 2
        for i, l in enumerate(self.lang):
            # Czech and French are only in task 1.
            caps = []
            if l in ["cs", "fr"]:
                l_stack.append(l)
                continue
            # en1 and de1 means that they should be chosen from task 1
            elif l.endswith('1'):
                l_stack.append(l[:-1])
                continue
            print("Running {} from task 2".format(l))
            for j in range(1, 6):
                f = open(data_path+'/data/task2/tok/'+'{}.lc.norm.tok.{}.{}'.format(self.data_split, j, l), 'rb').read().split("\n")
                # Prepend Language id (EN, FR, DE, CS) to the captions.
                if self.langid:
                    f = map(lambda x: l.upper() + " " + x, f)
                caps.append(f[:-1])
            # Store each languages captions in a separate list.
            # When undersample, pick one out of 5 possible
            if self.undersample:
                c = [random.choice(tup) for tup in zip(*caps)]
            # Otherwise take all
            else:
                c = zip(*caps)
            self.captions.append(c)
        # If there were languages that should be taken from task 1 collect them.
        if len(l_stack) > 0:
            for l in l_stack:
                # Add the image vetors again for each language
                print("Running {} from task 1".format(l))
                caps = []
                with open(data_path +'/data/task1/tok/' + '{}.lc.norm.tok.{}'.format(self.data_split, l), 'rb') as f:
                    for line in f:
                        c = line.strip()
                        if self.langid:
                            c = l.upper() + " " + c
                        caps.append(c)
                self.captions.append(caps)
        # Create a bigass list of all possible pairs
        self.allpairs = []
        n_caption = len(self.captions[0])
        for i in range(n_caption):
            # captions given an image
            l =[list(x[i]) if isinstance(x[i], tuple) else [x[i]] for x in self.captions]
            # All language-pair combinations
            for c in itertools.combinations(l, 2):
                # All possible pairs in language pair
                for p in itertools.product(*c):
                    self.allpairs.append(p)
        self.allpairs = np.array(self.allpairs)
        self.length = len(self.allpairs)
        self.reset()
        print('Number of sentencepairs:', self.length)
        # Image features
        self.datasets = len(self.captions)
        if self.char_level:
            chars = set(list("".join(self.vocab.idx2word.values())))
            self.vocab = Vocabulary()
            self.vocab.add_word('<pad>')
            self.vocab.add_word('<start>')
            self.vocab.add_word('<end>')
            self.vocab.add_word('<unk>')
            for c in chars:
                self.vocab.add_word(c)
    
    def tokenize(self, cap):
        '''Raw string caption to torch tensor of ints.'''
        tokens = nltk.tokenize.word_tokenize(
            str(cap).lower().decode('utf-8'))
        if self.char_level:
            tokens = list(" ".join(tokens))
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        caption = torch.Tensor(caption)
        return caption
 
    def shuffle_data(self):
        """
        Shuffle the full dataset.
        """
        inds = np.random.permutation(self.length)
        self.allpairs = self.allpairs[inds]

    def reset(self):
        """
        Reset batching, shuffle the dataset
        """
        self.bottom = 0
        self.top = self.batch_size
        self.shuffle_data()
    
    def next(self):
        """Use sampling with replacement, infinite iterator."""
        #TODO not replicate image vectotrs a billion times
        # Take 2 languages
        capsA, capsB = zip(*self.allpairs[self.bottom:self.top])
        if self.top == self.length:
            self.reset()
        else:
            self.bottom += self.batch_size
            self.top = min(self.bottom + self.batch_size, self.length)
        # If captions are from Task 2. randomly sample one out of 5.
        if isinstance(capsA[0], tuple):
            capsA = map(random.choice, capsA)
        if isinstance(capsB[0], tuple):
            capsB = map(random.choice, capsB)
        captionsA, captionsB = [], []
        lengthsA, lengthsB = [], []
        for ca, cb in zip(capsA, capsB):
            capA, capB = self.tokenize(ca), self.tokenize(cb)
            captionsA.append(capA)
            captionsB.append(capB)
            lengthsA.append(len(capA))
            lengthsB.append(len(capB))
            # lengthsA.append(len(capA))
            # lengthsB.append(len(capB))
        # Have to sort for the CUDA padding stuff.
        targetsA = torch.zeros(len(captionsA), max(lengthsA)).long()
        for i, cap in enumerate(captionsA):
            end = lengthsA[i]
            targetsA[i, :end] = cap[:end]
        targetsB = torch.zeros(len(captionsB), max(lengthsB)).long()
        for i, cap in enumerate(captionsB):
            end = lengthsB[i]
            targetsB[i, :end] = cap[:end]
        # Convert caption (string) to word ids.
        return targetsA, targetsB, lengthsA, lengthsB

    def __iter__(self):
        return self

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



def get_loader_single(data_name, split, root, json, vocab, transform,
                      batch_size=100, shuffle=True,
                      num_workers=2, ids=None, collate_fn=collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if 'coco' in data_name:
        # COCO custom dataset
        dataset = CocoDataset(root=root,
                              json=json,
                              vocab=vocab,
                              transform=transform, ids=ids)
    elif 'f8k' in data_name or 'f30k' in data_name:
        dataset = FlickrDataset(root=root,
                                split=split,
                                json=json,
                                vocab=vocab,
                                transform=transform)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2, lang=False, langid=False,
                       lang_vocab=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    lang = lang if lang else opt.lang
    dset = Multi30KDataset(opt.data_path, data_split, vocab, log_path=opt.logger_name,
            lang=lang, undersample=opt.undersample, half=opt.half, 
            disalligned=opt.disalligned)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader, dset.vocab




def get_loaders(data_name, vocab, crop_size, batch_size, num_workers, opt):
    '''
    TODO: This function is poorly designed. We need to pass a vocab object to
    get_precomp_loader before we know the vocab? Ugh.
    '''
    dpath = os.path.join(opt.data_path, data_name)

    langs = opt.lang.split("-")
    # Run this to get vocabulary for all languages
    # lang_vocab is the separate language ID vocabulary
    # the None means we pass is no existing vocab object.
    t_loader, t_vocab = get_precomp_loader(dpath, 'train',
                                                    None, opt,
                                                    batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)

    # Set the vocabulary size to be used by the model
    opt.vocab_size = t_vocab.idx
    
    print("Validation sets")
    if len(langs) == 1:
        # If only one language just use train vocab for the val-set and we are done.
        val_loader, _ = get_precomp_loader(dpath, "val", t_vocab,
                                              opt, batch_size,
                                              shuffle=False, num_workers=num_workers)
        train_loader = t_loader
    else:
        # If there are multiple languages create mutliple val sets using the joint vocab.
        val_loader = []
        for l in langs:
            vloader, _ = get_precomp_loader(dpath, 'val', t_vocab,
                                               opt, batch_size, 
                                               shuffle=False,
                                               num_workers=num_workers,
                                               lang=l)
            val_loader.append(vloader)
            #print(l, vloader.__len__())

        # Separate training sets for each language.
        train_loader = []
        for l in langs:
            tloader, _ = get_precomp_loader(dpath, 'train',
                                               t_vocab, opt,
                                               batch_size, 
                                               shuffle=True, 
                                               num_workers=num_workers,
                                               lang=l,
                                               lang_vocab=l_vocab)
            train_loader.append(tloader)
            print(l, tloader.__len__())
        
        if opt.sentencepair:
            sentencepair_loader = M30KSentencePairDataset(opt.data_path, 'train',
                                                          opt.batch_size, 
                                                          t_vocab,
                                                          opt.lang,
                                                          langid=opt.langid,
                                                          undersample=opt.undersample)
            sentencepair_loader_val = M30KSentencePairDataset(opt.data_path, 'val',
                                                          opt.batch_size, 
                                                          t_vocab,
                                                          opt.lang,
                                                          langid=opt.langid,
                                                          undersample=opt.undersample)
            train_loader.append(sentencepair_loader)
            val_loader.append(sentencepair_loader_val)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    num_workers, opt, lang_vocab=None):
    dpath = os.path.join(opt.data_path, data_name)

    # the None means we pass is no existing vocab object.
    langs = opt.lang.split("-")
    if len(langs) == 1:
        test_loader, _ = get_precomp_loader(dpath, split_name, vocab, opt,
                                           batch_size, False, num_workers)
    else:
        # If there are multiple languages create mutliple test sets
        # using the joint vocab.
        test_loader = []
        for l in langs:
            vloader, v_vocab = get_precomp_loader(dpath,
                                                  split_name,
                                                  vocab, opt,
                                                  batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers, 
                                                  lang=l)
            test_loader.append(vloader)  

    return test_loader
