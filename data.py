import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
# from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
from collections import Counter
from vocab import Vocabulary
import sys
import random
import itertools

# Download required NLTK modules.
nltk.download('punkt', quiet=True)


def build_vocabulary(captions, log_path, threshold=4, ignore_tab=False):
    """
    Build a simple vocabulary wrapper.
    """
    print("Building vocabulary")
    counter = Counter()
    for i, caption in enumerate(captions):
        if ignore_tab:
            tokens = nltk.tokenize.word_tokenize(
                caption[0].split("\t")[0].lower().decode('utf-8'))
        else:
            tokens = nltk.tokenize.word_tokenize(
                caption.lower().decode('utf-8'))
        counter.update(tokens)
        if i % 1000 == 0:
            print("[%d/%d] tokenized the captions." % (i, len(captions)))

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
            open(os.path.join(log_path, 'vocab.pkl'), 'w'),
                pickle.HIGHEST_PROTOCOL)
    return vocab


def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f8k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        # if ids provided by get_paths, use split-specific ids
        if ids is None:
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root, caption, img_id, path, image = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def get_raw_item(self, index):
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')

        return root, caption, img_id, path, image

    def __len__(self):
        return len(self.ids)


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, vocab, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'
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

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
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
    Load precomputed captions and image features for the Multi30K data set.

    """

    def __init__(self, data_path, data_split,
                 vocab, lang, undersample=False, log_path=None,
                 half=False, disaligned=False, lang_prefix=False, char_level=False):
        """
        Parameters
        ----------
        data_path : str
            Root of the multi30k data set.
        data_split : str
            'train', 'val' or 'test'           
        vocab : Vocabulary object or None
            When None create a vocabulary, otherwise use given.
        lang : str
            Languages to load from multi30k delimited by '-' e.g.: en-de-fr.
            'en' and 'de' for English and German comparable and 'en1', 'de1', 'fr', 'cs'
            for the translation pairs in English, German, French and Czech.
        undersample : bool
            Use only one caption on the comparable portion.
        log_path : str
            Path where the model is checkpointed.
        half : bool
            Use only half of the images on the comparable English or German.
        disaligned : bool
            When using half and both English and German, choose non-overlapping
            images for the two languages.
        """
        
        self.lang = lang.split("-")
        self.data_split = data_split
        self.vocab = vocab
        self.img_path = data_path + '/data/imgfeats/'
        self.undersample = undersample
        self.log_path = log_path
        self.half = half
        self.disaligned = disaligned
        self.lang_prefix = lang_prefix
        self.char_level = char_level
        #Captions
        self.captions = []
        l_stack = []
        # First let's do langauges in task 2
        if self.data_split == 'test':
            split = 'test_2016_flickr'
        else:
            split = data_split

        self.imgpath = os.path.join(self.img_path, split +'-resnet50-avgpool.npy')
        self.image_vectors = np.load(self.imgpath).astype("float32")
        self.images = []
        # When halving task 2 dataset
        if self.half and self.data_split == 'train':
            ids = np.arange(0, 29000)
            np.random.shuffle(ids)
            # Separating task 2. data to non-overlapping pairs.
            if self.disaligned:
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
                text = '{}.lc.norm.tok.{}.{}'.format(split, i, l)
                path = os.path.join('/data/task2/tok/', text)
                # Add language prefix to each word in all captions like 'en_woman en_sits en_on en_the en_bench.'
                if self.lang_prefix:
                    prefix = lambda x: " ".join(map(lambda y: l+"_"+y, x.split()))
                with open(data_path + path) as f:
                    t = f.read().split('\n')
                    if self.lang_prefix:
                        t = map(prefix, t)
                    caps.append(t[:-1])

            # Pick one caption per image.
            if self.undersample:
                # Pick all captions per image.
                captions_candidate = [random.choice(tup) for tup in zip(*caps)]
            else:
                captions_candidate =  zip(*caps)

            # When running the half-task2 experiments, filter english, german caps and images based on imgids.
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
                # Otherwise take the vectors as they are
                images = image_vectors

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
                        if self.lang_prefix:
                            c = " ".join(map(lambda x: l+"_"+x, c.split()))
                        self.captions.append(c)

        self.images = np.concatenate(self.images, axis=0)
        if vocab is not None:
            self.vocab = vocab
        # Build potentiall multilingual vocab.
        else:
            self.vocab = build_vocabulary(self.captions, self.log_path)
        # Image features
        self.length = len(self.captions)
        print('LANG:', self.lang, 'SPLIT:', self.data_split, 'LENGTH:', self.length)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        # TODO super inefficient For task 2 repeat the images 5 times for each caption
        # Re-write dictionary to character level.
        if self.char_level:
            chars = set(list("".join(self.vocab.idx2word.values())))
            self.vocab = Vocabulary()
            self.vocab.add_word('<pad>')
            self.vocab.add_word('<start>')
            self.vocab.add_word('<end>')
            self.vocab.add_word('<unk>')
            for c in chars:
                self.vocab.add_word(c)

    def __getitem__(self, index):
        #TODO not replicate image vectotrs a billion times
        img_id = index
        image = torch.Tensor(self.images[img_id])
        tokens = self.captions[index]
        vocab = self.vocab
        caption = []
        if self.char_level:
            tokenized = list(tokens.lower())  # split into characters
        else:
            # Convert caption (string) to word ids.
            # We're not using the language ID token
            tokenized = nltk.tokenize.word_tokenize(
                str(tokens).lower().decode('utf-8'))
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokenized])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length


class COCONumpyDataset(data.Dataset):
    """
    Load precomputed captions and image features for the COCO data set.

    """

    def __init__(self, data_path, data_split,
                 vocab, lang, undersample=False, log_path=None,
                 half=False, disaligned=False, lang_prefix=False, char_level=False):
        """
        Parameters
        ----------
        data_path : str
            Root of the multi30k data set.
        data_split : str
            'train', 'val' or 'test'           
        vocab : Vocabulary object or None
            When None create a vocabulary, otherwise use given.
        lang : str
            Languages to load from COCO delimited by '-' e.g.: en-jp.
        undersample : bool
            Use only one caption on the comparable portion.
        log_path : str
            Path where the model is checkpointed.
        half : bool
            Use only half of the images on the comparable English or German.
        disaligned : bool
            When using half and both English and Japanese, choose non-overlapping
            images for the two languages.

        TODO: Reimplement the lang, undersample, half, and disaligned features.
        """
        
        self.lang = lang.split("-")
        self.data_split = data_split
        self.vocab = vocab
        self.img_path = data_path + '/imgfeats/'
        self.undersample = undersample
        self.log_path = log_path
        self.half = half
        self.disaligned = disaligned
        self.lang_prefix = lang_prefix
        self.char_level = char_level
        #Captions
        self.captions = []
        images_map_data = open(data_path + "ids2files.txt").readlines()
        self.images_map = dict()

        for idx, x in enumerate(images_map_data):
            splitx = x.split(":")
            self.images_map[splitx[0]] = idx

        self.imgpath = os.path.join(self.img_path, data_split +'-resnet50-avgpool.npy')
        self.image_vectors = np.load(self.imgpath).astype("float32")
        self.images = []
        caps = []
        text = '{}_captions.txt'.format(data_split)
        path = os.path.join(data_path, text)
        with open(path) as f:
            t = f.read().split('\n')
            caps.append(t[:-1])
            captions_candidate =  zip(*caps)
            self.captions += captions_candidate
            print(len(captions_candidate))

        self.images = self.image_vectors
        if vocab is not None:
            self.vocab = vocab
        # Build potentiall multilingual vocab.
        else:
            self.vocab = build_vocabulary(self.captions, self.log_path, ignore_tab=True)
        # Image features
        self.length = len(self.captions)
        print('LANG:', self.lang, 'SPLIT:', self.data_split, 'LENGTH:', self.length)

    def __getitem__(self, index):
        '''
        This is complex because we need to map the COCO captions to images.
        The COCO images are stored with strange identifiers based on the file
        names, so we keep the caption and the image ID in the text file.
        We split these out here so we don't accidentally encode the image ID.
        '''
        cap = self.captions[index][0]  # WHY does the data loader store tuples instead of strings?
        tokens = cap.split("\t")[0]
        img_id = self.images_map[cap.split("\t")[1]]
        image = torch.Tensor(self.images[img_id])
        vocab = self.vocab
        # Convert caption (string) to word ids.
        caption = []
        if self.char_level:
            tokenized = list(tokens.lower())  # split into characters
        else:
            # Convert caption (string) to word ids.
            # We're not using the language ID token
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
    Generates sentence pairs for the Multi30K dataset.

    Each pair is defined as belonging to the same image, but
    to different languages. Iterates throught all possible 
    such pairs. Doesn't __len__ or __getitem__ 
    its infinite.
    """

    def __init__(self, data_path, data_split, batch_size,
                 vocab, lang, undersample=False, lang_prefix=False, char_level=False):
        """
        Parameters
        ----------
        data_path : str
            Root of the multi30k data set.
        data_split : str
            'train', 'val' or 'test'           
        batch_size : int
            Number of samples per batch.
        vocab : Vocabulary object 
            Vocabulary object to use to map words to integers.
        lang : str
            Languages to load from multi30k delimited by '-' e.g.: en-de-fr.
            'en' and 'de' for English and German comparable and 'en1', 'de1', 'fr', 'cs'
            for the translation pairs in English, German, French and Czech.
        undersample : bool
            Use only one caption on the comparable portion.
        """
        self.lang = lang.split("-")
        self.data_split = data_split
        self.batch_size = batch_size
        self.vocab = vocab
        self.vocab = vocab
        self.undersample = undersample
        self.lang_prefix = lang_prefix
        self.char_level = char_level
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
            if self.lang_prefix:
                prefix = lambda x: " ".join(map(lambda y: l+"_"+y, x.split()))
            for j in range(1, 6):
                f = open(data_path+'/data/task2/tok/'+'{}.lc.norm.tok.{}.{}'.format(self.data_split, j, l), 'rb').read().split("\n")
                if self.lang_prefix:
                    f = map(prefix, f)
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
                        if self.lang_prefix:
                            c = " ".join(map(lambda x: l+"_"+x, c.split()))
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

    def tokenize(self, cap):
        '''Raw string caption to torch tensor of ints.'''
        caption = []
        if self.char_level:
            tokenized = list(cap.lower())  # split into characters
        else:
            # Convert caption (string) to word ids.
            # We're not using the language ID token
            tokenized = nltk.tokenize.word_tokenize(
                str(cap).lower().decode('utf-8'))
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokenized])
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
        # Have to sort for the CUDA padding stuff.
        targetsA = torch.zeros(len(captionsA), max(lengthsA)).long()
        for i, cap in enumerate(captionsA):
            end = lengthsA[i]
            targetsA[i, :end] = cap[:end]
        targetsB = torch.zeros(len(captionsB), max(lengthsB)).long()
        for i, cap in enumerate(captionsB):
            end = lengthsB[i]
            targetsB[i, :end] = cap[:end]
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
                       shuffle=True, num_workers=2, lang=False):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if opt.data_name == "m30k":
        lang = lang if lang else opt.lang
        dset = Multi30KDataset(opt.data_path, data_split, vocab, log_path=opt.logger_name, 
                               lang=lang, undersample=opt.undersample, lang_prefix=opt.lang_prefix,
                               half="half" in opt and opt.half, disaligned="disaligned" in opt and opt.disaligned,
                               char_level=opt.char_level)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)
        return data_loader, dset.vocab
    elif opt.data_name == "coconumpy":
        lang = lang if lang else opt.lang
        dset = COCONumpyDataset(opt.data_path, data_split, vocab, log_path=opt.logger_name, 
                               lang=lang, undersample=opt.undersample, lang_prefix=opt.lang_prefix,
                               half="half" in opt and opt.half, disaligned="disaligned" in opt and opt.disaligned,
                               char_level=opt.char_level)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)
        return data_loader, dset.vocab
    else:
        dset = PrecompDataset(data_path, data_split, vocab)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)
        return data_loader


def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(data_name, vocab, crop_size, batch_size, num_workers, opt):
    '''
    TODO: This function is poorly designed. We need to pass a vocab object to
    get_precomp_loader before we know the vocab? Ugh.
    '''
    dpath = os.path.join(opt.data_path, data_name)

    if opt.data_name.endswith('_precomp'):
        train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                          batch_size, True, num_workers)
        val_loader = get_precomp_loader(dpath, "dev", vocab, opt,
                                        batch_size, False, num_workers)
    elif opt.data_name == "m30k":
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

            # Separate training set for each language.
            train_loader = []
            for l in langs:
                tloader, _ = get_precomp_loader(dpath, 'train',
                                                   t_vocab, opt,
                                                   batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   lang=l)
                train_loader.append(tloader)
                print(l, tloader.__len__())
            if opt.sentencepair:
                sentencepair_loader = M30KSentencePairDataset(opt.data_path, 'train',
                                                              opt.batch_size,
                                                              t_vocab,
                                                              opt.lang,
                                                              undersample=opt.undersample,
                                                              lang_prefix=opt.lang_prefix,
                                                              char_level=opt.char_level)
                sentencepair_loader_val = M30KSentencePairDataset(opt.data_path, 'val',
                                                              opt.batch_size,
                                                              t_vocab,
                                                              opt.lang,
                                                              undersample=opt.undersample,
                                                              lang_prefix=opt.lang_prefix,
                                                              char_level=opt.char_level)
                train_loader.append(sentencepair_loader)
                val_loader.append(sentencepair_loader_val)
    elif opt.data_name == "coconumpy":
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
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, 'train', opt)
        train_loader = get_loader_single(opt.data_name, 'train',
                                         roots['train']['img'],
                                         roots['train']['cap'],
                                         vocab, transform, ids=ids['train'],
                                         batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers,
                                         collate_fn=collate_fn)

        transform = get_transform(data_name, 'val', opt)
        val_loader = get_loader_single(opt.data_name, 'val',
                                       roots['val']['img'],
                                       roots['val']['cap'],
                                       vocab, transform, ids=ids['val'],
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers,
                                       collate_fn=collate_fn)

    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    num_workers, opt):
    dpath = os.path.join(opt.data_path, data_name)

    if opt.data_name.endswith('_precomp'):
        test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                         batch_size, False, workers)
    elif opt.data_name == "m30k":
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

    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, split_name, opt)
        test_loader = get_loader_single(opt.data_name, split_name,
                                        roots[split_name]['img'],
                                        roots[split_name]['cap'],
                                        vocab, transform, ids=ids[split_name],
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers,
                                        collate_fn=collate_fn)

    return test_loader
