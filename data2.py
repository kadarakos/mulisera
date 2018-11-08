import os 
import numpy as np  

# prefix = lambda x: " ".join(map(lambda y: l+"_"+y, x.split()))


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
    print("Running {} from task 2".format(lang))
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
    captions = [y for x in caps for y in x]
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
        print(img_inds)
        #Generate indices for the corresponding captions
        cap_inds = [np.arange(x*5, (x*5)+5) for x in img_inds]
        cap_inds = [y for x in cap_inds for y in x]
        print(cap_inds)
        #Pick the samples
        image_vectors = image_vectors[img_inds]
        captions = captions[cap_inds]
    #Repeast each image 5 times
    images = np.repeat(image_vectors, 5, axis=0)
    return images, captions
