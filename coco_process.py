import json
import sys
import os
from collections import defaultdict
import argparse
from img_feat_extract import extract_img_feats


def extract_karpathy_valsets(root, output_path):
    '''
    Extract the raw captions from Karpathy's JSON file.

    WARNING: throws away the captions when there are more than 5 per image.
             We need five captions per image for the evaluation.py scripts
             to correctly work in the i2t and t2i functions.

    Creates three output files:
      filenames.txt: the names of the .JPG image files
      ids2files.txt: maps the JSON image_id to the filename
      captions.txt: the raw captions, stored in ascending image_id order

    Assumes you are working with this JSON file:
    https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
    '''
    path = os.path.join(root, 'annotations/dataset_coco.json')
    print("Loading {}".format(path))
    d = json.load(open(path))
    
    for split in ['val', 'restval', 'test', 'train']:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        imgs_path = os.path.join(coco_root, '{}{}'.format('val', 2014))
        if split == 'train':
            imgs_path = os.path.join(coco_root, '{}{}'.format('train', 2014))
            
        ids2files = dict()
        
        ''' 
        Collect the list of image filenames and their IDs.
        Then sort the image IDs in ascending order.
        '''
        for i in d['images']:
            if i['split'] == split:
                ids2files[i['cocoid']] = i['filename']
        
        sortedids = sorted(ids2files.keys())

        '''
        Write the names of the files onto disk (filenames.txt),
        write the mapping of imageID:filename to disk (ids2file.txt)
        '''
        handle0 = open(os.path.join(output_path, '{}_filenames.txt'.format(split)), 'w')
        handle2 = open(os.path.join(output_path, "{}_ids2files.txt".format(split)), "w")
        index =   open(os.path.join(imgs_path, '{}-index.txt'.format(split)), 'w')
 
        for i in sortedids:
            handle0.write("{}\n".format(ids2files[i]))
            index.write("{}\n".format(ids2files[i]))
            handle2.write('{}:{}\n'.format(i, ids2files[i]))
        handle0.close()
        handle2.close()
        index.close()
        
        '''
        Collect the captions for each image. These are stored in a 
        dictionary. We then sort the captions by the image_id keys. This means
        they can be written to disk in the same order as the names of the images
        (filenames.txt).
        '''
        handle1 = open(os.path.join(output_path, "{}_captions.txt".format(split)), "w")
        captions = defaultdict(list)
        for i in d['images']:
            if i['split'] == split:
                for s in i['sentences']:
                    captions[i['cocoid']].append(s['raw'])
        
        sorted_captions = sorted(captions.keys())
        
        '''
        Keep a counter of how many times an image contains too many captions.
        '''
        ignored = 0
        for i in sorted_captions:
            for idx, c in enumerate(captions[i]):
                if idx < 5:
                    handle1.write('{}\t{}\n'.format(c.replace("\n",""), str(i)))
                else:
                    ignored += 1
        
        handle1.close()
        
        print("Ignored {} captions because there were more than 5 captions for a given image".format(ignored))


def extract(root, output_path, split="val"):
    '''
    Extract the raw captions from the COCOdataset.org JSON file.

    WARNING: throws away the captions when there are more than 5 per image.
             We need five captions per image for the evaluation.py scripts
             to correctly work in the i2t and t2i functions.

    Creates three output files:
      filenames.txt: the names of the .JPG image files
      ids2files.txt: maps the JSON image_id to the filename
      captions.txt: the raw captions, stored in ascending image_id order

    Assumes you are working with these JSON files:
    http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    '''
    path = os.path.join(root, 'annotations/captions_{}2014.json')
    print("Loading {}".format(path.format(split)))
    d = json.load(open(path.format(split)))
    
    ids2files = dict()
    
    ''' 
    Collect the list of image filenames and their IDs.
    Then sort the image IDs in ascending order.
    '''
    for i in d['images']:
        ids2files[i['id']] = i['file_name']
    
    sortedids = sorted(ids2files.keys())
    imgs_path = os.path.join(coco_root, '{}{}'.format(split, 2014))

    '''
    Write the names of the files onto disk (filenames.txt),
    write the mapping of imageID:filename to disk (ids2file.txt)
    '''
    handle0 = open(os.path.join(output_path, '{}_filenames.txt'.format(split)), 'w')
    handle2 = open(os.path.join(output_path, "{}_ids2files.txt".format(split)), "w")
    index =   open(os.path.join(imgs_path, 'index.txt'), 'w')
 
    for i in sortedids:
        handle0.write("{}\n".format(ids2files[i]))
        index.write("{}\n".format(ids2files[i]))
        handle2.write('{}:{}\n'.format(i, ids2files[i]))
    handle0.close()
    handle2.close()
    index.close()
    
    '''
    Collect the captions for each image. These are stored in a 
    dictionary. We then sort the captions by the image_id keys. This means
    they can be written to disk in the same order as the names of the images
    (filenames.txt).
    '''
    handle1 = open(os.path.join(output_path, "{}_captions.txt".format(split)), "w")
    captions = defaultdict(list)
    for i in d['annotations']:
        captions[i['image_id']].append(i['caption'])
    
    sorted_captions = sorted(captions.keys())
    
    '''
    Keep a counter of how many times an image contains too many captions.
    '''
    ignored = 0
    for i in sorted_captions:
        for idx, c in enumerate(captions[i]):
            if idx < 5:
                handle1.write('{}\t{}\n'.format(c.replace("\n",""), str(i)))
            else:
                ignored += 1
    
    handle1.close()
    
    print("Ignored {} captions because there were more than 5 captions for a given image".format(ignored))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='extract-cnn-features')
    parser.add_argument('-r', '--coco_root', type=str, required=True,
                        help='Folder to image files i.e. /images/train')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='Batch size for forward pass.')
    parser.add_argument('-o', '--output_path', type=str, default='resnet50',
                        help='Path to output directory.')

    # Parse arguments
    args = parser.parse_args()
    coco_root = args.coco_root
    output_path = args.output_path
    img_path = os.path.join(output_path, 'imgfeats')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    # Create text files to handle caption-image correspondances
    #extract(coco_root, output_path, "train")
    extract_karpathy_valsets(coco_root, output_path)
    # Extract image features
    extract_img_feats(coco_root, "train", args.batch_size, img_path, indexname="train-index.txt")
    extract_img_feats(coco_root, "val", args.batch_size, img_path, indexname="val-index.txt")
    extract_img_feats(coco_root, "val", args.batch_size, img_path, savename='restval', indexname="restval-index.txt")
    extract_img_feats(coco_root, "val", args.batch_size, img_path, savename='test', indexname="test-index.txt")

