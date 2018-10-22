import json
from collections import defaultdict

def extract(split="val"):
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
    print("Loading {}".format('annotations/captions_{}2014.json'.format(split)))
    d = json.load(open('annotations/captions_{}2014.json'.format(split)))
    
    ids2files = dict()
    
    ''' 
    Collect the list of image filenames and their IDs.
    Then sort the image IDs in ascending order.
    '''
    for i in d['images']:
        ids2files[i['id']] = i['file_name']
    
    sortedids = sorted(ids2files.keys())
    
    '''
    Write the names of the files onto disk (filenames.txt),
    write the mapping of imageID:filename to disk (ids2file.txt)
    '''
    handle0 = open('{}_filenames.txt'.format(split), 'w')
    handle2 = open("{}_ids2files.txt".format(split), "w")
    for i in sortedids:
        handle0.write("{}\n".format(ids2files[i]))
        handle2.write('{}:{}\n'.format(i, ids2files[i]))
    handle0.close()
    handle2.close()
    
    '''
    Collect the captions for each image. These are stored in a 
    dictionary. We then sort the captions by the image_id keys. This means
    they can be written to disk in the same order as the names of the images
    (filenames.txt).
    '''
    handle1 = open("{}_captions.txt".format(split), "w")
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
    extract("val")
    extract("train")
