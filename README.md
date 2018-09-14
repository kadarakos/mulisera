# Multi-modal Multi-lingual image sentence ranking

This code is based on the [official code base](https://github.com/fartashf/vsepp) 
for ["VSE++: Improving Visual-Semantic Embeddings with Hard Negatives" 
(Faghri, Fleet, Kiros, Fidler.  2017)](https://arxiv.org/abs/1707.05612).

## Dependencies
We recommended to use Anaconda for the following packages.

* Python 2.7
* [PyTorch](http://pytorch.org/) (>0.2)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data

Download the Multi30K the caption data by cloneing the official repo:

```bash
git clone https://github.com/multi30k/dataset
```

The pre-computed image-features for Multi30K are available on [google-drive](https://drive.google.com/drive/folders/1I2ufg3rTva3qeBkEc-xDpkESsGkYXgCf).  

To run expmerinemts on COCO and F30K download the dataset files and pre-trained models.
Splits are the same as [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). 
The precomputed image features are from [here](https://github.com/ryankiros/visual-semantic-embedding/) and [here](https://github.com/ivendrov/order-embedding). 
To use full image encoders, download the images from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

```bash
wget http://www.cs.toronto.edu/~faghri/vsepp/vocab.tar
wget http://www.cs.toronto.edu/~faghri/vsepp/data.tar
wget http://www.cs.toronto.edu/~faghri/vsepp/runs.tar
```

## Experiments in the paper


All commands should be concatenated with 
`--data_name m30k --img_dim 2048 --max_violation --patience 10`  
and given a `--seed`.

### Tables 2-3


| Method    | Arguments |
| :-------: | :-------: |
| Monolingual English      | `--lang en --num_epochs 1000` |
| Monolingual German       | `--lang de --num_epochs 1000` |
| Bilingual    		   | `--lang en-de` 		   |
| Bilingual + c2c   	   | `--lang en-de --sentencepair` |

### Table 4

| Method    | Arguments |
| :-------: | :-------: |
| Monolingual English      | `--lang en1 --num_epochs 1000`   		  |
| Monolingual German       | `--lang de1 --num_epochs 1000`   		  |
| Bi-translation   	   | `--lang en1-de1` 		      		  |
| Bi-translation + c2c     | `--lang en1-de1 --sentencepair`  		  |
| Bi-comperable   	   | `--lang en-de --undersample`   		  |
| Bi-comperable + c2c      | `--lang en-de --undersample --sentencepair`  |

### Table 5

| Method    | Arguments |
| :-------: | :-------: |
| Full Monolingual English      | `--lang en --num_epochs 1000`   		  |
| Full Monolingual German       | `--lang de --num_epochs 1000`   		  |
| Half Monolingual English      | `--lang en --half --num_epochs 1000`   	  |
| Half Monolingual German       | `--lang de --half --num_epochs 1000`  	  |
| Bi-aligned	  	   	| `--lang en-de --half`         		  |
| Bi-aligned  + c2c  	   	| `--lang en-de --half --sentencepair`		  |
| Bi-disjoint	  	   	| `--lang en-de --half --disaligned`  		  |

### Table 6

| Method    | Arguments |
| :-------: | :-------: |
| Monolingual English         | `--lang en1 --num_epochs 1000`   		  |
| Monolingual German          | `--lang de1 --num_epochs 1000`   		  |
| Monolingual French          | `--lang fr --num_epochs 1000`   		  |
| Monolingual Czech	      | `--lang cs --num_epochs 1000`   		  |
| Multi-translation   	      | `--lang en1-de1-fr-cs` 		      		  |
| Multi-translation + c2c     | `--lang en1-de1-fr-cs --sentencepair`  		  |
| Multi-comperable   	      | `--lang en-de-fr-cs --undersample`   		  |
| Multi-comperable + c2c      | `--lang en-de-fr-cs --undersample --sentencepair` |

### Table 7

| Method    | Arguments |
| :-------: | :-------: |
| Monolingual French          | `--lang fr --num_epochs 1000`   		  |
| Monolingual Czech	      | `--lang cs --num_epochs 1000`   		  |
| Multilingual French  	      | `--lang en1-de1-fr-cs --primary fr`    		  |
| Multilingual Czech  	      | `--lang en1-de1-fr-cs --primary cs`    		  |
| + Comparable French  	      | `--lang en-de-fr-cs --primary fr`    		  |
| + Comparable Czech  	      | `--lang en-de-fr-cs --primary cs`    		  |
| + Comparable + c2c French   | `--lang en-de-fr-cs --primary fr --sentencepair`  |
| +Comparable + c2c Czech     | `--lang en-de-fr-cs --primary cs --sentencepair`  |


## Reference

If you found this code useful, please cite the following paper:

    @article{faghri2017vse++,
      title={VSE++: Improving Visual-Semantic Embeddings with Hard Negatives},
      author={Faghri, Fartash and Fleet, David J and Kiros, Jamie Ryan and Fidler, Sanja},
      journal={arXiv preprint arXiv:1707.05612},
      year={2017}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
