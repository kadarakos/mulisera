from __future__ import print_function
import os
import pickle
import itertools 
from data2 import get_test_loader
import time
import numpy
import pandas
from vocab import Vocabulary
import torch
from torch.autograd import Variable
from model import VSE, order_sim
from collections import OrderedDict
import random
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb = model.forward_emb(images, captions, lengths,
                                             volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = numpy.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = numpy.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # measure accuracy and record loss
        val_loss = model.forward_loss(img_emb, cap_emb)
        val_loss = float(val_loss.detach().cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions

    return img_embs, cap_embs, val_loss


def sentencepair_eval(model, sentencepair_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()
    val_loss = 0
    for i, (capA, capB, lenA, lenB) in enumerate(sentencepair_loader):
        # make sure val logger is used
        model.logger = val_logger

        captionsA = Variable(capA, requires_grad=False)
        captionsB = Variable(capB, requires_grad=False)
        if torch.cuda.is_available():
            captionsA = captionsA.cuda()
            captionsB = captionsB.cuda()
        # Create permute and inverse permute indices t so t on length
        indsA = numpy.argsort(numpy.array(lenA))
        indsB = numpy.argsort(numpy.array(lenB))
        revA  = numpy.zeros(len(lenA), dtype='int')
        revB  = numpy.zeros(len(lenA), dtype='int')
        for j in range(len(lenA)):
            revA[indsA[j]] = j
            revB[indsB[j]] = j
        indsA, indsB = torch.LongTensor(indsA), torch.LongTensor(indsB)
        revA, revB = torch.LongTensor(revA), torch.LongTensor(revB)
        if torch.cuda.is_available():
            indsA, indsB = indsA.cuda(), indsB.cuda()
            revA, revB = revA.cuda(), revB.cuda()
        capA_emb = model.txt_enc(captionsA[indsA], sorted(lenA, reverse=True))
        capB_emb = model.txt_enc(captionsB[indsB], sorted(lenB, reverse=True))
        model.optimizer.zero_grad()
        val_loss += model.forward_loss(capA_emb[revA], capB_emb[revB]).data
        # compute the embeddings
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, sentencepair_loader.length, batch_time=batch_time,
                        e_log=str(model.logger)))

        if i * sentencepair_loader.batch_size >= sentencepair_loader.length:
            break
        del capA, capB
    return val_loss / i

def run_eval(model, data_loader, fold5, opt, loader_lang):
    print('Computing results for {}...'.format(loader_lang))
    logs = {}
    img_embs, cap_embs, val_loss = encode_data(model, data_loader)
    n_caps = 5
    print(img_embs.shape, cap_embs.shape)
    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, cap_embs, measure=opt.measure, n=n_caps, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs,
                      measure=opt.measure, n=n_caps, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        r = (loader_lang,) + r + (ar,)  # python tuple concat
        ri = (loader_lang,) + ri + (ari,)  # python tuple concat
        logs["i2t"] = r[1:]
        logs["t2i"] = ri[1:]
        print("rsum: %.1f" % rsum)
        #print("Average i2t Recall: %.1f" % ar)
        print(" %s Image to text: R@1 %.1f | R@5 %.1f | R@10 %.1f | Medr %.1f | Meanr %.1f | Average %.1f" % r)

        #print("Average t2i Recall: %.1f" % ari)
        print(" %s Text to image: R@1 %.1f | R@5 %.1f | R@10 %.1f | Medr %.1f | Meanr %.1f | Average %.1f" % ri)
        #print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
        with open(os.path.join(opt.logger_name, '{}_img2cap.pkl'.format(r[0])), "w") as f:
            pickle.dump(logs, f)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                         cap_embs[i * 5000:(i + 1) *
                                  5000], measure=opt.measure, n=n_caps,
                         return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                           cap_embs[i * 5000:(i + 1) *
                                    5000], measure=opt.measure, n=n_caps,
                           return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(numpy.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')
    return img_embs, cap_embs


def evalrank(model_path, data_set, split='dev', fold5=False, 
             lang=None, caption_rank=False, dump_word_embeddings=False,
             dump_image_embeddings=False, dump_caption_embeddings=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    datasets = data_set.split('-')
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    # Never use undersample when testing.
    opt.undersample = False
    print(opt)
    
    #Load vocabulary used by the model
    with open(os.path.join(opt.vocab_path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
        opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])
    
    emb_dict = {}
    for name in datasets:
        loader = get_test_loader(name, split, opt.batch_size, lang_prefix=False, downsample=False)
        loader.dataset.vocab = vocab
        img_emb, cap_emb = run_eval(model, loader, fold5, opt, name)
        if caption_rank:
            emb_dict[name] = cap_emb
        if dump_image_embeddings:
            path = os.path.join(os.path.dirname(model_path), 'image_embeddings.vec')
            numpy.save(path, img_emb)	
        if dump_caption_embeddings:
            path = os.path.join(os.path.dirname(model_path), 'caption_embeddings.vec')
            numpy.save(path, cap_emb)	
    if caption_rank:
        for l1, l2 in itertools.permutations(emb_dict.keys(), 2):
            print(l1,l2)
            ca, cb = emb_dict[l1], emb_dict[l2]
            r, rt = i2t(ca, cb, measure=opt.measure, n=5, return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            rsum = r[0] + r[1] + r[2] 
            r = (l1,l2,) +  r + (ar,) 
            print("rsum: %.1f" % rsum)
            print("Caption-Caption retrieval %s-%s : R@1 %.1f | R@5 %.1f | R@10 %.1f | Medr %.1f | Meanr %.1f | Average %.1f" % r)
    
    if dump_word_embeddings:
        v_sorted = sorted(vocab.idx2word.items(), key=lambda x: x[0])
        index_col = map(lambda x: x[1], v_sorted)    
        data = model.txt_enc.embed.weight.detach().cpu().numpy()
        frame = pandas.DataFrame(data=data, columns=None, index=index_col)
        path = os.path.join(os.path.dirname(model_path), 'word_embeddings.vec')
        frame.to_csv(path, sep=' ', header=False, encoding = 'utf-8')




def i2t(images, captions, npts=None, n=5, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / n
    index_list = []
    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[n * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], n * (index + bs))
                im2 = images[n * index:mx:n]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(n * index, n * index + n, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, n=5, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / n
    ims = numpy.array([images[i] for i in range(0, len(images), n)])
    ranks = numpy.zeros(n * npts)
    top1 = numpy.zeros(n * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[n * index:n * index + n]

        # Compute scores
        if measure == 'order':
            bs = 100
            if n * index % bs == 0:
                mx = min(captions.shape[0], n * index + bs)
                q2 = captions[n * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (n * index) % bs:(n * index) % bs + n].T
        else:
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[n * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[n * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", type=str, default='m30ken-m30kde',
                        help='For now the eval is setup to run on M30K task 2.')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--fold5", action="store_true")
    parser.add_argument("--split", default="val")
    parser.add_argument("--caption_rank", action="store_true", 
                       help="Run cross-lingual sentenceranking experiment")
    parser.add_argument("--dump_word_embeddings", action="store_true", 
                       help="Save word embeddings to model directory.") 
    parser.add_argument("--dump_image_embeddings", action="store_true", 
                       help="Save word embeddings to model directory.") 
    parser.add_argument("--dump_caption_embeddings", action="store_true", 
                       help="Save word embeddings to model directory.") 
    args = parser.parse_args()
    evalrank(args.model_path, data_set=args.data_set, split=args.split,
             fold5=args.fold5, caption_rank=args.caption_rank,
             dump_word_embeddings=args.dump_word_embeddings,
	     dump_image_embeddings=args.dump_image_embeddings,
	     dump_caption_embeddings=args.dump_caption_embeddings)
