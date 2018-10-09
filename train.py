import pickle
import os
import time
import shutil
import random
import numpy as np
import torch
import torch.utils.data as torchdata
import data
from vocab import Vocabulary  # NOQA
from model import VSE
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, sentencepair_eval
from torch.autograd import Variable
import logging
import tensorboard_logger as tb_logger
from torch.nn.utils.clip_grad import clip_grad_norm
import argparse

rseed = 41376566

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=".",
                        help='path to datasets')
    parser.add_argument('--data_name', default='m30k',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k|m30k')
    parser.add_argument('--lang', default='en',
                        help='Which language(s) to use from m30k, en-de, trains on en+de.')
    parser.add_argument('--sentencepair', action='store_true',
                        help='Train caption-caption ranking as well.')
    parser.add_argument('--sentencepair_p', default=0.5, type=float,
                        help='Probability of training on caption-caption and not image-caption.')
    parser.add_argument('--primary', default=None,
                        help='Which language to monitor for early stopping. Multiple with l1-l2-l3')
    parser.add_argument('--undersample', action='store_true',
                        help='Pick only one of the 5 possilbe captions for m30k task 2.')
    parser.add_argument('--half', action='store_true',
                        help='Use only half of the M30K from task 2.')
    parser.add_argument('--disaligned', action='store_true',
                        help='Use only half of the M30K from task 2.')
    parser.add_argument('--lang_prefix', action='store_true',
                        help='Put the language id infront of each word to split vocabularies.')
    parser.add_argument('--vocab_path', default='.',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--patience', default=10, type=int,
                        help='Number of validation steps to tolerate without improvement.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--bidi', action='store_true',
                        help='Run BiGRU instead of GRU.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_path', default='.', 
                        help='Path where to save the model and Tensorboard log.')
    parser.add_argument('--logger_name',
                        help='Name of the folder where to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--sum_violation', dest="max_violation", action='store_false',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='vgg19',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true',
                        help='Ensure the training is always done in '
                        'train mode (Not recommended).')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed.')
    opt = parser.parse_args()

    if torch.__version__ >= "0.3":
        opt.reset_train = True

    opt.vocab_path = os.path.join(opt.vocab_path, "vocab")
    if opt.logger_name is None:
        name = "lang{}_half-{}_undersample-{}_disaligned-{}_sentencepair-{}_primary-{}_epochs-{}"
        name = name.format(opt.lang, opt.half, opt.undersample, opt.disaligned, opt.sentencepair, opt.primary, opt.num_epochs)
        opt.logger_name = os.path.join(opt.data_name, name)
            
    opt.logger_name = os.path.join(opt.logger_path, opt.logger_name, str(opt.seed))
    print(opt)
    random.seed(rseed+opt.seed)
    np.random.seed(rseed+opt.seed)
    torch.cuda.manual_seed(rseed+opt.seed)
    torch.cuda.manual_seed_all(rseed+opt.seed)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)
    
    # For multi30k compute vocabulary mappings on the fly.
    if opt.data_name == "m30k":
        vocab = None
        langs = opt.lang.split("-")
    # Load Vocabulary Wrapper for COCO or F30K
    else:
        vocab = pickle.load(open(os.path.join(
            opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
        opt.vocab_size = len(vocab)
        langs = [opt.data_name]
    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.crop_size, opt.batch_size, opt.workers, opt)
    # Construct the model
    model = VSE(opt)
    print(model.txt_enc)
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model, "")
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if len(langs) == 1 or opt.data_name != 'm30k':

        # Train the Model on a single data set
        best_rsum = 0
        model.train_start()
        for epoch in range(opt.num_epochs):
            if opt.reset_train:
                # Always reset to train mode, this is not the default behavior
                model.train_start()
            adjust_learning_rate(opt, model.optimizer, epoch)

            # train for one epoch
            train(opt, train_loader, model, epoch, val_loader)

            # evaluate on validation set
            rsum = validate(opt, val_loader, model, langs[0])

            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, prefix=opt.logger_name + '/')
            if is_best:
                patience_count = 0
                print("New best: {}".format(best_rsum))
            else:
                patience_count += 1
                print("No improvement in {}".format(patience_count))
                if patience_count == opt.patience:
                    print("No improvement in {} epochs, stoppin".format(patience_count))
                    break

    else:
        joint_train(opt, train_loader, model, val_loader)

def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    best_score = 0
    end = time.time()
    for i, train_data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        loss = model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        tb_logger.log_value('train', float(loss.detach().cpu().numpy()), step=model.Eiters)
        tb_logger.log_value('c2c', 0., step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def joint_train(opt, train_loader, model, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    best_score = 0
    stop = False
    iters = 0
    langs = opt.lang.split("-")
    # switch to train mode
    model.train_start()
    # Sentencepair is always the last data loader in the list
    if opt.sentencepair:
        sentencepair_loader = train_loader.pop()
        sentencepair_loader_val = val_loader.pop()
    # Call iterator on the DatasetLoader returning DatasetLoaderIterator
    train_loader_its = list(map(iter, train_loader))
    end = time.time()
    patience_count = 0
    if opt.primary:
        primary = opt.primary.split("-")
    while not stop:
        iters += 1
        # Pick a data set and batch
        ind = random.randint(0, len(train_loader)-1)
        train_cap2cap = random.random() < opt.sentencepair_p and opt.sentencepair
        if opt.reset_train:
            # Always reset to train mode, this is not the default behavior
            model.train_start()
        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        loss = None
        loss_c2c = None
        # Train caption-caption ranking.
        if train_cap2cap and opt.sentencepair:
            capA, capB, lenA, lenB = next(sentencepair_loader)
            captionsA = Variable(capA)
            captionsB = Variable(capB)
            if torch.cuda.is_available():
                captionsA = captionsA.cuda()
                captionsB = captionsB.cuda()
            # Create permute and inverse permute indices t so t on length
            indsA = np.argsort(np.array(lenA))
            indsB = np.argsort(np.array(lenB))
            revA  = np.zeros(len(lenA), dtype='int')
            revB  = np.zeros(len(lenA), dtype='int')
            for i in range(len(lenA)):
                revA[indsA[i]] = i
                revB[indsB[i]] = i
            indsA, indsB = torch.LongTensor(indsA), torch.LongTensor(indsB)
            revA, revB = torch.LongTensor(revA), torch.LongTensor(revB)
            if torch.cuda.is_available():
                indsA, indsB = indsA.cuda(), indsB.cuda()
                revA, revB = revA.cuda(), revB.cuda()
            model.Eiters += 1
            model.logger.update('Eit', model.Eiters)
            # Pass length sorted captions for encoding
            capA_emb = model.txt_enc(captionsA[indsA], sorted(lenA, reverse=True))
            capB_emb = model.txt_enc(captionsB[indsB], sorted(lenB, reverse=True))
            model.optimizer.zero_grad()
            # Unsort captions for the loss computation
            loss_c2c = model.forward_loss(capA_emb[revA], capB_emb[revB])
            # compute gradient and do SGD step
            loss_c2c.backward()
            if model.grad_clip > 0:
                clip_grad_norm(model.params, model.grad_clip)
            model.optimizer.step()
            # Don't count this as an iter
        # Train image-sentence ranking.
        else:
            tloader = train_loader_its[ind]
            # Call next element of if its exhausted re-init the DatasetLoaderIterators
            try:
                train_data = next(tloader)
            except StopIteration:
                train_loader_its = map(iter, train_loader)
                tloader = train_loader_its[ind]
                train_data = next(tloader)
            loss = model.train_emb(*train_data)
        # Train with sentence-pair ranking batch.
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(batch_time=batch_time,
                        data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('step', iters, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)

        if loss is not None:
            tb_logger.log_value('train', float(loss.detach().cpu().numpy()), step=model.Eiters)

        if loss_c2c is not None:
            tb_logger.log_value('c2c', float(loss_c2c.detach().cpu().numpy()), step=model.Eiters)

        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            total_score = 0
            for l, vloader in zip(langs, val_loader):
                with torch.no_grad():
                    score = validate(opt, vloader, model, l)

                if opt.primary:
                    if l in primary:
                        total_score += score
                else:
                    total_score += score
            # Compute val loss on sentencepair task
            if opt.sentencepair:
                # val_loss = sentencepair_eval(model, sentencepair_loader_val)
                # tb_logger.log_value('valid_c2c', val_loss, step=model.Eiters)
                # print('Sentence Pair Val Loss {}'.format(val_loss))
                tb_logger.log_value('valid_c2c', 0., step=model.Eiters)
            else:
                tb_logger.log_value('valid_c2c', 0., step=model.Eiters)

            if total_score > best_score:
                is_best = True
                print("New best: {}".format(total_score))
                best_score = total_score
                patience_count = 0
            else:
                patience_count += 1
                is_best = False
                print("No improvement in {}".format(patience_count))
                if patience_count >= opt.patience:
                    print("No improvement in {} evaluations, stopping".format(patience_count))
                    break
            save_checkpoint({
                'iter': iters,
                'model': model.state_dict(),
                'best_rsum': best_score,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, prefix=opt.logger_name + '/')

    print("Finished trained. Best score: {}".format(best_score))


def validate(opt, val_loader, model, lang, n=5):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, val_loss = encode_data(
        model, val_loader, opt.log_step, logging.info)
    if lang in ['en', 'de']:
        n = 5
    else:
        n = 1
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure=opt.measure, n=n)
    logging.info("%s Image to text: R@1 %.1f | R@5 %.1f | R@10 %.1f | Medr %.1f | Meanr %.1f" %
                 (lang, r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, measure=opt.measure, n=n)
    logging.info("%s Text to image: R@1 %.1f | R@5 %.1f | R@10 %.1f | Medr %.1f | Meanr %.1f" %
                 (lang, r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)
    tb_logger.log_value('valid', val_loss, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
