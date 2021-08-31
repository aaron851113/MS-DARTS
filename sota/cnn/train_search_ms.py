import os
import sys
sys.path.insert(0, '../../')
import time
import glob
import numpy as np
import torch
import optimizers.darts.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from sota.cnn.model_search import Network
from optimizers.darts.architect import Architect
from sota.cnn.spaces import spaces_dict

from attacker.perturb_ms import Mean_Shift_alpha

from flop_benchmark import get_model_infos
from model import Network as NetworkCIFAR

parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../../../data',help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=40, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--search_space', type=str, default='s5', help='searching space to choose from')
parser.add_argument('--perturb_alpha', type=str, default='ms', help='perturb for alpha')
parser.add_argument('--epsilon_alpha', type=float, default=0.25, help='max epsilon for alpha')
parser.add_argument('--bandwidth', type=float, default=1, help='max bandwidth for alpha')
parser.add_argument('--samplingN', type=int, default=3, help='sampling points nums for mean-shift')
parser.add_argument('--iteration', type=int, default=2, help='iteration nums for mean-shift')

args = parser.parse_args()

args.save = './{}/search-{}-{}-{}'.format(
    args.dataset, args.save, time.strftime("%m%d-%H%M%S"), args.search_space)

if args.search_space != 's5':
    args.cutout = True

if not args.perturb_alpha == 'none':
    args.save += '-' + args.perturb_alpha
if args.unrolled:
    args.save += '-unrolled'
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)
    
if not args.perturb_alpha == 'none':
    args.save += '-' + 'e' + str(args.epsilon_alpha) + '-' + 'h' + str(args.bandwidth) + '-' + 'N' + str(args.samplingN) + '-' + 'T' + str(args.iteration)

    
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10

def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    if args.perturb_alpha == 'none':
        perturb_alpha = None
    elif args.perturb_alpha == 'ms':
        perturb_alpha = Mean_Shift_alpha

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space])
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    if 'debug' in args.save:
        split = args.batch_size
        num_train = 2 * args.batch_size

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    
    architect = Architect(model, args)
    
    
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        if args.cutout:
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)
        
        bandwidth = args.bandwidth
        T = args.iteration
        N = args.samplingN
        epsilon_alpha = 0
        if args.perturb_alpha:
            epsilon_alpha = (args.epsilon_alpha/10) + (args.epsilon_alpha - (args.epsilon_alpha/10)) * epoch / args.epochs
            #epsilon_alpha = args.epsilon_alpha
            logging.info('epoch %d epsilon_alpha %e bandwidth %f ', epoch, epsilon_alpha, bandwidth)
        
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        
        ############################# Evaluation Architecture Paramss ####################################
        with torch.no_grad():
            cifar_model = NetworkCIFAR(36,10,20,False,genotype)
            cifar_model.drop_path_prob = 0
            cifar_model.eval()
            model_input = Variable(torch.randn(1,3,32,32).type(torch.FloatTensor), requires_grad=False)
            macs , params  = get_model_infos(cifar_model, model_input)
            evaluation_model_text = 'Params = {:.2f} M | MACs = {:.2f} M'.format(params,macs)
            logging.info(evaluation_model_text)
            del cifar_model, model_input, macs, params
        ############################# Evaluation Architecture Params #####################################
        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, perturb_alpha, epsilon_alpha, bandwidth, T, N, epoch)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
                        
        #utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, perturb_alpha, epsilon_alpha, bandwidth, T, N, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)
        
        if epoch >= 5 :
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()

        model.softmax_arch_parameters()
        
        # perturb on alpha
        if perturb_alpha:
            perturb_alpha(model, input, target, epsilon_alpha, bandwidth, T, N)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()
        
        logits = model(input, updateType='weight')
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        model.restore_arch_parameters()
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data , n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                break
    return  top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                if 'debug' in args.save:
                    break

    return top1.avg, objs.avg


if __name__ == '__main__':
    main() 
