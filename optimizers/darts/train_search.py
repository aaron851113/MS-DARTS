import argparse
import glob
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from torch.autograd import Variable

sys.path.insert(0, '../../')
from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import NasbenchWrapper

from optimizers.darts import utils
from optimizers.darts.architect import Architect
from optimizers.darts.model_search import Network

from attacker.perturb_ms import Mean_Shift_alpha

from optimizers.analyze import Analyzer
from copy import deepcopy
from numpy import linalg as LA
import csv
#from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../../data',
                    help='location of the darts corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=80, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=9, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random_ws seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training darts')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--output_weights', type=bool, default=True, help='Whether to use weights on the output nodes')
parser.add_argument('--search_space', choices=['1', '2', '3'], default='1')
parser.add_argument('--warm_start_epochs', type=int, default=0,
                    help='Warm start one-shot model before starting architecture updates.')
parser.add_argument('--perturb_alpha', type=str, default='mean_shift', help='perturb for alpha')
parser.add_argument('--epsilon_alpha', type=float, default=0.25, help='max epsilon for alpha')
parser.add_argument('--bandwidth', type=float, default=1.0, help='max bandwidth for mean shift')
parser.add_argument('--samplingN', type=int, default=3, help='sampling points nums for mean-shift')
parser.add_argument('--iteration', type=int, default=2, help='iteration nums for mean-shift')

args = parser.parse_args()

args.save = '../../experiments/darts/search_space_{}/search-{}-{}-s{}'.format(
    args.search_space, args.save,
    time.strftime("%Y%m%d-%H%M%S"),
    args.search_space)

if args.unrolled:
    args.save += '-unrolled'
if not args.weight_decay == 3e-4:
    args.save += '-weight_l2-' + str(args.weight_decay)
if not args.arch_weight_decay == 1e-3:
    args.save += '-alpha_l2-' + str(args.arch_weight_decay)
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)
if not args.perturb_alpha == 'none':
    args.save += '-' + args.perturb_alpha + '-e' + str(args.epsilon_alpha)
args.save += '-h' + str(args.bandwidth)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# Dump the config of the run
with open(os.path.join(args.save, 'config.json'), 'w') as fp:
    json.dump(args.__dict__, fp)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
#writer = SummaryWriter('/root/notebooks/tensorflow/logs')

CIFAR_CLASSES = 10


def main():
    if not 'debug' in args.save:
        from nasbench_analysis import eval_darts_one_shot_model_in_nasbench as naseval
    # Select the search space to search in
    if args.search_space == '1':
        search_space = SearchSpace1()
    elif args.search_space == '2':
        search_space = SearchSpace2()
    elif args.search_space == '3':
        search_space = SearchSpace3()
    else:
        raise ValueError('Unknown search space')
    
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
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    
    if args.perturb_alpha == 'none':
        perturb_alpha = None
    elif args.perturb_alpha == 'pgd_linf':
        perturb_alpha = Linf_PGD_alpha
    elif args.perturb_alpha == 'random':
        perturb_alpha = Random_alpha
    elif args.perturb_alpha == 'mean_shift':
        perturb_alpha = Mean_Shift_alpha

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, 
                    output_weights=args.output_weights, steps=search_space.num_intermediate_nodes, 
                    search_space=search_space)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

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

    analyzer = Analyzer(model, args)
    architect = Architect(model, args)
    
    with open(args.save+str(args.perturb_alpha)+'-b'+str(args.bandwidth)+'-e'+str(args.epsilon_alpha)+'.csv', 'w', newline='') as csvFile:
        # 建立 CSV 檔寫入器
        csv_writer = csv.writer(csvFile)
        csv_writer.writerow(['eigenvalue','test error','valid error','runtime','params','train_obj','valid_obj'])
        
        for epoch in range(args.epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            if args.cutout:
                # increase the cutout probability linearly throughout search
                train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
                logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                            train_transform.transforms[-1].cutout_prob)
            else:
                logging.info('epoch %d lr %e', epoch, lr)

            if args.perturb_alpha:
                T = args.iteration
                N = args.samplingN
                epsilon_alpha = (args.epsilon_alpha/10) + (args.epsilon_alpha -  (args.epsilon_alpha/10)) * epoch / args.epochs
                bandwidth = args.bandwidth
                logging.info('epoch %d epsilon_alpha %e bandwidth %e', epoch, epsilon_alpha, bandwidth)

            # Save the one shot model architecture weights for later analysis
            arch_filename = os.path.join(args.save, 'one_shot_architecture_{}.obj'.format(epoch))
            with open(arch_filename, 'wb') as filehandler:
                numpy_tensor_list = []
                for tensor in model.arch_parameters():
                    numpy_tensor_list.append(tensor.detach().cpu().numpy())
                pickle.dump(numpy_tensor_list, filehandler)

            # # Save the entire one-shot-model
            # filepath = os.path.join(args.save, 'one_shot_model_{}.obj'.format(epoch))
            # torch.save(model.state_dict(), filepath)

            if not 'debug' in args.save:
                for i in numpy_tensor_list:
                    logging.info(str(i))

            # training
            train_acc, train_obj, ev = train(train_queue, valid_queue, model, architect, criterion, 
                                             optimizer, lr, epoch, analyzer, perturb_alpha, epsilon_alpha, bandwidth, T, N)
            logging.info('train_acc %f', train_acc)
            logging.info('eigenvalue %f', ev)
            #writer.add_scalar('Acc/train', train_acc, epoch)
            #writer.add_scalar('Obj/train', train_obj, epoch)
            #writer.add_scalar('Analysis/eigenvalue', ev, epoch)

            # validation
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)
            #writer.add_scalar('Acc/valid', valid_acc, epoch)
            #writer.add_scalar('Obj/valid', valid_obj, epoch)

            utils.save(model, os.path.join(args.save, 'weights.pt'))

            if not 'debug' in args.save:
                # benchmark
                logging.info('STARTING EVALUATION')
                test, valid, runtime, params = naseval.eval_one_shot_model(
                    config=args.__dict__, model=arch_filename)

                index = np.random.choice(list(range(3)))
                test, valid, runtime, params = np.mean(test), np.mean(valid), np.mean(runtime), np.mean(params)
                logging.info('TEST ERROR: %.3f | VALID ERROR: %.3f | RUNTIME: %f | PARAMS: %d'
                            % (test, valid, runtime, params))
                #writer.add_scalar('Analysis/test', test, epoch)
                #writer.add_scalar('Analysis/valid', valid, epoch)
                #writer.add_scalar('Analysis/runtime', runtime, epoch)
                #writer.add_scalar('Analysis/params', params, epoch)
            csv_writer.writerow([ev,test,valid,runtime,params,train_obj,valid_obj])
        #writer.close()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, 
          analyzer, perturb_alpha, epsilon_alpha, bandwidth, T, N):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)

        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        # Allow for warm starting of the one-shot model for more reliable architecture updates.
        if epoch >= args.warm_start_epochs:
            architect.step(input, target, input_search, target_search, lr, optimizer, args.unrolled)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()
        
        # print('before softmax', model.arch_parameters())
        model.softmax_arch_parameters()
            
        # perturb on alpha
        # print('after softmax', model.arch_parameters())
        if perturb_alpha:
            perturb_alpha(model, input, target, epsilon_alpha, bandwidth, T, N)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()
        # print('afetr perturb', model.arch_parameters())
        
        logits = model(input, updateType='weight')
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        model.restore_arch_parameters()
        # print('after restore', model.arch_parameters())

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                break
            
    # analyze
    _data_loader = deepcopy(train_queue)
    input, target = next(iter(_data_loader))

    input = input.cuda()
    target = target.cuda(non_blocking=True)
    
    H = analyzer.compute_Hw(input, target, input_search, target_search,
                            lr, optimizer, False)
    # g = analyzer.compute_dw(input, target, input_search, target_search,
    #                         lr, optimizer, False)
    # g = torch.cat([x.view(-1) for x in g])

    del _data_loader
    
    ev = max(LA.eigvals(H.cpu().data.numpy()))
    ev = np.linalg.norm(ev)

    return top1.avg, objs.avg, ev


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
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                if 'debug' in args.save:
                    break

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
