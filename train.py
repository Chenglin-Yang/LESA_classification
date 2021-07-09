import argparse
import os
from tqdm import tqdm
import time
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboardX
import numpy as np
import lib
from lib.utils import adjust_learning_rate, cross_entropy_with_label_smoothing, \
    accuracy, save_model, load_model, resume_model
from mmcv import Config

best_val_acc = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Image classification')
    parser.add_argument('--dataset', default='imagenet1k',
                        help='Dataset names.')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='The number of classes in the dataset.')
    parser.add_argument('--train_dirs', default='./data/imagenet/train',
                        help='path to training data')
    parser.add_argument('--val_dirs', default='./data/imagenet/val',
                        help='path to validation data')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='input batch size for val')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument("--color_jitter", action='store_true', default=False,
                        help="To apply color augmentation or not.")
    parser.add_argument('--model', default='resnet50',
                        help='Model names.')
    parser.add_argument('--epochs', type=int, default=130,
                        help='number of epochs to train')
    parser.add_argument('--test_epochs', type=int, default=1,
                        help='number of internal epochs to test')
    parser.add_argument('--save_epochs', type=int, default=1,
                        help='number of internal epochs to save')
    parser.add_argument('--optim', default='sgd',
                        help='Model names.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--warmup_epochs', type=float, default=10,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.00008,
                        help='weight decay')
    parser.add_argument("--label_smoothing", action='store_true', default=False,
                        help="To use label smoothing or not.")
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='To use nesterov or not.')
    parser.add_argument('--work_dirs', default='./work_dirs',
                        help='path to work dirs')
    parser.add_argument('--name', default='resnet50',
                        help='the name of work_dir')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--lr_scheduler', type=str, default="cosine", choices=["linear", "cosine"],
                        help='how to schedule learning rate')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Test')
    parser.add_argument('--test_model', type=int, default=-1,
                        help="Test model's epochs")
    parser.add_argument('--test_weight', type=str, default=None, 
                        help='path to specific model weight')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--CutMixAug', action='store_true', 
                        help='flag to use augmentation used in CutMix paper')
    parser.add_argument('--beta', default=0, type=float, 
                        help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
    parser.add_argument('--cfg', type=str, default='./config/LESA_resnet.py')
    args = parser.parse_args()
    if not os.path.exists(args.work_dirs):
        os.system('mkdir -p {}'.format(args.work_dirs))
    args.log_dir = os.path.join(args.work_dirs, 'log')
    if not os.path.exists(args.log_dir):
        os.system('mkdir -p {}'.format(args.log_dir))
    args.log_dir = os.path.join(args.log_dir, args.name)
    args.work_dirs = os.path.join(args.work_dirs, args.name)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    return args


def val(model, val_loader, criterion, epoch, args, log_writer=False):
    global best_val_acc
    model.eval()
    
    val_loss = lib.AverageMeter('val_loss')
    val_accuracy_top1 = lib.AverageMeter('val_accuracy_top1')
    val_accuracy_top5 = lib.AverageMeter('val_accuracy_top5')

    if epoch == -1:
        epoch = args.epochs - 1

    with tqdm(total=len(val_loader),
              desc='Validate Epoch #{}'.format(epoch + 1)) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                # val_loss.update(criterion(output, target))
                # val_accuracy.update(accuracy(output, target))
                val_loss.update(criterion(output, target), output.shape[0])
                acc_top1, acc_top5 = lib.accuracy(output, target, topk=(1,5))
                val_accuracy_top1.update(acc_top1, output.shape[0])
                val_accuracy_top5.update(acc_top5, output.shape[0])

                t.update(1)

    print("\nloss: {}, accuracy_top1: {:.2f}, accuracy_top5: {:.2f}, best acc: {:.2f}\n".format(
        val_loss.avg.item(), 
        val_accuracy_top1.avg.item(),
        val_accuracy_top5.avg.item(),
        max(best_val_acc, val_accuracy_top1.avg))
    )

    if val_accuracy_top1.avg > best_val_acc and log_writer:
        save_model(model, None, -1, args)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy_top1', val_accuracy_top1.avg, epoch)
        log_writer.add_scalar('val/accuracy_top5', val_accuracy_top1.avg, epoch)
        best_val_acc = max(best_val_acc, val_accuracy_top1.avg)
        log_writer.add_scalar('val/best_acc', best_val_acc, epoch)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train(model, train_loader, optimizer, criterion, epoch, log_writer, args):
    #atorch.autograd.set_detect_anomaly(True)
    train_loss = lib.AverageMeter('train_loss')
    train_accuracy_top1 = lib.AverageMeter('train_accuracy_top1')
    train_accuracy_top5 = lib.AverageMeter('train_accuracy_top5')

    model.train()
    N = len(train_loader)
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        lr_cur = adjust_learning_rate(args, optimizer, epoch, batch_idx, N, type=args.lr_scheduler)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        
        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(data.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
            data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
            # compute output
            output = model(data)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            output = model(data)
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.detach(), output.shape[0])
        acc_top1, acc_top5 = lib.accuracy(output.detach(), target, topk=(1,5))
        train_accuracy_top1.update(acc_top1, output.shape[0])
        train_accuracy_top5.update(acc_top5, output.shape[0])

        if (batch_idx + 1) % 20 == 0:
            memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            used_time = time.time() - start_time
            eta = used_time / (batch_idx + 1) * (N - batch_idx)
            eta = str(datetime.timedelta(seconds=int(eta)))
            training_state = '  '.join(['Epoch: {}', '[{} / {}]', 'eta: {}', 'lr: {:.9f}', 'max_mem: {:.0f}',
                                        'loss: {:.3f}', 'accuracy_top1: {:.3f}', 'accuracy_top5: {:.3f}'])
            training_state = training_state.format(
                epoch + 1, batch_idx + 1, N, eta, lr_cur, memory,
                train_loss.avg.item(), 
                train_accuracy_top1.avg.item(),
                train_accuracy_top5.avg.item(),
            )
            print(training_state)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy_top1', train_accuracy_top1.avg, epoch)
        log_writer.add_scalar('train/accuracy_top5', train_accuracy_top5.avg, epoch)


def test_net(args):
    print("Init...")
    config = Config.fromfile(args.cfg).config
    _, _, val_loader, _ = lib.build_dataloader(args)
    model = lib.build_model(args, cfg=config)
    load_model(model, args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        model.cuda()
    if args.label_smoothing:
        criterion = cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()

    print("Start testing...")
    val(model, val_loader, criterion, args.test_model, args)


def train_net(args):
    print("Init...")
    config = Config.fromfile(args.cfg).config
    log_writer = tensorboardX.SummaryWriter(args.log_dir)
    train_loader, _, val_loader, _ = lib.build_dataloader(args)
    model = lib.build_model(args, cfg=config)
    print('Parameters:', sum([np.prod(p.size()) for p in model.parameters()]))

    model = torch.nn.DataParallel(model)
    optimizer = lib.build_optimizer(args, model)

    epoch = 0
    if args.resume:
        epoch = resume_model(model, optimizer, args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    cudnn.benchmark = True

    if args.label_smoothing:
        criterion = cross_entropy_with_label_smoothing
    else:
        criterion = nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda()
    
    print("Start training...")
    while epoch < args.epochs:
        train(model, train_loader, optimizer, criterion, epoch, log_writer, args)

        if (epoch + 1) % args.test_epochs == 0:
            val(model, val_loader, criterion, epoch, args, log_writer)

        if (epoch + 1) % args.save_epochs == 0:
            save_model(model, optimizer, epoch, args)

        epoch += 1

    save_model(model, optimizer, epoch - 1, args)


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        test_net(args)
    else:
        train_net(args)


