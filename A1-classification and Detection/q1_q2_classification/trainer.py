from __future__ import print_function

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
from voc_dataset import VOCDataset


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("saving model at ", filename)
    torch.save(model, filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    writer = SummaryWriter()
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)
            
            # TODO implement a suitable loss function for multi-label classification
            # This function should take in network `output`, ground-truth `target`, weights `wgt` and return a single floating point number
            # You are NOT allowed to use any pytorch built-in functions
            # Remember to take care of underflows / overflows when writing your function
            # # log softmax cross entropy
          
            # def log_softmax(x):
            #     m = torch.amax(x, axis=1).expand(x.size()[1], -1).T
            #     return x - m - torch.log(torch.sum(torch.exp(x - m), axis=1)).expand(x.size()[1], -1).T

            # loss = - torch.mean(wgt*log_softmax(output)*log_softmax(target))

            # fn = torch.nn.CrossEntropyLoss()
            # loss = torch.mean(wgt*fn(output.softmax(dim=1), target.softmax(dim=1)))

            # fn = torch.nn.BCEWithLogitsLoss(weight=wgt)
            # loss = fn(output, target)

            # loss.to(args.device)

            # binary cross entropy loss: multi-label classification
            sigmoid = torch.nn.Sigmoid()
            loss = torch.sum(- wgt * (target * torch.log(sigmoid(output))+(1-target)*torch.log(1-sigmoid(output))))

            loss.backward()
            
            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                
                # Log gradients
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)

            optimizer.step()
            
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                print("map: ", map)
                writer.add_scalar("map", map, cnt)
                model.train()
            
            cnt += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # save model # change back!!
        # if save_this_epoch(args, epoch):
        #     save_model(epoch, model_name, model)
        if epoch == args.epochs-1:
            save_model(epoch, model_name, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
