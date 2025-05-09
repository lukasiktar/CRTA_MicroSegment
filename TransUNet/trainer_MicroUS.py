import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import attention_BCE_loss
from torchvision import transforms
import cv2
import torch.nn.functional as F


def trainer_MicroUS(args, model, snapshot_path):
    from datasets.dataset_MicroUS import MicroUS_dataset, MultiscaleGenerator

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    hard_weight = args.weight

    db_train = MicroUS_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [MultiscaleGenerator(output_size=[[28, 28], [56, 56], [112, 112], [args.img_size, args.img_size]])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    print(model)

    #Segmentattion training parameters
    #Probati AdamW
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch_seg = args.max_epochs_seg
    max_iterations = args.max_epochs_seg * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    seg_iterator = tqdm(range(max_epoch_seg), ncols=70)

    #Classification training parameters


    criterion_cls=nn.BCEWithLogitsLoss()
    max_epoch_cls = args.max_epochs_cls
    cls_iterator=tqdm(range(max_epoch_cls), ncols=70)

    train_segmentation(args,model, trainloader, optimizer, seg_iterator, base_lr, hard_weight,writer,max_epoch_seg, max_iterations, snapshot_path,iter_num)

    train_classification(args, model, trainloader, optimizer, cls_iterator, base_lr ,criterion_cls, writer, max_epoch_cls, max_iterations, snapshot_path, iter_num)
    # with open("logit_logs.txt","w") as f:

    #     for epoch_num in iterator:
    #         epoch_loss = []
    #         for i_batch, sampled_batch in enumerate(tqdm(trainloader)):
    #             image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['label0'], sampled_batch['label1'], sampled_batch['label2'], sampled_batch['non_expert'], sampled_batch['non_expert0'], sampled_batch['non_expert1'], sampled_batch['non_expert2']
    #             image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = image_batch.cuda(), label_batch.cuda(), label0_batch.cuda(), label1_batch.cuda(), label2_batch.cuda(), non_expert_batch.cuda(), non_expert0_batch.cuda(), non_expert1_batch.cuda(), non_expert2_batch.cuda()
    #             # Calculate cls_label_batch
    #             cls_label_batch = (label_batch.sum(dim=(1,2)) > 0).float()
    #             outputs, out0, out1, out2, cls_output = model(image_batch)
    #             outputs = torch.sigmoid(outputs).squeeze(dim=1)
    #             out0 = torch.sigmoid(out0).squeeze(dim=1)
    #             out1 = torch.sigmoid(out1).squeeze(dim=1)
    #             out2 = torch.sigmoid(out2).squeeze(dim=1)

    #             loss0 = attention_BCE_loss(hard_weight, label0_batch, out0, non_expert0_batch, ks=1)
    #             loss1 = attention_BCE_loss(hard_weight, label1_batch, out1, non_expert1_batch, ks=2)
    #             loss2 = attention_BCE_loss(hard_weight, label2_batch, out2, non_expert2_batch, ks=3)
    #             loss3 = attention_BCE_loss(hard_weight, label_batch, outputs, non_expert_batch, ks=4)
    #             #print(f"cls_loss:{cls_loss}")
    #             cls_output = cls_output.squeeze(-1)
    #             f.write(f"Cls_Label_batch_val:  {cls_label_batch}")
    #             f.write(f"Cls_output: {torch.sigmoid(cls_output)}")
    #             cls_loss = criterion_cls(cls_output, cls_label_batch.float())
           
    #             #For  classification
    #             loss = loss0 + loss1 + loss2 + loss3 + 0.065*cls_loss
    #             #Standard
    #             #loss = loss0 + loss1 + loss2 + loss3

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
                
    #             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
    #             for param_group in optimizer.param_groups:
    #                 param_group['lr'] = lr_

    #             iter_num = iter_num + 1
    #             epoch_loss.append(loss)
    #             writer.add_scalar('info/lr', lr_, iter_num)
    #             writer.add_scalar('info/total_loss', loss, iter_num)

    #         average_loss = sum(epoch_loss)/len(epoch_loss)
    #         logging.info('Epoch %d : loss : %f' % (epoch_num, average_loss.item()))

    #         if epoch_num >= max_epoch - 1:
    #             save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
    #             torch.save(model.state_dict(), save_mode_path)
    #             logging.info("save model to {}".format(save_mode_path))
    #             iterator.close()
    #             break

    
    
            
            

    writer.close()
    logging.shutdown()

    return "Training Finished!"

def train_segmentation(args,model, trainloader, optimizer, iterator, base_lr, hard_weight,writer,max_epoch, max_iterations, snapshot_path, iter_num):
        #Freeze the classification head
        for param in model.classification_head.parameters():
            param.requires_grad=False

        for param in model.decoder.parameters():
            param.requires_grad = True
        for param in model.segmentation_head.parameters():
            param.requires_grad = True

        with open("logit_logs.txt","w") as f:

            for epoch_num in iterator:
                epoch_loss = []
                for i_batch, sampled_batch in enumerate(tqdm(trainloader)):
                    image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['label0'], sampled_batch['label1'], sampled_batch['label2'], sampled_batch['non_expert'], sampled_batch['non_expert0'], sampled_batch['non_expert1'], sampled_batch['non_expert2']
                    image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = image_batch.cuda(), label_batch.cuda(), label0_batch.cuda(), label1_batch.cuda(), label2_batch.cuda(), non_expert_batch.cuda(), non_expert0_batch.cuda(), non_expert1_batch.cuda(), non_expert2_batch.cuda()
                    # Calculate cls_label_batch
                    cls_label_batch = (label_batch.sum(dim=(1,2)) > 0).float()
                    outputs, out0, out1, out2, cls_output = model(image_batch)
                    outputs = torch.sigmoid(outputs).squeeze(dim=1)
                    out0 = torch.sigmoid(out0).squeeze(dim=1)
                    out1 = torch.sigmoid(out1).squeeze(dim=1)
                    out2 = torch.sigmoid(out2).squeeze(dim=1)

                    loss0 = attention_BCE_loss(hard_weight, label0_batch, out0, non_expert0_batch, ks=1)
                    loss1 = attention_BCE_loss(hard_weight, label1_batch, out1, non_expert1_batch, ks=2)
                    loss2 = attention_BCE_loss(hard_weight, label2_batch, out2, non_expert2_batch, ks=3)
                    loss3 = attention_BCE_loss(hard_weight, label_batch, outputs, non_expert_batch, ks=4)
                    #print(f"cls_loss:{cls_loss}")
                    cls_output = cls_output.squeeze(-1)
                    f.write(f"Cls_Label_batch_val:  {cls_label_batch}")
                    f.write(f"Cls_output: {torch.sigmoid(cls_output)}")
            
                    #For  classification
                    #loss = loss0 + loss1 + loss2 + loss3 + 0.065*cls_loss
                    #Standard
                    loss = loss0 + loss1 + loss2 + loss3

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

                    iter_num = iter_num + 1
                    epoch_loss.append(loss)
                    writer.add_scalar('info/lr', lr_, iter_num)
                    writer.add_scalar('info/total_loss', loss, iter_num)

                average_loss = sum(epoch_loss)/len(epoch_loss)
                logging.info('Epoch %d : loss : %f' % (epoch_num, average_loss.item()))

                if epoch_num >= max_epoch - 1:
                    save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    iterator.close()
                    return model

def train_classification(args, model, trainloader, optimizer, iterator, base_lr ,criterion_cls, writer, max_epoch, max_iterations, snapshot_path, iter_num):
    for param in model.parameters():
         param.requires_grad= False

    for param in model.classification_head.parameters():
            param.requires_grad=True

    with open("logit_logs_classification.txt","w") as f:

            for epoch_num in iterator:
                epoch_loss = []
                for i_batch, sampled_batch in enumerate(tqdm(trainloader)):
                    image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['label0'], sampled_batch['label1'], sampled_batch['label2'], sampled_batch['non_expert'], sampled_batch['non_expert0'], sampled_batch['non_expert1'], sampled_batch['non_expert2']
                    image_batch, label_batch, label0_batch, label1_batch, label2_batch, non_expert_batch, non_expert0_batch, non_expert1_batch, non_expert2_batch = image_batch.cuda(), label_batch.cuda(), label0_batch.cuda(), label1_batch.cuda(), label2_batch.cuda(), non_expert_batch.cuda(), non_expert0_batch.cuda(), non_expert1_batch.cuda(), non_expert2_batch.cuda()
                    # Calculate cls_label_batch
                    cls_label_batch = (label_batch.sum(dim=(1,2)) > 0).float()
                    outputs, out0, out1, out2, cls_output = model(image_batch)
                                        
                    cls_output = cls_output.squeeze(-1)            
                    cls_loss = criterion_cls(cls_output, cls_label_batch.float())

                    f.write(f"Cls_Label_batch_val:  {cls_label_batch}")
                    f.write(f"Cls_output: {torch.sigmoid(cls_output)}")
           
                    #print(f"cls_loss:{cls_loss}")
                    cls_output = cls_output.squeeze(-1)
                    
            
                    #For classification
                    loss =0.5*cls_loss
                    

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

                    iter_num = iter_num + 1
                    epoch_loss.append(loss)
                    writer.add_scalar('info/lr', lr_, iter_num)
                    writer.add_scalar('info/total_loss', loss, iter_num)

                average_loss = sum(epoch_loss)/len(epoch_loss)
                logging.info('Epoch %d : loss : %f' % (epoch_num, average_loss.item()))

                if epoch_num >= max_epoch - 1:
                    save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    iterator.close()
                    continue