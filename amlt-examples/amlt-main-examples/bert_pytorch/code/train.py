
import numpy as np
import torch
import utils
import torch.nn.functional as F
from sklearn import metrics
import time
import os
from shutil import copyfile
from pytorch_pretrained_bert.optimization import BertAdam


def train(config, model, train_iter, dev_iter, test_iter):
    """
    Train model
    """

    # get the model's parameters
    param_optimizer = list(model.named_parameters())
    # which parameters do not need to be decay
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
    ]

    optimizer = BertAdam(params=optimizer_grouped_parameters,
                         schedule='warmup_linear',
                         lr=config.learning_rate,
                         warmup=config.warmup,
                         t_total=len(train_iter) * (config.num_epochs - config.start_epoch))

    if config.continue_training:
        optimizer.load_state_dict(config.checkpoint['optimizer_state_dict'])
        config.hvd.broadcast_object(0, root_rank=0)

    optimizer = config.hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    start_time = time.time()
    # activate BatchNormalization & dropout
    model.train()

    total_batch = 0
    # Best loss in dev
    dev_best_loss = config.start_loss

    for epoch in range(config.start_epoch, config.num_epochs):
        if config.hvd.rank() == 0:
            print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (token_ids, label, seq_len, mask) in enumerate(train_iter):
            token_ids = token_ids.to(config.device)
            label_gpu = label.to(config.device)
            seq_len = seq_len.to(config.device)
            mask = mask.to(config.device)

            outputs = model(token_ids, seq_len, mask)
            model.zero_grad()
            loss = F.cross_entropy(outputs, label_gpu)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    if config.hvd.rank() == 0:
                        torch.save({'epoch': epoch,
                                    'loss': dev_loss.item(),
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()}, config.checkpoint_path)
                    improve = '*'
                else:
                    improve = ''
                time_dif = utils.get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>5.4}, Val Loss: {2:>5.4}, Val Acc: {3:>6.2%}, Time: {4} {5} '
                loss = config.hvd.allreduce(loss)
                if config.hvd.rank() == 0:
                    print(msg.format(total_batch, loss.item(), dev_loss, dev_acc, time_dif, improve))
            total_batch = total_batch + 1

    if config.hvd.rank() == 0:
        if os.path.exists(config.checkpoint_path) and config.checkpoint_path != config.save_path:
            copyfile(config.checkpoint_path, config.save_path)
            os.remove(config.checkpoint_path)


def evaluate(config, model, dev_iter, test=False):

    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for token_ids, label, seq_len, mask in dev_iter:

            token_ids = token_ids.to(config.device)
            label_gpu = label.to(config.device)
            seq_len = seq_len.to(config.device)
            mask = mask.to(config.device)

            outputs = model(token_ids, seq_len, mask)
            loss = F.cross_entropy(outputs, label_gpu)
            loss_total = loss_total + loss
            label = label.numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(dev_iter), report, confusion
    model.train()

    return acc, loss_total / len(dev_iter)
