import torch
import torch.nn as nn
import os
from pytorch_pretrained_bert import BertModel


class Config(object):
    """
    Configuration
    """

    def __init__(self, args):

        self.dataset = args.data_path + '/' + args.dataset + '/'
        if not os.path.exists(self.dataset):
            os.makedirs(self.dataset)

        self.model_name = 'bert'

        self.txts = ['train.txt', 'test.txt', 'dev.txt', 'class.txt']
        self.train_path, self.test_path, self.dev_path, self.class_path = [self.dataset + txt for txt in self.txts]

        self.datasetpkl = self.dataset + 'dataset.pkl'
        self.class_list = None
        self.job_name = args.job_name

        self.checkpoint_path = args.checkpoint_path + '/' + self.model_name
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.checkpoint_path = self.checkpoint_path + '/' + self.job_name + '.ckpt'

        self.save_path = args.model_path + '/' + self.job_name + '.ckpt'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = None
        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.pad_size = 32
        self.learning_rate = args.learning_rate
        self.tokenizer = None
        self.hidden_size = 768
        self.fine_tune = True
        self.weight_decay = args.weight_decay
        self.warmup = args.warmup
        self.continue_training = False
        self.checkpoint = None
        self.start_epoch = 0
        self.start_loss = float('inf')


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        for param in self.bert.parameters():
            param.requires_grad = config.fine_tune

        if not config.class_list:
            config.class_list = [x.strip() for x in open(config.dataset + '/class.txt').readlines()]
            config.num_classes = len(config.class_list)

        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, token_ids, seq_len, mask):

        _, pooled = self.bert(input_ids=token_ids, attention_mask=mask)
        out = self.fc(pooled)

        return out
