from pytorch_pretrained_bert import BertModel, BertTokenizer

if __name__ == '__main__':

    BertModel.from_pretrained('bert-base-chinese')
    BertTokenizer.from_pretrained('bert-base-chinese')
