import torch
from torch.utils.data import TensorDataset
import pandas as pd

train_path = "../../data/ann/new_aan_pairs.train.tsv"
test_path = "../../data/ann/new_aan_pairs.test.tsv"
eval_path = "../../data/ann/new_aan_pairs.eval.tsv"

def get_data(path):
    df = pd.read_csv(path, sep="\t", header=None)
    labels = df[0]
    text1 = df[3]
    text2 = df[4]
    return labels, text1, text2

train_labels, train_text1, train_text2 = get_data(train_path)
test_labels, test_text1, test_text2 = get_data(test_path)
eval_labels, eval_text1, eval_text2 = get_data(eval_path)
'''
字节级编码
砍到4000长度，
两个文档拼接
'''
def text_to_bytes(example):
    byte_values = list(example.encode('utf-8'))
    byte_values = byte_values[4:]
    if len(byte_values) >4000:
        example = byte_values[:4000]
    else:
        example = byte_values + [0] * (4000 - len(byte_values))
    return example

train_text1 = train_text1.map(text_to_bytes)
train_text2 = train_text2.map(text_to_bytes)
test_text1 = test_text1.map(text_to_bytes)
test_text2 = test_text2.map(text_to_bytes)
eval_text1 = eval_text1.map(text_to_bytes)
eval_text2 = eval_text2.map(text_to_bytes)


train_dataset = TensorDataset(torch.Tensor(train_text1), torch.Tensor(train_text2), torch.Tensor(train_labels))
test_dataset = TensorDataset(torch.Tensor(test_text1), torch.Tensor(test_text2), torch.Tensor(test_labels))
eval_dataset = TensorDataset(torch.Tensor(eval_text1), torch.Tensor(eval_text2), torch.Tensor(eval_labels))
torch.save({'train':train_dataset,
            'test':test_dataset,
            'eval':eval_dataset,
            },"../../data/ann/data.pt")


