import torch
from typing import List, Dict
from collections import Counter
import re
import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk

ds = load_from_disk("../../data/aclImdb/")
train_dataset = ds['train']
test_dataset = ds['test']

'''
编码，字节级处理
填充
'''
def text_to_bytes(example):
    byte_values = list(example['text'].encode('utf-8'))
    if len(byte_values) >4000:
        example['input_ids'] = byte_values[:4000]
    else:
        example['input_ids'] = byte_values + [0] * (4000 - len(byte_values))
    return example
train_dataset = train_dataset.map(text_to_bytes)
test_dataset = test_dataset.map(text_to_bytes)
torch.save({'train':train_dataset,
            'test':test_dataset,
            }, "../../data/aclImdb/data.pt")




