import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import TensorDataset

ds = load_from_disk("../../data/aclImdb/")
train_raw = ds['train']
test_raw = ds['test']
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
    return {
        'input_ids': example['input_ids'],
        'label': example['label'],
    }
train_processed = train_raw.map(text_to_bytes, remove_columns=train_raw.column_names)
test_processed = test_raw.map(text_to_bytes, remove_columns=test_raw.column_names)

train_x = torch.tensor(train_processed['input_ids'], dtype=torch.int)  # (25000, 4000)
train_y = torch.tensor(train_processed['label'],     dtype=torch.long)

test_x  = torch.tensor(test_processed['input_ids'],  dtype=torch.int)
test_y  = torch.tensor(test_processed['label'],      dtype=torch.long)

train_ds = TensorDataset(train_x, train_y)
test_ds  = TensorDataset(test_x, test_y)

torch.save({
    'train': train_ds,
    'test':  test_ds
}, "../../data/aclImdb/data.pt")



