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
'''
def text_to_bytes(text, max_len=4000):
    byte_values = list(text.encode("utf-8"))
    byte_values = byte_values[:max_len]
    byte_values = byte_values + [0] * (max_len - len(byte_values))
    return byte_values

train_text1 = train_text1.map(text_to_bytes)
train_text2 = train_text2.map(text_to_bytes)
test_text1 = test_text1.map(text_to_bytes)
test_text2 = test_text2.map(text_to_bytes)
eval_text1 = eval_text1.map(text_to_bytes)
eval_text2 = eval_text2.map(text_to_bytes)

train_dataset = TensorDataset(
    torch.tensor(train_text1.tolist(), dtype=torch.long),
    torch.tensor(train_text2.tolist(), dtype=torch.long),
    torch.tensor(train_labels.tolist(), dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(test_text1.tolist(), dtype=torch.long),
    torch.tensor(test_text2.tolist(), dtype=torch.long),
    torch.tensor(test_labels.tolist(), dtype=torch.long)
)
eval_dataset = TensorDataset(
    torch.tensor(eval_text1.tolist(), dtype=torch.long),
    torch.tensor(eval_text2.tolist(), dtype=torch.long),
    torch.tensor(eval_labels.tolist(), dtype=torch.long)
)
torch.save({'train':train_dataset,
            'test':test_dataset,
            'eval':eval_dataset,
            },"../../data/ann/data.pt")
print("over")


