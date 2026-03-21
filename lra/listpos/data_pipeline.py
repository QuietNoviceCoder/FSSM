import torch
from typing import List, Dict
from collections import Counter
import re
import os
from torch.utils.data import Dataset, DataLoader


# 定义一个自定义的 Dataset 类来处理数据
class AlgorithmicDataset(Dataset):
    def __init__(self, file_path, vocab=None, stoi=None, max_length=2000):
        self.max_length = max_length
        self.data = self._read_data(file_path)
        if vocab is None:
            self.vocab ,self.stoi= self._build_vocab()
        else:
            self.vocab = vocab
            self.stoi = stoi

    def _read_data(self, file_path):
        """读取 TSV 文件并进行预处理"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过表头
            next(f)
            for line in f:
                source, target = line.strip().split('\t')
                source = self.rename_close_brackets(source)
                data.append({'source': source, 'target': int(target)})
        return data

    def rename_close_brackets(self, text):
        """
        输入的"]"变成了"X","("")"都去掉
        """
        text = re.sub(r'\]', 'X', text)
        text = re.sub(r'\(', '', text)
        text = re.sub(r'\)', '', text)
        return text

    def _build_vocab(self):
        """构建词汇表"""
        print("Building vocab...")
        #返回token列表
        class TokenDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, index):
                return self.data[index]['source'].split()
        dataset = TokenDataset(self.data)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=True)
        #统计词频
        token_counter = Counter()
        for tokens in dataloader:
            for token in tokens:
                token_counter[token] += 1
        specials = ['<pad>', '<unk>']
        vocab: List[str] = specials.copy()  # 词汇表（初始为特殊标记）
        # 按词频降序添加其他 token（过滤已存在的特殊标记）
        # 注意：Counter.most_common() 返回按词频降序的 (token, count) 列表
        for token, _ in token_counter.most_common():
            if token not in vocab:  # 避免重复添加特殊标记
                vocab.append(token)
        # ----------------------
        # 步骤4：生成 token → index 映射字典
        # ----------------------
        token2idx: Dict[str, int] = {token: idx for idx, token in enumerate(vocab)}

        print(f"Finished building vocab. Vocab size: {len(vocab)}")
        return vocab,token2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 分词
        source_tokens = item['source'].split()
        # 编码：将分词结果转换为整数序列
        source_encoded = [self.stoi[token] for token in source_tokens]

        # 截断
        if len(source_encoded) > self.max_length:
            source_encoded = source_encoded[:self.max_length]

        return {'inputs': torch.tensor(source_encoded, dtype=torch.long),
                'targets': torch.tensor(item['target'], dtype=torch.long)}
# 主流程函数
def get_datasets(task_name, data_dir, batch_size=256, max_length=2000):
    """
    PyTorch 版本的 get_datasets 函数
    :param task_name: 任务名称
    :param data_dir: 数据目录
    :param batch_size: 批次大小
    :param max_length: 最大序列长度
    :return: 训练、验证、测试 DataLoader 和词汇表对象
    """
    train_path = os.path.join(data_dir, f'{task_name}/basic_train.tsv')
    val_path = os.path.join(data_dir, f'{task_name}/basic_val.tsv')
    test_path = os.path.join(data_dir, f'{task_name}/basic_test.tsv')

    print("Loading validation dataset to build vocab...")
    # 首先创建验证集 Dataset 对象来构建词汇表
    val_dataset = AlgorithmicDataset(val_path, max_length=max_length)
    vocab = val_dataset.vocab
    stoi = val_dataset.stoi

    # 使用共享的词汇表创建其他 Dataset
    train_dataset = AlgorithmicDataset(train_path, vocab, stoi, max_length=max_length)
    test_dataset = AlgorithmicDataset(test_path, vocab, stoi, max_length=max_length)

    print("Creating DataLoaders...")

    return train_dataset, val_dataset, test_dataset, vocab, stoi

data_dir = '../../data/'
task_name = 'listops'
batch_size = 256
max_length = 2000

# 确保数据目录存在
if not os.path.exists(data_dir):
    print(f"Error: Directory '{data_dir}' not found. Please create it and place your data files inside.")
else:
    train_datasets, val_datasets, test_datasets, vocab, stoi = get_datasets(
        task_name, data_dir, batch_size, max_length
    )
    def load_lra_listops(max_len=2000):
        train_data = train_datasets.data
        val_data = val_datasets.data
        test_data = test_datasets.data

        def encode_sequence(tokens):
            token = tokens.split()
            source_encoded = [stoi[token] for token in token]
            if len(token) > max_len:
                source_encoded = source_encoded[:max_len]
            else:
                source_encoded = source_encoded + [0] * (max_len - len(token))
            return torch.tensor(source_encoded, dtype=torch.long)

        def prepare_dataset(raw_data):
            inputs = []
            labels = []
            for datas in raw_data:
                tokens = datas['source']
                label = datas['target']
                inputs.append(encode_sequence(tokens))
                labels.append(label)
            return torch.stack(inputs), torch.tensor(labels, dtype=torch.long)

        train_x, train_y = prepare_dataset(train_data)
        val_x, val_y = prepare_dataset(val_data)
        test_x, test_y = prepare_dataset(test_data)
        return train_x, train_y,val_x, val_y, test_x, test_y
    train_x, train_y, val_x, val_y, test_x, test_y = load_lra_listops()
    torch.save({'train_x':train_x,
                    'train_y':train_y,
                    'val_x':val_x,
                    'val_y':val_y,
                    'test_x':test_x,
                    'test_y':test_y,
                    'vocab':vocab,
                    'stoi':stoi
            },"../../data/listops/data.pth")
