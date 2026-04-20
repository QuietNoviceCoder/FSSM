import torch
import torch.nn as nn
import SSM_function as sf  # 确保这两个文件在路径下
import fssm


class Model(nn.Module):
    def __init__(self, batch_size, len, emb_dim, hidden_size, step, activation):
        super(Model, self).__init__()
        self.pad_id = 0
        self.embedding = nn.Embedding(256, embedding_dim=emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer1 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer2 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer3 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer4 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        # self.layer5 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        # self.layer6 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer5 = fssm.FS4Ddeq_model(hidden_size, activation, emb_dim, layers=2,
                                         final_act='gelu', skip=True, norm='LN', dropout=0.01,
                                         state_size=[batch_size, len, emb_dim],
                                         feed_size=[batch_size, len, emb_dim],
                                         feed_act='gelu')
        self.fc = nn.Linear(emb_dim, 2)

    def forward(self, x):
        # 测量参数量其实不需要 forward，但保留结构完整性
        pass


# 2. 测量函数
def measure_params():
    # 配置与你训练时一致的参数
    config = {
        "batch_size": 16,
        "len": 4000,
        "emb_dim": 128,
        "hidden_size": 64,
        "step": 0.001,
        "activation": 'tanh'
    }

    # 实例化模型（不需要 .to(cuda)，CPU 即可统计）
    model = Model(**config)

    # 统计逻辑
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-" * 40)
    print(f"{'Component':<20} | {'Params (M)':<10}")
    print("-" * 40)

    # 逐层查看，方便在论文中写各模块占比
    for name, child in model.named_children():
        p_num = sum(p.numel() for p in child.parameters()) / 1e6
        print(f"{name:<20} | {p_num:>10.4f} M")

    print("-" * 40)
    print(f"Total Parameters: {total_params / 1e6:.4f} M")
    print(f"Trainable Params: {trainable_params / 1e6:.4f} M")
    print("-" * 40)


if __name__ == "__main__":
    measure_params()