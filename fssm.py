import torch
import torch.nn as nn
import SSM_function as sf
from flashfft.flashfftconv import FlashFFTConv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torchdeq import get_deq, apply_norm, reset_norm
from torchdeq.solver import anderson,broyden
from torchdeq.loss.jacobian import jac_reg

def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
#定义一个层间反馈层：
#结构为，反馈层，中间层，输出层
#输出层的输出通过反馈进入反馈层，这样可以保证中间层可以任意调整

#定义输入层，中间层，反馈层

class middle_fssm(nn.Module):
    def __init__(self,hidden_size, step, activation,len,channels,use_flashfft=False):
        super().__init__()
        D_tensor = torch.tensor([0]).float()
        self.D = nn.Parameter(D_tensor, requires_grad=True)
        A,B,C,P,Q,diag =  sf.get_LegS(hidden_size,channels,DPLR=True)
        Ab,_,Cb = sf.discreatize(A, B, C, step, Discrete_method="B_trans")
        B = torch.from_numpy(B)
        C = torch.from_numpy(Cb)
        P = torch.from_numpy(P)
        Q = torch.from_numpy(Q)
        diag = torch.from_numpy(diag)
        step = torch.tensor(step)
        len = sf.return_L(len)
        len = torch.tensor(len)
        if use_flashfft:
            self.flashfftconv = FlashFFTConv(2*len, dtype=torch.float16)
        self.B = nn.Parameter(B, requires_grad=True)
        self.C = nn.Parameter(C, requires_grad=True)
        self.P = nn.Parameter(P, requires_grad=True)
        self.Q = nn.Parameter(Q, requires_grad=True)
        self.diag = nn.Parameter(diag, requires_grad=True)
        self.step = nn.Parameter(step, requires_grad=True)
        self.activation = Activation(activation)
        self.use_flashfft = use_flashfft

        #缓存机制
        self._K_c_cache = {}
        self._param_hash_cache = None
    def _compute_param_hash(self):
        """计算几个影响K_c的参数的哈希值"""
        param_tensors = [self.B.real, self.B.imag,
                         self.C.real, self.C.imag,
                         self.P.real, self.P.imag,
                         self.Q.real, self.Q.imag,
                         self.diag.real, self.diag.imag,]
        #拼在一起作为K_c的哈希值
        hash_input = torch.cat([p.flatten() for p in param_tensors])
        hash_value = torch.sum(
            hash_input * torch.arange(len(hash_input), device=hash_input.device, dtype=hash_input.dtype))
        return hash_value.item()

    def clear_cache(self):
        """手动清除缓存"""
        self._K_c_cache.clear()

    def _get_K_c_cached(self,seq_len):
        """使用缓存得到K_c"""
        current_hash = self._compute_param_hash()
        cache_key = current_hash
        #检查参数有无变化
        if cache_key in self._K_c_cache:
            return self._K_c_cache[cache_key]
        # 计算新K_c然后缓存
        K_c = sf.torch_get_K_derta(self.B, self.C, self.P, self.Q, self.diag, self.step, seq_len,
                                   DPLR=True)
        # 清理旧缓存and更新
        self._K_c_cache.clear()
        self._K_c_cache[cache_key] = K_c
        self._param_hash_cache = current_hash

        return K_c

    def forward(self,r):
        self.step.data.clamp_(min=1e-6)
        K_c = self._get_K_c_cached(r.shape[1])
        if self.use_flashfft:
            u = r.permute(0, 2, 1).half().contiguous()
            K = K_c.contiguous()
            h1 = self.flashfftconv(u, K).float().permute(0, 2, 1)
            y1 = (h1 + self.D * r)
        else:
            h1 = sf.torch_convolution(r, K_c, fft=True)
            y1 = (h1 + self.D * r)
        y = self.activation(y1)
        return y

class FSSM_Block(nn.Module):
    def __init__(
            self,
            hidden_size,
            step,
            mult_activation,
            len,
            channels,
            model='input',
            h = 0.1,
            final_act='gelu',
            skip = False,
            dropout=0.0,
            norm = False,
            feed_model='linear',
            input_size=None,
            feed_size=None,
            feed_act=None,
            use_flashfft=False,
            ):
        super().__init__()
        if model == 'input': self.fssm = middle_fssm(hidden_size, step, mult_activation,len,channels,use_flashfft=use_flashfft)
        if model == 'middle':self.fssm = middle_fssm(hidden_size, step, mult_activation,len,channels,use_flashfft=use_flashfft)
        if model == 'output':
            self.fssm = middle_fssm(hidden_size, step, mult_activation,len,channels,use_flashfft=use_flashfft)
            if feed_model == 'attention':
                self.attn = nn.MultiheadAttention(embed_dim=channels,num_heads=1,batch_first=True)
            else:
                self.fc1 = nn.Linear(input_size[1], feed_size[1])
                self.fc2 = nn.Linear(feed_size[2], feed_size[2])
                self.feedact = Activation(feed_act)
        self.model = model
        self.feed_model = feed_model
        self.final_act = Activation(final_act)
        self.fc = nn.Linear(channels,channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = skip
        self.normlization = norm
        """
        listops任务中，H＝0.6，
        text任务中，H=0.05
        """
        self.H = nn.Parameter(torch.tensor(h,dtype=torch.float32), requires_grad=True)
        if norm == 'BN':self.norm = nn.BatchNorm1d(channels)
        if norm == 'LN':self.norm = nn.LayerNorm(channels)
    def forward(self,x,feedback=None,r=None):
        if self.model == 'input':
            u = x-feedback
            y1 = self.fssm(u)
            # y1 = self.fssm(x)
        else:
            y1 = self.fssm(x)
        y2 = self.fc(y1)
        y2 = self.final_act(y2)
        if self.skip: y2 = y2 + x
        if self.normlization == 'BN' :y2 = self.norm(y2.transpose(1, 2)).transpose(1, 2)
        if self.normlization == 'LN' :y2 = self.norm(y2)
        y = self.dropout(y2)
        if self.model == 'output':
            if self.feed_model == 'attention':
                feed,_ = self.attn(query = r.permute(1,0,2), key = y.permute(1,0,2), value = y.permute(1,0,2))
            else:
                feed = self.fc2(y)
                feed = self.fc1(feed.permute(0,2,1)).permute(0,2,1)
                feed = self.feedact(feed)
            h = torch.linalg.norm(feed, ord=float('inf'), dim=1) / (torch.linalg.norm(x, ord=float('inf'), dim=1)+1e-6)
            return y, feed,h
        elif self.model == 'middle':
            h = torch.linalg.norm(y, ord=float('inf'), dim=1) / (torch.linalg.norm(x, ord=float('inf'), dim=1)+1e-6)
            return y, h
        elif self.model == 'input':
            h = torch.linalg.norm(y, ord=float('inf'), dim=1) / (torch.linalg.norm(u, ord=float('inf'), dim=1)+1e-6)
            return y, h

class FSSM_model(nn.Module):
    def __init__(
            self,
            hidden_size,
            step,
            mult_activation,
            len,
            channels,
            mid_layers=0,
            h = 0.1,
            final_act='gelu',
            skip = False,
            dropout=0.0,
            norm = False,
            input_size=None,
            feed_model='linear',
            feed_size=None,
            feed_act=None,
            use_flash = False,
            gamma = 0.0,
            ):
        super().__init__()
        self.input = FSSM_Block(hidden_size,step,mult_activation,len,channels,'input',h,final_act,skip,dropout,norm,
                                input_size=None,use_flashfft=use_flash)
        self.mid = nn.ModuleList()
        if mid_layers > 0:
            for i in range(mid_layers):self.mid.append(
                FSSM_Block(hidden_size, step, mult_activation, len, channels, 'middle', h,final_act, skip, dropout,
                           norm,use_flashfft=use_flash)
            )
        else:self.mid.append(nn.Identity())
        self.midlayers = mid_layers
        self.output = FSSM_Block(hidden_size,step,mult_activation,len,channels,'output',h,final_act,skip,dropout,norm,
                                feed_model,input_size, feed_size, feed_act,use_flashfft=use_flash)
        self.feed_model = feed_model

        #加一个门控，控制反馈强度
        self.gamma = nn.Parameter(torch.tensor(1.0)*gamma, requires_grad=True)
    def forward(self,x):
        self.clear_all_caches()
        feed = torch.zeros_like(x)
        final_feed = None
        history_feed = []
        history_feedback = []
        #先不带梯度进行循环计算
        with torch.no_grad():
            for i in range(20):
                y1, h1 = self.input(x, self.gamma*feed)
                h2 = torch.ones_like(h1)
                y2 = y1
                if self.midlayers > 0:
                    for layer in self.mid:
                        y2,h2_ = layer(y2)
                        h2 = h2 * h2_
                if self.feed_model == 'attention':
                    y3,feedback,h3 = self.output(y2,r=x)
                else:
                    y3,feedback,h3 = self.output(y2)
                e = torch.abs((feedback - feed) / (torch.abs(feed) + 1e-8) * 100).detach()
                idx = torch.randint(0, e.numel(), (50_000,), device=e.device)
                sample = e.flatten()[idx]
                threshold = torch.quantile(sample, 0.95)
                mean_95_percent = torch.mean(e[e < threshold])
                norm_e = torch.mean(torch.norm(feedback - feed, p=2, dim=1))
                if mean_95_percent < 5 or norm_e < 0.1:
                    final_feed = feed
                    # print('tiao_i=', i)
                    break
                else:
                    try:
                        feed, history_feed, history_feedback= anderson_update(
                            feed, feedback, history_feed, history_feedback, m=5, lam=1e-6
                        )
                    except:
                        feed = (feed + 0.7 * (feedback - feed)).detach()
                    # history_e.append(norm_e[0, 0].detach().cpu().numpy())
                    # plt.figure()
                    # plt.plot(history_e, color='red', label='norm_e')
                    # plt.savefig('history_e.png')
                    if i==14:
                        final_feed = feedback
                        print('迭代错误')
        if final_feed is not None:
            self.clear_all_caches()
            final_feed = final_feed.requires_grad_(True)
            #再迭代两次，使输出y可以影响到反馈网络的梯度，从而训练反馈网络
            for i in range(2):
                y1, h1 = self.input(x, self.gamma*final_feed)
                loss1 = loss_h(h1)
                loss2 = torch.zeros_like(loss1)
                h2_ = torch.ones_like(h1)
                y2 = y1
                if self.midlayers > 0:
                    for layer in self.mid:
                        y2, h2 = layer(y2)
                        h2_ = h2 * h2_
                        loss2 += loss_h(h2)
                if self.feed_model == 'attention':
                    y3, feedback, h3 = self.output(y2, r=x)
                else:
                    y3, feedback, h3 = self.output(y2)
                final_feed = feedback
        h = h1*h3*h2_
        mean_h = torch.mean(h)
        loss = loss1 + loss2 + loss_h(h3)
        return  y3,loss,h

    def clear_all_caches(self):
        """清除所有块的缓存"""
        if hasattr(self.input, 'fssm'):
            self.input.fssm.clear_cache()
        for layer in self.mid:
            if hasattr(layer, 'fssm'):
                layer.fssm.clear_cache()
        if hasattr(self.output, 'fssm'):
            self.output.fssm.clear_cache()

def loss_h(h,Target=0.1):
    mask = (h > Target).float()
    over = h * mask
    # mean pooling
    sum = over.sum(dim=1)
    len = mask.sum(dim=1)
    mean = sum / (len + 1e-8)
    return mean.mean()

#安德森加速
def anderson_update(feed, feedback, history_x, history_f, m=5, lam=1e-2):
    """
    参数:
        feed: 当前输入 (B, L, C)
        feedback: 当前输出 f(feed) (B, L, C)
        history_x: 历史输入 list，每个元素 shape = (B, L, C)
        history_f: 历史输出 list，每个元素 shape = (B, L, C)
        m: Anderson 窗口大小
        lam: 正则项，保证数值稳定
    返回:
        feed_new: 更新后的 feed (B, L, C)
    """
    B, L, C = feed.shape
    # 保存历史
    history_x.append(feed.detach())
    history_f.append(feedback.detach())
    if len(history_x) > m:
        history_x.pop(0)
        history_f.pop(0)

    k = len(history_x)
    if k < 2:
        # 历史不足，退化为普通更新
        return (feed + 0.7 * (feedback - feed)).detach(),history_x,history_f
    F = []
    G = []
    for i in range(k):
        residual = history_f[i] - history_x[i]  #(B,L,C)
        F.append(residual.reshape(B,-1))   #展开为（B,LC）
    for i in range(1,k):
        G.append(F[i]-F[i-1])
    G_matrix = torch.stack(G,dim=-1)
    # 求解最小二乘问题: min ||G @ alpha + F[-1]||^2 + lam * ||alpha||^2
    # 对每个batch单独求解
    alphas = []
    for b in range(B):
        G_b = G_matrix[b]  # (L*C, k-1)
        F_b = F[-1][b].unsqueeze(-1)  # (L*C, 1)

        # 构建正规方程: (G^T G + lam * I) alpha = -G^T F
        GTG = torch.matmul(G_b.T, G_b)  # (k-1, k-1)
        GTF = torch.matmul(G_b.T, F_b)  # (k-1, 1)

        # 添加正则项
        reg_matrix = GTG + lam * torch.eye(GTG.shape[0], device=GTG.device)

        try:
            # 求解线性系统
            alpha_b = torch.linalg.solve(reg_matrix, -GTF).squeeze(-1)  # (k-1,)
        except:
            # 数值不稳定时使用伪逆
            alpha_b = torch.linalg.pinv(reg_matrix) @ (-GTF).squeeze(-1)  # (k-1,)

        alphas.append(alpha_b)

    alphas = torch.stack(alphas, dim=0)  # (B, k-1)

    # 计算 Anderson 更新
    # x_new = (1 - sum(alpha)) * x_k + sum(alpha_i * x_i) +
    #         (1 - sum(alpha)) * (f_k - x_k) + sum(alpha_i * (f_i - x_i))

    alpha_sum = alphas.sum(dim=-1, keepdim=True)  # (B, 1)

    # 计算加权的 x 和 f
    weighted_x = torch.zeros_like(feed)  # (B, L, C)
    weighted_f = torch.zeros_like(feed)  # (B, L, C)

    for i in range(k - 1):  # 前 k-1 个历史点
        alpha_i = alphas[:, i:i + 1, None]  # (B, 1, 1)
        weighted_x += alpha_i * history_x[i]
        weighted_f += alpha_i * history_f[i]

    # 当前点的权重
    current_weight = (1 - alpha_sum).unsqueeze(-1)  # (B, 1, 1)
    weighted_x += current_weight * feed
    weighted_f += current_weight * feedback

    # Anderson 更新
    feed_new = weighted_x + (weighted_f - weighted_x)

    return feed_new.detach(), history_x, history_f


class SSMRTF_model(nn.Module):
    def __init__(self,hidden_size,channels,activation):
        super().__init__()
        A, B = sf.get_RTF(hidden_size,channels)
        A_tensor = torch.from_numpy(A).float()
        B_tensor = torch.from_numpy(B).float()
        self.A = nn.Parameter(A_tensor,requires_grad=True)
        self.B = nn.Parameter(B_tensor,requires_grad=True)
        self.h0 = nn.Parameter(torch.tensor([0]).float(),requires_grad=True)
        self.activation = Activation(activation)
        #缓存机制
        self._K_c_cache = {}
        self._param_hash_cache = None
    def _compute_param_hash(self):
        """计算几个影响K_c的参数的哈希值"""
        param_tensors = [self.A, self.B, self.h0]
        #拼在一起作为K_c的哈希值
        hash_input = torch.cat([p.flatten() for p in param_tensors])
        hash_value = torch.sum(
            hash_input * torch.arange(len(hash_input), device=hash_input.device, dtype=hash_input.dtype))
        return hash_value.item()

    def clear_cache(self):
        """手动清除缓存"""
        self._K_c_cache.clear()

    def _get_K_c_cached(self,seq_len):
        """使用缓存得到K_c"""
        current_hash = self._compute_param_hash()
        cache_key = current_hash
        #检查参数有无变化
        if cache_key in self._K_c_cache:
            return self._K_c_cache[cache_key]
        # 计算新K_c然后缓存
        K_c = sf.torch_get_RTF(self.A, self.B,seq_len)
        # 清理旧缓存and更新
        self._K_c_cache.clear()
        self._K_c_cache[cache_key] = K_c
        self._param_hash_cache = current_hash

        return K_c

    def forward(self,x):
        K_c = self._get_K_c_cached(x.shape[1])
        h1 = sf.torch_convolution(x, K_c, fft=True)
        y1 = h1 + self.h0 * x
        return self.activation(y1)

class RTFFSSM_Block(nn.Module):
    def __init__(
            self,
            hidden_size,
            mult_activation,
            channels,
            model='input',
            h = 0.1,
            final_act='gelu',
            skip = False,
            dropout=0.0,
            norm = False,
            feed_model='linear',
            input_size=None,
            feed_size=None,
            feed_act=None,
            ):
        super().__init__()
        self.encode = nn.Linear(channels,channels)
        if model == 'input': self.fssm = SSMRTF_model(hidden_size,channels,mult_activation)
        if model == 'middle':self.fssm = SSMRTF_model(hidden_size,channels,mult_activation)
        if model == 'output':
            self.fssm = SSMRTF_model(hidden_size,channels,mult_activation)
            if feed_model == 'attention':
                self.attn = nn.MultiheadAttention(embed_dim=channels,num_heads=1,batch_first=True)
            else:
                self.fc1 = nn.Linear(input_size[1], feed_size[1])
                self.fc2 = nn.Linear(feed_size[2], feed_size[2])
                self.feedact = Activation(feed_act)
        self.model = model
        self.feed_model = feed_model
        self.final_act = Activation(final_act)
        self.fc = nn.Linear(channels,channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = skip
        self.normlization = norm
        self.decode = nn.Linear(channels, channels)
        if norm == 'BN':self.norm = nn.BatchNorm1d(channels)
        if norm == 'LN':self.norm = nn.LayerNorm(channels)
    def forward(self,x,feedback=None,r=None):
        x = self.encode(x)
        if self.model == 'input':
            u = x-feedback
            y1 = self.fssm(u)
        else:
            y1 = self.fssm(x)
        y2 = self.fc(y1)
        y2 = self.final_act(y2)
        if self.skip: y2 = y2 + x
        if self.normlization == 'BN' :y2 = self.norm(y2.transpose(1, 2)).transpose(1, 2)
        if self.normlization == 'LN' :y2 = self.norm(y2)
        y = self.dropout(y2)
        y = self.decode(y)
        if self.model == 'output':
            if self.feed_model == 'attention':
                feed,_ = self.attn(query = r.permute(1,0,2), key = y.permute(1,0,2), value = y.permute(1,0,2))
            else:
                feed = self.fc2(y)
                feed = self.fc1(feed.permute(0,2,1)).permute(0,2,1)
                feed = self.feedact(feed)
            h = torch.linalg.norm(feed, ord=float('inf'), dim=1) / (torch.linalg.norm(x, ord=float('inf'), dim=1)+1e-6)
            return y, feed,h
        elif self.model == 'middle':
            h = torch.linalg.norm(y, ord=float('inf'), dim=1) / (torch.linalg.norm(x, ord=float('inf'), dim=1)+1e-6)
            return y, h
        elif self.model == 'input':
            h = torch.linalg.norm(y, ord=float('inf'), dim=1) / (torch.linalg.norm(u, ord=float('inf'), dim=1)+1e-6)
            return y, h

class RTFFSSM_model(nn.Module):
    def __init__(
            self,
            hidden_size,
            mult_activation,
            channels,
            mid_layers=0,
            h = 0.1,
            final_act='gelu',
            skip = False,
            dropout=0.0,
            norm = False,
            input_size=None,
            feed_model='linear',
            feed_size=None,
            feed_act=None,
            gamma = 0.0,
            ):
        super().__init__()
        self.input = RTFFSSM_Block(hidden_size,mult_activation,channels,'input',h,final_act,skip,dropout,norm)
        self.mid = nn.ModuleList()
        if mid_layers > 0:
            for i in range(mid_layers):self.mid.append(
                RTFFSSM_Block(hidden_size,mult_activation,channels,'middle',h,final_act,skip,dropout,norm)
            )
        else:self.mid.append(nn.Identity())
        self.midlayers = mid_layers
        self.output = RTFFSSM_Block(hidden_size,mult_activation,channels,'output',h,final_act,skip,dropout,norm,
                                    feed_model,input_size,feed_size,feed_act)
        self.feed_model = feed_model

        #加一个门控，控制反馈强度
        self.gamma = nn.Parameter(torch.tensor(1.0)*gamma, requires_grad=True)
    def forward(self,x):
        self.clear_all_caches()
        feed = torch.zeros_like(x)
        final_feed = None
        history_feed = []
        history_feedback = []
        #先不带梯度进行循环计算
        with torch.no_grad():
            for i in range(20):
                y1, h1 = self.input(x, self.gamma*feed)
                h2 = torch.ones_like(h1)
                y2 = y1
                if self.midlayers > 0:
                    for layer in self.mid:
                        y2,h2_ = layer(y2)
                        h2 = h2 * h2_
                if self.feed_model == 'attention':
                    y3,feedback,h3 = self.output(y2,r=x)
                else:
                    y3,feedback,h3 = self.output(y2)
                e = torch.abs((feedback - feed) / (torch.abs(feed) + 1e-8) * 100).detach()
                idx = torch.randint(0, e.numel(), (50_000,), device=e.device)
                sample = e.flatten()[idx]
                threshold = torch.quantile(sample, 0.95)
                mean_95_percent = torch.mean(e[e < threshold])
                norm_e = torch.mean(torch.norm(feedback - feed, p=2, dim=1))
                if mean_95_percent < 5 or norm_e < 0.1:
                    final_feed = feed
                    # print('tiao_i=', i)
                    break
                else:
                    try:
                        feed, history_feed, history_feedback= anderson_update(
                            feed, feedback, history_feed, history_feedback, m=5, lam=1e-6
                        )
                    except:
                        feed = (feed + 0.7 * (feedback - feed)).detach()
                    # history_e.append(norm_e[0, 0].detach().cpu().numpy())
                    # plt.figure()
                    # plt.plot(history_e, color='red', label='norm_e')
                    # plt.savefig('history_e.png')
                    if i==14:
                        final_feed = feedback
                        print('迭代错误')
        if final_feed is not None:
            self.clear_all_caches()
            final_feed = final_feed.requires_grad_(True)
            #再迭代两次，使输出y可以影响到反馈网络的梯度，从而训练反馈网络
            for i in range(2):
                y1, h1 = self.input(x, self.gamma*final_feed)
                h2_ = torch.ones_like(h1)
                y2 = y1
                if self.midlayers > 0:
                    for layer in self.mid:
                        y2, h2 = layer(y2)
                        h2_ = h2 * h2_
                if self.feed_model == 'attention':
                    y3, feedback, h3 = self.output(y2, r=x)
                else:
                    y3, feedback, h3 = self.output(y2)
                final_feed = feedback
        h = h1*h3*h2_
        loss = loss_h(h)
        return  y3,loss,h

    def clear_all_caches(self):
        """清除所有块的缓存"""
        if hasattr(self.input, 'fssm'):
            self.input.fssm.clear_cache()
        for layer in self.mid:
            if hasattr(layer, 'fssm'):
                layer.fssm.clear_cache()
        if hasattr(self.output, 'fssm'):
            self.output.fssm.clear_cache()


class S4D_model(nn.Module):
    def __init__(self,hidden_size,channels,activation):
        super().__init__()

        self.kernel = sf.S4DKernel(hidden_size,channels)
        self.D = nn.Parameter(torch.tensor(0.0),requires_grad=True)
        self.activation = Activation(activation)
        # 缓存机制
        self._K_cache = {}
        self._param_hash_cache = None

    def _compute_param_hash(self):
        """计算几个影响K_c的参数的哈希值"""
        param_tensors = [self.kernel.log_dt,
                         self.kernel.A_real, self.kernel.A_imag]
        # 拼在一起作为K_c的哈希值
        hash_input = torch.cat([p.flatten() for p in param_tensors])
        hash_value = torch.sum(
            hash_input * torch.arange(len(hash_input), device=hash_input.device, dtype=hash_input.dtype))
        return hash_value.item()

    def clear_cache(self):
        """手动清除缓存"""
        self._K_cache.clear()

    def _get_K_cached(self, seq_len,get_state):
        # if self.training:
        #     return self.kernel(seq_len)
        """使用缓存得到K_c"""
        current_hash = self._compute_param_hash()
        cache_key = current_hash
        # 检查参数有无变化
        if cache_key in self._K_cache:
            return self._K_cache[cache_key]
        # 计算新K_c然后缓存
        K_c = self.kernel(seq_len,get_state)
        # 清理旧缓存and更新
        self._K_cache.clear()
        self._K_cache[cache_key] = K_c
        self._param_hash_cache = current_hash

        return K_c

    def forward(self,u,get_state=False):
        if get_state:
            K, G_k = self.kernel(u.shape[1], get_state)  # (C, N, L)
            y = sf.torch_convolution(u, K, fft=True)
            feed = sf.torch_convolution(u, G_k, fft=True)
            y = y + self.D * u
            return self.activation(y), feed
        else:
            K = self._get_K_cached(u.shape[1],get_state)#(C, N, L)
            y = sf.torch_convolution(u, K, fft=True)
            y = y + self.D * u
            return self.activation(y)

class S4D_Block(nn.Module):
    def __init__(
            self,
            hidden_size,
            mult_activation,
            channels,
            model='input',
            final_act='gelu',
            skip = False,
            dropout=0.0,
            norm = False,
            glu = True,
            ):
        super().__init__()
        if model == 'input': self.fssm = S4D_model(hidden_size,channels,mult_activation)
        if model == 'middle':self.fssm = S4D_model(hidden_size,channels,mult_activation)
        if model == 'output':self.fssm = S4D_model(hidden_size,channels,mult_activation)
        self.model = model
        self.pre = nn.Linear(channels,channels)
        self.final_act = Activation(final_act)
        self.dropout = nn.Dropout(dropout)
        self.skip = skip
        self.normlization = norm
        if norm == 'BN':self.norm = nn.BatchNorm1d(channels)
        if norm == 'LN':self.norm = nn.LayerNorm(channels)
        self.glu_use = glu
        self.glu_proj = nn.Linear(channels, 2 * channels)

    def glu(self,x):
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)

    def forward(self,x,feedback=None,r=None):
        res = x
        if self.normlization == 'BN': x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        if self.normlization == 'LN': x = self.norm(x)
        if self.model == 'input':
            u = x - feedback
            y = self.fssm(u)
        elif self.model == 'middle':
            y = self.fssm(x)
        elif self.model == 'output':
            y, feed = self.fssm(x,get_state=True)
        y = self.dropout(y)
        if self.skip: y = y + x

        res = y
        if self.normlization == 'BN' :y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        if self.normlization == 'LN' :y = self.norm(y)
        if self.glu_use:
            z = self.glu_proj(y)
            z = self.glu(z)
        else:
            z = self.pre(y)
        z = self.dropout(z)
        out = z + res
        if self.model == 'output':
            return out,feed
        return out

class Feed_Block(nn.Module):
    def __init__(
            self,
            model = 'linear',
            feed_activation = None,
            input_size = None,
            output_size = None,
            norm = None,
            ):
        super().__init__()
        self.model = model
        if model == 'attention':
            self.attn = nn.MultiheadAttention(embed_dim=input_size[2], num_heads=1, batch_first=True)
        else:
            self.pre = nn.Linear(input_size[2], output_size[2])
        self.feedact = Activation(feed_activation)
        if norm == 'BN': self.norm = nn.BatchNorm1d(input_size[2])
        if norm == 'LN': self.norm = nn.LayerNorm(input_size[2])
        self.normlization = norm
    def forward(self,feed,r=None):
        if self.model == 'attention':
            out, _ = self.attn(query=r.permute(1, 0, 2), key=feed.permute(1, 0, 2), value=feed.permute(1, 0, 2))
            out = feed.permute(1, 0, 2)
        else:
            out = self.pre(feed)
        out = self.feedact(out)
        if self.normlization == 'BN' : out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        if self.normlization == 'LN' : out = self.norm(out)
        return out

def plot_e(history_e):
    data = np.array(history_e.cpu())*100
    x_idx = data.shape[0]
    y_idx = data.shape[1]
    X, Y = np.meshgrid(np.arange(x_idx), np.arange(y_idx))
    Z = data.T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Channel')
    ax.set_zlabel('norm_e')
    ax.set_title('norm_e by times')
    plt.show()
    plt.savefig("history_e_visualization.png")

class DEQFunc(nn.Module):
    def __init__(self, input_block, mid_blocks, output_block, feed_block, gamma, dropout):
        super().__init__()
        self.input_block = input_block
        self.mid_blocks = mid_blocks
        self.output_block = output_block
        self.feed_block = feed_block
        self.gamma = gamma
        self.dropout = nn.Dropout(dropout)
        self.cache = {}
    def forward(self, feed, x):
        feed = self.feed_block(feed, x)
        gamma = torch.sigmoid(self.gamma)
        # gamma = self.gamma
        feed = self.dropout(feed) * gamma
        # Forward pass through the network
        y1 = self.input_block(x, feed)
        y2 = y1
        for layer in self.mid_blocks:
            if not isinstance(layer, nn.Identity):
                y2 = layer(y2)
        y3, feed = self.output_block(y2)
        self.cache['y'] = y3
        return feed

class FS4Ddeq_model(nn.Module):
    def __init__(
            self,
            hidden_size,
            mult_activation,
            channels,
            layers=0,
            final_act='gelu',
            skip = False,
            dropout=0.0,
            norm = False,
            state_size=None,
            feed_model='linear',
            feed_size=None,
            feed_act=None,
            #DEQ参数
            #可选  "broyden","anderson","fixed_point_iter"
            deq_solver = 'anderson',
            deq_f_max_iter = 30,
            deq_b_max_iter = 30,
            deq_f_tol = 5e-2,
            deq_b_tol = 1e-2,
            deq_anderson_m = 3,
            ):
        super().__init__()
        input = S4D_Block(hidden_size,mult_activation,channels,'input',final_act,skip,dropout,norm)
        mid = nn.ModuleList()
        if layers == 1:
            mid.append(nn.Identity())
            output = nn.Identity()
        elif layers == 2:
            mid.append(nn.Identity())
            output = S4D_Block(hidden_size, mult_activation, channels, 'output', final_act, skip,dropout, norm)
        elif layers > 2:
            for i in range(layers-2):mid.append(
                S4D_Block(hidden_size, mult_activation, channels, 'middle', final_act, skip,dropout, norm)
            )
            output = S4D_Block(hidden_size, mult_activation, channels, 'output', final_act, skip,dropout, norm)
        feedblock = Feed_Block(feed_model, feed_act, state_size, feed_size, norm)
        gamma = nn.Parameter(torch.tensor(-5.0),requires_grad=True)
        self.state_size = state_size
        self.deq_func = DEQFunc(input, mid, output, feedblock, gamma, dropout)
        # apply_norm(self.deq_func,filter_out='fssm')
        apply_norm(self.deq_func)
        self.deq = get_deq(
            ift = True,
            f_solver=deq_solver,
            b_solver=deq_solver,
            f_max_iter=deq_f_max_iter,
            b_max_iter=deq_b_max_iter,
            f_tol=deq_f_tol,
            b_tol=deq_b_tol,
            stop_mode = 'rel',
            f_anderson_m=deq_anderson_m if deq_solver == 'anderson' else None,
            b_anderson_m=deq_anderson_m if deq_solver == 'anderson' else None,
            store_sequence = False,
            record_f_norm=False,
        )
        self.register_buffer('z_init', None),

    def forward(self,x):
        self.clear_all_caches()
        reset_norm(self.deq_func)
        # if self.training and self.z_init is not None:
        #     y = self.z_init
        # else:y = torch.zeros_like(x)
        feed = torch.zeros_like(x)
        f = lambda z: self.deq_func(z, x)
        solve , deq_info = self.deq(f,feed)
        z_star = solve[-1]
        y = self.deq_func.cache['y']
        jac_loss = torch.tensor(0.0).to(z_star.device)
        if self.training:
            self.z_init = z_star.detach()
            # 有概率计算，并不在全部的情况下计算jac，防止过度正则化
            if torch.rand(1) < 0.8:
                new_z_star = f(z_star.requires_grad_())
                jac_loss = jac_reg(new_z_star,z_star,create_graph=True)

        norm_x = torch.norm(x, p=2, dim=1)
        norm_feed = torch.norm(z_star, p=2, dim=1)
        zhanbi = norm_feed / (norm_x + 1e-8)
        mean_zhanbi = torch.mean(zhanbi)
        mean_e = torch.mean(deq_info['rel_trace'][:,-1]*100)
        # print('mean_e',mean_e)
        nstep = deq_info['nstep']
        # print('nstep',torch.mean(nstep))
        if mean_e > 10 :
            print("迭代失败")
            print("rel_mean",mean_e)

        # h = self.deq_func.cache['L']
        # h = torch.linalg.norm(z_star, ord=2, dim=1) / (torch.linalg.norm((x-feed), ord=2, dim=1)+1e-6)
        # feed_h = torch.linalg.norm(feed, ord=2, dim=1) / (torch.linalg.norm(z_star, ord=2, dim=1)+1e-6)
        # print('h',torch.mean(h*feed_h))
        # return z_star, loss_h(h ,1.0)+loss_h(feed_h,0.3), h, feed_h
        return y, jac_loss, torch.mean(nstep), mean_zhanbi

    def clear_all_caches(self):
        """清除所有块的缓存"""
        if hasattr(self.deq_func.input_block, 'fssm'):
            self.deq_func.input_block.fssm.clear_cache()
        for layer in self.deq_func.mid_blocks:
            if hasattr(layer, 'fssm'):
                layer.fssm.clear_cache()
        if hasattr(self.deq_func.output_block, 'fssm'):
            self.deq_func.output_block.fssm.clear_cache()
