import torch
import torch.nn as nn
import numpy as np
# date  2025.9.12
from flashfft.flashfftconv import FlashFFTConv
import warnings
import math


'''
#2025.8.27
修改了多通道SSM，不同通道使用独立的SSM，但是其他的没有改，只有SSM函数
#2025.10.10
修改了多通道RTF，并写了RTF的反馈网络
#2024.10.13
修改多通道S4D，并写S4D的反馈网络
'''
#test
warnings.filterwarnings("ignore",category=  UserWarning,message="ComplexHalf support is experimental.*")
#定义hippo矩阵和离散方法
def get_LegT(N,slide_window):
    A = np.zeros((N, N), dtype=float)
    B = np.zeros((N, 1))
    for i in range(N):
        for j in range(N):
            base = -1/slide_window*(np.sqrt((2*i+1)*(2*j+1)))
            if i > j : A[i,j] = base
            else : A[i,j] = base*((-1)**(i-j))
        B[i,0] = 1/slide_window*(np.sqrt(2*i+1))
    C = np.ones((1, N))
    return A,B,C
#求输入的对角阵和转换矩阵
def eig_matrix(input):
    value, vectors = np.linalg.eig(input)
    eig_value = np.imag(value) * 1j
    for i in range(len(value)):
        vectors[:, i] = vectors[:, i] / np.linalg.norm(vectors[:, i])
    U = vectors
    return eig_value , U
def conj_round(input):
    threshold = 1e-6
    real_part = input.real
    imag_part = input.imag
    real_part[torch.abs(real_part) < threshold] = 0
    imag_part[torch.abs(imag_part) < threshold] = 0
    return real_part+imag_part*1j
def get_LegS(N,channels,DPLR = False):
    if DPLR==False:
        A = np.zeros((N, N), dtype=float)
        B = np.zeros((N, 1))
        for i in range(N):
            for j in range(N):
                if i > j : A[i,j] = -(np.sqrt((2*i+1)*(2*j+1)))
                elif i == j: A[i,j] = -(i+1)
            B[i,0] = np.sqrt(2*i+1)
        C = np.ones((channels, N))
        B = np.tile(B, (1, channels))
        return A,B,C
    if DPLR==True:
        S = np.zeros((N, N), dtype=float)
        B = np.zeros((N, 1))
        C = np.ones((1, N))
        P = np.zeros((N, 1))
        Q = np.zeros((N, 1))
        for i in range(N):
            for j in range(N):
                if i == j : S[i,j] = 0
                elif i > j : S[i,j] = -1/2*np.sqrt((2*i+1)*(2*j+1))
                elif i < j : S[i,j] = 1/2*np.sqrt((2*i+1)*(2*j+1))
            B[i, 0] = np.sqrt(2 * i + 1)
            P[i, 0] = np.sqrt(2 * i + 1)
            Q[i, 0] = np.sqrt(2 * i + 1)
        eig_value , U = eig_matrix(S)
        eig_value = eig_value - 1/2
        diag = np.diag(eig_value)
        diag = diag - np.eye(N)*1/2
        P = U.conj().T @ P * np.sqrt(2) / 2
        Q = U.conj().T @ Q * np.sqrt(2) / 2
        B = U.conj().T @ B
        C = C @ U
        A = diag -  P @ Q.conj().T

        B = np.tile(B, (1, channels))
        C = np.tile(C, (channels, 1))
        P = np.tile(P, (1, channels))
        Q = np.tile(Q, (1, channels))

        return A,B,C,P,Q,eig_value
def get_RTF(N,channels):
    B = np.random.randn(channels, N)
    A = np.zeros((channels, N))
    return A,B

def discreatize(A,B,C,step,Discrete_method="B_trans"):
    I = np.eye(A.shape[0])
    if Discrete_method == "F_trans":
        Ab = I + step * A
        Bb = step * B
        return Ab,Bb,C
    if Discrete_method == "Back_trans":
        Ab = np.linalg.inv(I - step* A)
        Bb = step * Ab @ B
        return Ab, Bb, C
    if Discrete_method == "B_trans":
        BL = np.linalg.inv(I - (step / 2.0) * A)
        Ab = BL @ (I + (step / 2.0) * A)
        Bb = step * BL @ B
        return Ab, Bb, C

#定义RNN过程
def scan_SSM(Ab,Bb,Cb,u,x0):
    x1 = Ab @ x0 + Bb*u
    y = Cb @ x1
    return x1,y

def run_SSM(Ab,Bb,Cb,u):
    L = u.shape[0]
    N = Ab.shape[0]
    x0 = torch.zeros((N,1))
    y = torch.zeros((1,L))
    for i in range(L):
        x0,y[0,i] = scan_SSM(Ab,Bb,Cb,u[i],x0)
    return y

#定义卷积过程
#卷积核
def get_K(A,B,C,n_times):
    for i in range(n_times):
        if i > 0 :
            raw_date = A @ raw_date
            K = np.hstack([K,raw_date])
        elif i == 0:
            raw_date = B
            K = B
    return C @ K
def cauchy(QP,w,lamda):
    den = w.unsqueeze(1) - lamda.unsqueeze(0)
    division = QP.unsqueeze(0)/den.unsqueeze(1)
    out = division.sum(dim = 2).T
    return out

def torch_get_K(*args,DPLR=False):
    if DPLR == False :
        A,B,C,n_times = args
        k_ = torch.zeros(C.shape[0],n_times)
        for c in range(C.shape[0]):
            for i in range(n_times):
                if i > 0 :
                    raw_date = torch.mm(A,raw_date)
                    K = torch.cat((K,raw_date),dim=1)
                elif i == 0:
                    raw_date = B[:,c:c+1]
                    K = B[:,c:c+1]
            k_[c,:] = torch.mm(C[c:c+1,:], K)
        return k_
    if DPLR == True:
        A_L,B,C,P,Q,eig_value,derta,n_times = args
        if A_L.dtype != torch.complex128:
            A_L = torch.view_as_complex(A_L)
            B = torch.view_as_complex(B)
            C = torch.view_as_complex(C)
            P = torch.view_as_complex(P)
            Q = torch.view_as_complex(Q)
            eig_value = torch.view_as_complex(eig_value)
        I = torch.eye(A_L.shape[0]).to(A_L.device)
        z = torch.exp((torch.pi * -2j) * torch.arange(n_times) / n_times).to(A_L.device)
        w = 2 / derta * (1-z)/(1+z)
        #C波浪
        C_ = C @ (I - A_L)
        k00 = cauchy(C_ * B.T,w,eig_value)
        k01 = cauchy(C_ * P.T,w,eig_value)
        k10 = cauchy(Q.conj().T * B.T,w,eig_value)
        k11 = cauchy(Q.conj().T * P.T,w,eig_value)
        K_w = 2/(1+z)*(k00 - k01 / (1+k11) * k10)
        K = torch.fft.irfft(K_w,n_times)
        return K
def torch_get_K_derta(*args,DPLR=False):
    if DPLR == False :
        A,B,C,n_times = args
        k_ = torch.zeros(C.shape[0],n_times)
        for c in range(C.shape[0]):
            for i in range(n_times):
                if i > 0 :
                    raw_date = torch.mm(A,raw_date)
                    K = torch.cat((K,raw_date),dim=1)
                elif i == 0:
                    raw_date = B[:,c:c+1]
                    K = B[:,c:c+1]
            k_[c,:] = torch.mm(C[c:c+1,:], K)
        return k_
    if DPLR == True:
        B,C_,P,Q,eig_value,derta,n_times = args
        z = torch.exp((torch.pi * -2j) * torch.arange(n_times) / n_times).to(B.device)
        w = 2 / (derta+1e-6) * (1-z)/(1+z+1e-6)
        #C波浪
        k00 = cauchy(C_ * B.T,w,eig_value)
        k01 = cauchy(C_ * P.T,w,eig_value)
        k10 = cauchy(Q.conj().T * B.T,w,eig_value)
        k11 = cauchy(Q.conj().T * P.T,w,eig_value)
        K_w = 2/(1+z)*(k00 - k01 / (1+k11) * k10)
        K = torch.fft.irfft(K_w,n_times)
        return K

#重新定义卷积核K的获取函数
def get_K_H(*args,DPLR=False):
    if DPLR == False :
        A,B,C,n_times = args
        for i in range(n_times):
            if i > 0 :
                raw_date = torch.mm(A,raw_date)
                K = torch.cat((K,raw_date),dim=1)
            elif i == 0:
                raw_date = B
                K = B
        K = torch.mm(C,K)
        K_w = torch.fft.rfft(K,n_times)
        return K,torch.max(torch.abs(K_w))
    if DPLR == True:
        A_L,B,C,P,Q,eig_value,derta,n_times = args
        if A_L.dtype != torch.complex128:
            A_L = torch.view_as_complex(A_L)
            B = torch.view_as_complex(B)
            C = torch.view_as_complex(C)
            P = torch.view_as_complex(P)
            Q = torch.view_as_complex(Q)
            eig_value = torch.view_as_complex(eig_value)
        I = torch.eye(A_L.shape[0]).to(A_L.device)
        z = torch.exp((torch.pi * -2j) * torch.arange(n_times) / n_times).to(A_L.device)
        w = 2 / derta * (1-z)/(1+z)
        #C波浪
        C_ = C @ (I - A_L)
        k00 = cauchy(C_ * B.T,w,eig_value)
        k01 = cauchy(C_ * P.T,w,eig_value)
        k10 = cauchy(Q.conj().T * B.T,w,eig_value)
        k11 = cauchy(Q.conj().T * P.T,w,eig_value)
        K_w = 2/(1+z)*(k00 - k01 / (1+k11) * k10)
        K = torch.fft.irfft(K_w,n_times)
        return K , torch.max(torch.abs(K_w))
def torch_get_RTF(A,B,L):
    N = A.shape[1]
    num = torch.nn.functional.pad(B,(1,L-N-1))
    den = torch.nn.functional.pad(A,(1,L-N-1))
    den[:,0] = 1
    num_fft = torch.fft.rfft(num)
    den_fft = torch.fft.rfft(den)
    den_fft[den_fft.abs() < 1e-8] = 1e-8
    # num/den
    out_fft = num_fft / den_fft
    out = torch.fft.irfft(out_fft, n=L)
    return out

#定义卷积函数
def convolution(u,K,fft):
    K = K.reshape(u.shape[0])
    u = u.T.reshape(u.shape[0])
    if fft == False:
        y = np.convolve(u, K)[:u.shape[0]]
        return y
    if fft == True:
        Kd = np.fft.rfft(np.pad(K,(0,u.shape[0])))
        ud = np.fft.rfft(np.pad(u,(0,K.shape[0])))
        out = np.fft.irfft(Kd*ud,u.shape[0])
        return out


def torch_convolution(u,K,fft):
    if len(u.shape)<=2:
        u = u.reshape(1,-1)
        K = K.flatten()
        if fft == False:
            out = torch.nn.functional.conv1d(u, K)[:u.shape[1]]
            return out
        if fft == True:
            u_pad = torch.nn.functional.pad(u,(0,K.shape[0]))
            k_pad = torch.nn.functional.pad(K,(0,u.shape[1]))
            K_fft = torch.fft.rfft(k_pad)
            u_fft = torch.fft.rfft(u_pad)
            out_fft = K_fft * u_fft
            out = torch.fft.irfft(out_fft,u.shape[1]).float()
        return out
    if len(u.shape)==3:
        out = torch.zeros_like(u,device=u.device)
        if fft == False:
            out = torch.nn.functional.conv1d(u, K)[:u.shape[0]]
            return out
        if fft == True:
            u_pad = torch.nn.functional.pad(u,(0,0,0,K.shape[1],0,0))
            k_pad = torch.nn.functional.pad(K,(0,u.shape[1]))
            K_fft = torch.fft.rfft(k_pad)
            u_fft = torch.fft.rfft(u_pad.permute(0,2,1))
            out_fft = K_fft * u_fft
            out = torch.fft.irfft(out_fft,u.shape[1]+K.shape[1]).float()
        return out.permute(0,2,1)[:,:u.shape[1],:]

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
def return_L(len):
    n = len.bit_length()
    return 1<<n
#定义SSM线性层
class SSM_model(nn.Module):
    def __init__(self,*args,usd_D=True,DPLR=False):
        super().__init__()
        if usd_D == True:
            D_tensor = torch.tensor([0]).float()
            self.D = nn.Parameter(D_tensor, requires_grad=True)
        else:self.D = nn.Parameter(torch.tensor([0]).float(), requires_grad=False)
        if DPLR == False:
            hidden_size, step, activation, channels = args
            A, B, C = get_LegS(hidden_size,channels)
            A, B, C = discreatize(A, B, C, step, Discrete_method="B_trans")
            A_tensor = torch.from_numpy(A).float()
            B_tensor = torch.from_numpy(B).float()
            C_tensor = torch.from_numpy(C).float()
            self.A = nn.Parameter(A_tensor, requires_grad=False)
            self.B = nn.Parameter(B_tensor, requires_grad=True)
            self.C = nn.Parameter(C_tensor, requires_grad=True)
        if DPLR == True:
            hidden_size, step, activation,len, channels = args
            A,B,C,P,Q,diag = get_LegS(hidden_size,channels,DPLR=True)
            Ab,_,Cb = discreatize(A, B, C, step, Discrete_method="B_trans")
            A_L = np.linalg.matrix_power(Ab,len)
            A_L = torch.from_numpy(A_L)
            B = torch.from_numpy(B)
            C = torch.from_numpy(Cb)
            P = torch.from_numpy(P)
            Q = torch.from_numpy(Q)
            diag = torch.from_numpy(diag)
            step = torch.tensor(step)
            self.A_L = nn.Parameter(A_L, requires_grad=False)
            self.B = nn.Parameter(B, requires_grad=True)
            self.C = nn.Parameter(C, requires_grad=True)
            self.P = nn.Parameter(P, requires_grad=True)
            self.Q = nn.Parameter(Q, requires_grad=True)
            self.diag = nn.Parameter(diag, requires_grad=False)
            self.step = nn.Parameter(step, requires_grad=True)
            self.activation = Activation(activation)
    def forward(self,x,fft=True,DPLR=True):
        if DPLR == False:
            K_c = torch_get_K(self.A, self.B, self.C, x.shape[1])
            h1 = torch_convolution(x,K_c,fft)
            y1 = h1 + self.D * x
        if DPLR == True:
            K_c = torch_get_K(self.A_L, self.B, self.C, self.P, self.Q, self.diag,self.step, x.shape[1],
                              DPLR=True)
            h1 = torch_convolution(x, K_c, fft)
            y1 = (h1 + self.D * x)
        return self.activation(y1)
class SSM_model_derta(nn.Module):
    def __init__(self,*args,usd_D=True,DPLR=False):
        super().__init__()
        if usd_D == True:
            D_tensor = torch.tensor([0]).float()
            self.D = nn.Parameter(D_tensor, requires_grad=True)
        else:self.D = nn.Parameter(torch.tensor([0]).float(), requires_grad=False)
        if DPLR == False:
            hidden_size, step, activation, channels = args
            A, B, C = get_LegS(hidden_size,channels)
            A, B, C = discreatize(A, B, C, step, Discrete_method="B_trans")
            A_tensor = torch.from_numpy(A).float()
            B_tensor = torch.from_numpy(B).float()
            C_tensor = torch.from_numpy(C).float()
            self.A = nn.Parameter(A_tensor, requires_grad=False)
            self.B = nn.Parameter(B_tensor, requires_grad=True)
            self.C = nn.Parameter(C_tensor, requires_grad=True)
        if DPLR == True:
            hidden_size, step, activation,len, channels = args
            _,B,C,P,Q,diag = get_LegS(hidden_size,channels,DPLR=True)
            B = torch.from_numpy(B)
            C = torch.from_numpy(C)
            P = torch.from_numpy(P)
            Q = torch.from_numpy(Q)
            diag = torch.from_numpy(diag)
            step = torch.tensor(step)
            len = return_L(len)
            len = torch.tensor(len)
            self.B = nn.Parameter(B, requires_grad=True)
            self.C_ = nn.Parameter(C, requires_grad=True)
            self.P = nn.Parameter(P, requires_grad=True)
            self.Q = nn.Parameter(Q, requires_grad=True)
            self.diag = nn.Parameter(diag, requires_grad=True)
            self.step = nn.Parameter(step, requires_grad=True)
            self.activation = Activation(activation)
            self.flashfftconv = FlashFFTConv(2*len,dtype=torch.float16)
    def forward(self,x,fft=True,DPLR=True):
        if DPLR == False:
            K_c = torch_get_K(self.A, self.B, self.C, x.shape[1])
            h1 = torch_convolution(x,K_c,fft)
            y1 = h1 + self.D * x
        if DPLR == True:
            K_c = torch_get_K_derta(self.B, self.C_, self.P, self.Q, self.diag,self.step, x.shape[1],
                              DPLR=True)
            h1 = torch_convolution(x, K_c, fft)
            # 输入u的形状是（B，H,L）,K的形状是（H,L）
            # L必须是256-4,194,304之间的2的幂，若大于32768，必须是16的倍数
            # u的长度可以小于L，但是必须是2的倍数，L的大小必须是4的倍数
            # u = x.permute(0, 2, 1).half().contiguous()
            # K = K_c.contiguous()
            # h1 = self.flashfftconv(u, K).permute(0, 2, 1).float()
            y1 = (h1 + self.D * x)
        return self.activation(y1)

class SSM_Block(nn.Module):
    def __init__(
            self,
            hidden_size,
            step,
            mult_activation,
            len,
            channels,
            final_act='gelu',
            skip = False,
            dropout=0.0,
            norm = False,
            DPLR=True,
            ):
        super().__init__()
        # self.ssm = SSM_model(hidden_size,step,mult_activation,len,channels,DPLR=DPLR)
        self.ssm = SSM_model_derta(hidden_size, step, mult_activation, len, channels, DPLR=DPLR)
        self.final_act = Activation(final_act)
        self.fc = nn.Linear(channels,channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = skip
        self.normlization = norm
        self.encode = nn.Linear(channels,channels)
        self.decode = nn.Linear(channels,channels)
        if norm == 'BN':self.norm = nn.BatchNorm1d(channels)
        if norm == 'LN':self.norm = nn.LayerNorm(channels)
    def forward(self,x):
        x = self.encode(x)
        y1 = self.ssm(x)
        y2 = self.fc(y1)
        y2 = self.final_act(y2)
        if self.skip: y2 = y2 + x
        if self.normlization == 'BN' :y2 = self.norm(y2.transpose(1, 2)).transpose(1, 2)
        if self.normlization == 'LN' :y2 = self.norm(y2)
        y2 = self.decode(y2)
        y2 = self.dropout(y2)
        return y2

class SSMRTF_model(nn.Module):
    def __init__(self,hidden_size,channels,activation):
        super().__init__()
        A, B = get_RTF(hidden_size,channels)
        A_tensor = torch.from_numpy(A).float()
        B_tensor = torch.from_numpy(B).float()
        self.A = nn.Parameter(A_tensor,requires_grad=True)
        self.B = nn.Parameter(B_tensor,requires_grad=True)
        self.h0 = nn.Parameter(torch.tensor([0]).float(),requires_grad=True)
        self.activation = Activation(activation)

    def forward(self,x):
        K_c = torch_get_RTF(self.A, self.B,x.shape[1])
        h1 = torch_convolution(x, K_c, fft=True)
        y1 = h1 + self.h0 * x
        return self.activation(y1)

class RTF_Block(nn.Module):
    def __init__(
            self,
            hidden_size,
            mult_activation,
            channels,
            final_act='gelu',
            skip = False,
            dropout=0.0,
            norm = False,
                ):
        super().__init__()
        self.encode = nn.Linear(channels,channels)
        self.rtf = SSMRTF_model(hidden_size,channels,mult_activation)
        self.final_act = Activation(final_act)
        self.fc = nn.Linear(channels,channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = skip
        self.normlization = norm
        if norm == 'BN':self.norm = nn.BatchNorm1d(channels)
        if norm == 'LN':self.norm = nn.LayerNorm(channels)
        self.decode = nn.Linear(channels,channels)
    def forward(self,x):
        x = self.encode(x)
        y1 = self.rtf(x)
        y2 = self.fc(y1)
        y2 = self.final_act(y2)
        if self.skip: y2 = y2 + x
        if self.normlization == 'BN' :y2 = self.norm(y2.transpose(1, 2)).transpose(1, 2)
        if self.normlization == 'LN' :y2 = self.norm(y2)
        y2 = self.decode(y2)
        y2 = self.dropout(y2)
        return y2

class S4DKernel(nn.Module):
    """生成S4D-Lin方法的卷积核"""
    def __init__(self, N, channels, dt_min = 1e-3, dt_max = 1e-1):
        super().__init__()
        # Generate dt
        H = channels
        log_dt = torch.rand(H) * (math.log(dt_max)-math.log(dt_min)) + math.log(dt_min)
        B = torch.randn(H, N // 2, dtype=torch.cfloat)
        C = torch.ones(H, N // 2, dtype=torch.float)
        Kc = torch.ones(H, N // 2, dtype=torch.float)
        A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = (math.pi * (torch.arange(N // 2))).repeat(H, 1)
        I = torch.ones(H, N // 2)
        self.I = nn.Parameter(I,requires_grad=False)
        self.log_dt = nn.Parameter(log_dt, requires_grad=True)
        self.B = nn.Parameter(B, requires_grad=True)
        self.C = nn.Parameter(C, requires_grad=True)
        self.Kc = nn.Parameter(Kc, requires_grad=True)
        self.A_real = nn.Parameter(A_real,requires_grad=True)
        self.A_imag = nn.Parameter(A_imag,requires_grad=True)

    def forward(self, L, get_state = False):
        dt = torch.exp(self.log_dt) # (H)
        A = -torch.exp(self.A_real)-(1e-6) + 1j * self.A_imag # (H N)
        # Vandermonde multiplication
        dtA = (self.I + dt.unsqueeze(-1) * A / 2) / ((self.I - dt.unsqueeze(-1) * A / 2)+1e-6)
        dtB = dt.unsqueeze(-1) * self.B / ((self.I - dt.unsqueeze(-1) * A / 2)+1e-6)
        log_dtA = torch.log(dtA)
        log_V = log_dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = self.C * dtB
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(log_V)).real
        if get_state:
            Kc = self.Kc * dtB
            G_k = 2 * torch.einsum('hn, hnl -> hl', Kc, torch.exp(log_V)).real
            return K, G_k
        return K

class S4D_model(nn.Module):
    def __init__(self,hidden_size,channels,activation):
        super().__init__()

        self.kernel = S4DKernel(hidden_size, channels)
        self.D = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.activation = Activation(activation)

    def forward(self,u,get_state = False):
        if get_state:
            K, G_k = self.kernel(u.shape[1],get_state) # (C, N, L)
            y  = torch_convolution(u,K,fft=True)
            feed = torch_convolution(u,G_k,fft=True)
            y = y + self.D * u
            return self.activation(y),feed
        else:
            K = self.kernel(u.shape[1],get_state)
            y = torch_convolution(u,K,fft=True)
            y = y + self.D * u
            return self.activation(y)

class S4D_Block(nn.Module):
    def __init__(
            self,
            hidden_size,
            mult_activation,
            channels,
            final_act='gelu',
            skip = False,
            dropout=0.0,
            norm = False,
            glu = True,
            ):
        super().__init__()
        self.ssm = S4D_model(hidden_size,channels,mult_activation)
        self.final_act = Activation(final_act)
        self.fc = nn.Linear(channels,channels)
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
    def forward(self,x):
        res = x
        if self.normlization == 'BN' :x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        if self.normlization == 'LN' :x = self.norm(x)
        y = self.ssm(x)
        y = self.dropout(y)
        if self.skip: y = y + res

        res = y
        if self.normlization == 'BN' :y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        if self.normlization == 'LN' :y = self.norm(y)
        if self.glu_use:
            z = self.glu_proj(y)
            z = self.glu(z)
        else:
            z = self.fc(y)
        z = self.dropout(z)
        out = z + res

        return out
