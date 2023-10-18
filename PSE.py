import numpy as np
import matplotlib.pyplot as plt

def pse(x,fs,nfft):
    [Pxx1,_] = plt.psd(x,                       # 随机信号
                    NFFT=nfft,               # 每个窗的长度
                    Fs=fs,                   # 采样频率
                    detrend='mean',          # 去掉均值
                    window=np.hanning(nfft), # 加汉尼窗
                    noverlap=int(nfft*3/4),  # 每个窗重叠75%的数据
                    sides='twosided') 
    p=Pxx1**2/sum(Pxx1**2)
    H=-sum(p*np.log(p))
    return Pxx1,H
