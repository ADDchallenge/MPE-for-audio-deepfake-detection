import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy import signal as sig

def sampEn(L:np.array, std : float ,m: int= 2, r: float = 0.15):
    """ 
    计算时间序列的样本熵
    
    Input: 
        L: 时间序列
        std: 原始序列的标准差
        m: 1或2
        r: 阈值
        
    Output: 
        SampEn
    """
    N = len(L)
    B = 0
    A = 0

    # Split time series and save all templates of length m
    x_reconstruct=np.array([L[i:i+m] for i in range(N-m+1)])
    for x_temp in x_reconstruct:
        B+=(np.sum(np.abs(x_reconstruct-x_temp).max(axis=1) <= r * std)-1)
    m=m+1
    x_reconstruct=np.array([L[i:i+m] for i in range(N-m+1)])
    for x_temp in x_reconstruct:
        A+=(np.sum(np.abs(x_reconstruct-x_temp).max(axis=1) <= r * std)-1)
    
    # xmi = np.array([L[i:i+m] for i in range(N-m)])
    # xmj = np.array([L[i:i+m] for i in range(N-m+1)])
    
    # # Save all matches minus the self-match, compute B
    # B = np.sum([np.sum(np.abs(xmii-xmj).max(axis=1) <= r * std)-1 for xmii in xmi])
    # # Similar for computing A
    # m += 1
    # xm = np.array([L[i:i+m] for i in range(N-m+1)])
    
    # A = np.sum([np.sum(np.abs(xmi-xm).max(axis=1) <= r * std)-1 for xmi in xm])
    # # Return SampEn
    # return -np.log(A/B)


    if B==0:
        SE=float('nan')
    else:
        SE=-np.log(A/B)
        
    return SE

def MSE(signal , max_scale:int = 20):
    result = []
    # signal=signal[::2]
    length=len(signal)
    std = np.std(signal)
    for scale in range(1 , max_scale + 1):
        # 确定截取的长度
        length_scale = length % scale 
        signal_scale=np.array(signal[:length-length_scale]).reshape(-1,scale)
        signal_new=np.mean(signal_scale,axis=1)  
        # result.append(sampEn(signal_new, std ,r = 0.15))
        result.append(Permutation_Entropy(signal_new, 10 ,2))
        # print("scale:" , scale, 'SampEn' , result[-1])
    return result
    # for scale in range(1 , max_scale + 1):
    #     # 确定截取的长度
    #     length = int(len(signal) / scale) - 1
    #     # 分段取平均
    #     scale_i = signal[ : len(signal) : scale][:length]
    #     for i in range(1,scale):
    #         scale_i = scale_i + signal[i: len(signal) : scale][:length]
    #     scale_i = scale_i / scale
    #     #计算样本熵
    #     result.append(sampEn(scale_i, std ,r = 0.15))
    #     print("scale:" , scale, 'SampEn' , result[-1])
    # return result



def func(n):
    """求阶乘"""
    if n == 0 or n == 1:
        return 1
    else:
        return (n * func(n - 1))

def compute_p(S):
    """计算每一种 m 维符号序列的概率"""
    _map = {}
    for item in S:
        a = str(item)
        if a in _map.keys():
            _map[a] = _map[a] + 1
        else:
            _map[a] = 1
            
    freq_list = []
    for freq in _map.values():
        freq_list.append(freq / len(S))
    return freq_list

def Permutation_Entropy(x, m, t):
    """计算排列熵值"""
    length = len(x) - (m-1) * t
    # 重构 k*m 矩阵
    y = [x[i:i+m*t:t] for i in range(length)]
    
    # 将各个分量升序排序
    S = [np.argsort(y[i]) for i in range(length)]
    
    # 计算每一种 m 维符号序列的概率
    freq_list = compute_p(S)
    
    # 计算排列熵
    pe = 0
    for freq in freq_list:
        pe += (- freq * np.log(freq))
    
    return pe / np.log(func(m))



