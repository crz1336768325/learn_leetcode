import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

# transformer 
# encoder -decoder 结构
#multi self-attention
#  需要给出位置信息 因transformer本身不具备如rnn一样的序列信息，因此需给出位置信息，对位置进行编码
# Q,K,V 

# inputs-》positonal encoding
class scale_dot_production(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature=temperature
        self.attention_dropout=nn.Dropout(attn_dropout)

    def forward(self,q,k,v,mask=None): # q,k,v 的shape构造 (batchSize, n_head, seqLen, dim) 
        #batch_size表示批次大小，也就是一个批次含有多少个序列(句子)；seq_len表示一个序列(句子)的长度；
        # input_dim表示每个单词使用长度是input_dim的向量来表示
        scale_dot1=torch.matmul(q/self.temperature,k.transpose(2,3))
        net_1 = nn.Softmax(dim=0)
        attention=net_1(scale_dot1)
        return torch.matmul(attention,v),attention
# 多头机制在scale_dot_production 基础之上构建
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    # n_head头的个数，默认是8
    # d_model编码向量长度，例如本文说的512
    # d_k, d_v的值一般会设置为 n_head * d_k=d_model，
    # 此时concat后正好和原始输入一样，当然不相同也可以，因为后面有fc层
    # 相当于将可学习矩阵分成独立的n_head份
    # qkv 的原始输入为(b,100,512)， 100是单词len ，512 是input_dim
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.d_k=d_k
        self.d_v=d_v
        self.n_head=n_head
        self.d_model=d_model
        self.w_q=nn.Linear(self.d_model,self.d_k*self.n_head)
        self.w_k=nn.Linear(self.d_model,self.d_k*self.n_head)
        self.w_v=nn.Linear(self.d_model,self.d_v*self.n_head)
        self.dot_production=scale_dot_production(self.d_k**0.5)

    # 所谓的自注意力机制就是 qkv 是相等的
    def forward(self,q,k,v,mask=None):
        residual=q
        q=self.w_q(q).view(4,8,100,self.d_k)
        k=self.w_k(k).view(4,8,100,self.d_k)
        v=self.w_v(v).view(4,8,100,self.d_v)
        q,atten=self.dot_production(q,k,v)
        q=q.contiguous().view(4,100,-1)
        print("q",q.shape)
        print("residual",residual.shape)
        q+=residual
        return q,atten

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # 两个fc层，对最后的512维度进行变换
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        # Q K V是同一个，自注意力
        # enc_input来自源单词嵌入向量或者前一个编码器输出
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
if __name__ == "__main__":
    EL=EncoderLayer(512,100,8,64,64)
    inputs=torch.ones([4,100,512])
    out,atten=EL(inputs)
    print("out shape",len(out))
    print("out shape",out.shape,atten.shape)
