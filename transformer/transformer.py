import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

__author__ = "wenqiang su"

class ScaleDotProductAttention(nn.Module):
    """计算注意力权重并输出加权结果"""

    def __init__(self, attention_dropout=0.0):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        计算 注意力值 -> 缩小注意力值 -> 算出权重 -> 加权得到新的词向量
        :param q: query ,形状为 [batch_size, len_q, dim_q]
        :param k: key ,形状为 [batch_size, len_k, dim_k]
        :param v: value ,形状为 [batch_size, len_v, dim_v]
        :param scale: 缩放因子，一个浮点标量
        :param attn_mask: Mask张量，形状为 [batch_size, len_q, len_k]，其中1表示mask掉不计算，0表示没有被mask
        :return: 加权后的输出，和 attention
        """
        # ([batch_size, len_q, dim_q] @ [batch_size, dim_k, len_k]) = [batch_size, len_q, len_k]
        attention = torch.bmm(q, k.transpose(1, 2))   # 计算注意力值
        if scale:
            attention = attention * scale             # 减小注意力值，避免太大的值在softmax之后梯度太小

        if attn_mask is not None:
            # 把不需要计算地方mask掉,把这些部分的注意力值换成一个负无穷的值，这样在softmax之后就可以变成趋向于0了
            attention.masked_fill_(attn_mask, -np.inf)
        # 把注意力值缩放成和为1的权重
        attention = self.softmax(attention)            # 形状为 [batch_size, len_q, len_k]
        attention = self.dropout(attention)
        # 加权,输出包含上下文的词向量
        # [batch_size, len_q, len_k] @ [batch_size, len_v, dim_v] = [batch_size, len_q, dim_v] (len_k = len_v)
        context = torch.bmm(attention, v)              # 形状为 [batch_size, len_q, dim_v]
        return context, attention


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, model_dim=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads  # 每个头的维度
        self.num_heads = num_heads                  # 头的数量
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)  # 生成key的矩阵
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)  # 生成query的矩阵
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)  # 生成value的矩阵
        self.dot_product_attention = ScaleDotProductAttention(dropout)       # 计算attention输出
        self.linear_final = nn.Linear(model_dim, model_dim)                  # 线性层
        self.dropout = nn.Dropout(dropout)                                   # dropout
        self.layer_norm = nn.LayerNorm(model_dim)                            # layer norm

    def forward(self, key, value, query, attn_mask=None):
        """
        多头注意力计算
        :param key:
        :param value:
        :param query:
        :param attn_mask:
        :return: 多头注意力模块的输出， 注意力
        """
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # 生成key，query，value 向量
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # 把向量分成多头(多头在计算attention时相当于并列的样本，类似彩色图片中的红黄蓝三通道在卷积阶段其实可以看成三个样本)
        # 例如： 形状为：[10,20,100] 的"10个句子，每个句子20个字，每个字100维向量"的样本分成4个heads -> [40,20,25]
        key = key.view(batch_size * num_heads, -1, dim_per_head)  # shape[batch_size * num_heads, len_key, dim_per_head]
        value = value.view(batch_size * num_heads, -1, dim_per_head)  # shape 同上
        query = query.view(batch_size * num_heads, -1, dim_per_head)  # shape 同上

        # 上面多头处理时没有改变len维度每个词语的位置，只是增加了batch维度的样本数量，而mask是对位置mask的，所以在batch维度复制即可

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # 计算注意力的加权输出
        scale = (key.size(-1)) ** -0.5  # 多头后的词向量维度的根号
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask
        )

        # 拼接多头
        # shape = [batch_size, len, model_dim]
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # 线性层,dropout
        output = self.linear_final(context)
        output = self.dropout(output)

        # 残差 + layer norm
        output = self.layer_norm(residual + output)
        return output, attention


def padding_mask(seq_k, seq_q):
    """
    构建mask矩阵。
    :param seq_k: key
    :param seq_q:query
    :return: pad matrix
    """
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class PositionalEncoding(nn.Module):
    """位置嵌入"""
    def __init__(self,d_model, max_seq_len):
        """
        初始化位置编码
        :param d_model: 模型内向量维度
        :param max_seq_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        # PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j //2)) / d_model for j in range(d_model)]
            for pos in range(max_seq_len)
        ])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 第一行作为0向量, pad部分的位置编码也是0向量
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.tensor(position_encoding)
        position_encoding = torch.cat((pad_row, position_encoding), dim=0)

        # 构建嵌入矩阵，+1是为了pad的位置编码
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_len):
        """
        获取位置编码
        :param input_len: 一个张量，形状为[batch_size, 1]。每个张量表示这一批文本序列中对应的长度
        :return: 返回这一批序列的位置编码，进行了对齐。
        """
        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        # 对每个序列的位置进行对齐，在原序列位置后面补0
        # 这里每个range也从0开始，避开PAD(0)的位置
        input_pos = tensor(
            [list(range(1, len_seq+1)) + [0] * (max_len - len_seq) for len_seq in input_len]
        )
        return self.position_encoding(input_pos)


class PositionalWiseFeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(in_channels=model_dim, out_channels=ffn_dim, kernel_size=1)
        self.w2 = nn.Conv1d(in_channels=ffn_dim, out_channels=model_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        """
        前馈神经网络层
        :param x: 多头注意力模块的输出，前馈网络的输入 [batch_size, len_seq, dim]
        :return:  一个完整encoder层的输出
        """
        # 残差
        resdual = x
        # mlp
        # 把向量映射到一个高维向量上，然后用relu去筛选有效信息，在返回原来的维度
        output = x.transpose(1, 2)         # shape = [batch_size, dim, len_seq]
        output = F.relu(self.w1(output))   # shape = [batch_size, ffn_dim, len_seq]
        output = self.w2(output)           # shape = [batch_size, dim, len_seq]
        output = output.transpose(1, 2)    # shape = [batch_size, len_seq, dim]
        # add & norm
        output = self.layer_norm(resdual + output)
        return output


class EncoderLayer(nn.Module):
    """单层Encoder"""

    def __init__(self,model_dim=512,num_heads=8,ffn_dim=2021,dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)          # 多头注意力
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)  # 前馈网络

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)               # 计算多头注意力

        # feed forward
        output = self.feed_forward(context)                                          # 前馈网络计算

        return output, attention


class Encoder(nn.Module):
    """多层EcoderLayer"""
    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=3,
                 model_dim=512,
                 num_heads=8,
                 fnn_dim=1024,
                 dropout=0.0
                 ):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, fnn_dim, dropout) for _ in range(num_layers)
        ])

        self.seq_embedding = nn.Embedding(vocab_size+1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)       # 获取字向量
        output += self.pos_embedding(inputs_len)  # 获取位置编码

        self_attention_mask = padding_mask(inputs, inputs)  # 获取padding

        attentions = []

        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)


        return output, attentions

encoder_model = Encoder(vocab_size=52352, max_seq_len=30)
print(encoder_model)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# print(get_parameter_number(encoder_model))


# input_ids = torch.tensor([
#     [1,2,0],
#     [4,5,6]
# ])
# inputs_len = torch.tensor([
#     2,
#     3
# ])
# print(input_ids.shape)
# print(inputs_len.shape)
# print("------------开始推理----------")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#
# encoder_model.to(device)
# input_ids = input_ids.to(device)
# inputs_len = inputs_len.to(device)
# import time
# time1 = time.time()
# output, attentions = encoder_model(input_ids, inputs_len)
# time2 = time.time()
# print("一次前项传输耗时：", time2-time1)
# print(output.shape)
# print(len(attentions))
