import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=1)
    def forward(self, Query, Key, Value, masked= False):
        scaled_dot_product = torch.matmul(Query, Key.mT) / torch.sqrt(torch.tensor(self.d_k, dtype= torch.float32))
        if masked:
            self.mask = torch.triu(torch.multiply(torch.ones_like(scaled_dot_product), float("-inf")), diagonal=1)
            scaled_dot_product = scaled_dot_product+self.mask
        attention_pattern = self.softmax(scaled_dot_product)
        delta_E = torch.matmul(attention_pattern, Value)

        return delta_E

class SimpleAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super(SimpleAttention, self).__init__()
        self.d_model = d_model

        self.query = nn.Linear(d_model, d_k, bias=False)
        self.key = nn.Linear(d_model, d_k, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Query, Key, Value, masked= False):
        self.Q = self.query(Query)
        self.K = self.key(Key)
        self.V = self.value(Value)

        self.delta_E = self.attention(self.Q, self.K, self.V, masked)

        return self.delta_E


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.query = nn.Linear(d_model, self.d_k*num_heads, bias= False)
        self.key = nn.Linear(d_model, self.d_k*num_heads, bias= False)
        self.value = nn.Linear(d_model, self.d_k*num_heads, bias= False)
        self.output_matrix = nn.Linear(num_heads * self.d_k, d_model, bias= False)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Query, Key, Value, masked= False):
        batch_size = Query.shape[0]

        self.Q = self.query(Query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        self.K = self.key(Key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        self.V = self.value(Value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        self.delta_E = self.attention(self.Q, self.K, self.V, masked).transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        self.output = self.output_matrix(self.delta_E)

        return self.output