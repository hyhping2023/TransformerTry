import torch
import torch.nn as nn
import math
from tokenizer import Tokenizer, EmbeddingModel

class SelfAttention(nn.Module):
    def __init__(self, scale_factor, dropout = 0., device = 'cuda'):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(dropout).to(device)
        self.device = device
    def forward(self, q, k, v, mask = None):
        attention = torch.matmul(q, k.transpose(-2, -1)) / self.scale_factor
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(attention, dim = -1)
        attention = self.dropout(attention)
        return torch.matmul(attention, v), attention

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head = 8, d_model= 512, d_k = 64, d_v = 64, dropout = 0.1, device = 'cuda'):
        super().__init__()
        assert d_model % n_head == 0 and d_k * n_head == d_model 
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.W_Q = nn.Linear(d_model, d_k * n_head, bias = False).to(device)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias = False).to(device)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False).to(device)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.device = device
        self.attention = SelfAttention(scale_factor = d_k ** 0.5, dropout = dropout, device = device)
        self.layer_norm = nn.LayerNorm(n_head * d_v).to(device)
        self.init_weights()

    def init_weights(self):
        def method(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
        self.W_Q.apply(method)
        self.W_K.apply(method)
        self.W_V.apply(method)
        self.fc.apply(method)

    def forward(self, q, k, v, mask = None):
        batch_size, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)
        q = self.W_Q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.W_K(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.W_V(v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        out, attention = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_v)
        out = self.layer_norm(out)
        out = self.fc(out)
        out = self.dropout(out)
        return out, attention
    
class FeedForward(nn.Module):
    def __init__(self, d_model = 512, d_ff = 2048, device = 'cuda'):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        ).to(device)
        self.normal = nn.LayerNorm(d_model).to(device)
        self.init_weights()
    def init_weights(self):
        def method(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
        self.net.apply(method)
    def forward(self, x):
        out = x + self.net(x)
        return self.normal(out)

class EncoderLayer(nn.Module):
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(device=device)
        self.feed_forward = FeedForward(device=device)
    def forward(self, inputs, attention):
        out, attention = self.multi_head_attention(inputs, inputs, inputs, attention)
        out = self.feed_forward(out)
        return out, attention

class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_layer, device = 'cuda'):
        super().__init__()
        self.src_embedding = Embedding(device = device)
        self.layers = nn.ModuleList([EncoderLayer(device=device) for _ in range(n_layer)])
        self.device = device

    def mask_useless_tokens(self, sequence:list, ignore:list, device = 'cuda'):
        len_Q = len(sequence[0])
        batch_size = len(sequence)
        sequence = torch.stack([torch.tensor(self.src_embedding.projection.tokenize(text)) for text in sequence])
        ignore = [torch.tensor(self.src_embedding.projection.tokenize(text))for text in ignore]
        mask = torch.zeros(sequence.shape)
        for text in ignore:
            mask += (sequence == text).int()
        mask = (mask > 0).int().unsqueeze(1).expand(batch_size, len_Q, len_Q)
        return mask.to(device)

    def forward(self, inputs:list, ignore:list = []):
        for inp in ignore:
            assert len(inp) == 1
        out = [self.src_embedding.input_embed(inp) for inp in inputs]
        out = torch.cat(out, dim=0)
        attention_mask = self.mask_useless_tokens(inputs, ignore, self.device)
        # attention_mask = None
        attentions = []
        for layer in self.layers:
            out, attention = layer(out, attention_mask)
            attentions.append(attention)
        return out, attentions


class TransformerDecoderBlock(nn.Module):
    def __init__(self, ):
        super().__init__()



class Embedding(nn.Module):
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.projection = Tokenizer(pre_model_file=True, pre_index_file=True,
                          load_model_file='model_norelate.pkl', load_index_file='chars.pkl', 
                          train_data='train1.txt')
        self.device = device
        # self.embedding = nn.Embedding(len(self.projection.chars), 512)

    def input_embed(self, text:str):
        return self.PositionalEmbedding(self.projection.input_embedding(text, 0, 0, self.device))
    
    def output_project(self, x):
        return [self.projection.output_prediction(x[_]) for _ in range(x.shape[0])]

    def PositionalEmbedding(self, text_vector):
        pe = torch.zeros(text_vector.shape)
        poition = torch.arange(0, text_vector.shape[0], dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, text_vector.shape[1], 2).float() * (-math.log(10000.0) / text_vector.shape[1]))
        pe[:, 0::2] = torch.sin(poition * div_term)
        pe[:, 1::2] = torch.cos(poition * div_term)
        pe = pe.to(self.device)
        text_vector = text_vector + pe
        return text_vector.unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.embedding = Embedding()
        self.multi_head_attention = MultiHeadAttention()
        self.encoder = TransformerEncoderBlock(n_layer=6)
        self.decoder = TransformerDecoderBlock()

    def forward(self, text:str, ignore:list = []):
        def ignore_split(text) -> list:
            result = []
            for item in text:
                [result.append(_) for _ in item]
            return result
        self.encoder.forward(text, ignore_split(ignore))

if __name__ == '__main__':
    transformer = Transformer()
    # print(transformer.embedding.output_project(transformer.embedding.projection.input_embedding('你好，我叫阿尼亚', 0, 0).unsqueeze(0)))
    # transformer.test_function('你好，我叫阿尼亚')
    transformer.forward(['你好，我叫阿尼亚', '我真的不是基佬哦'], ['你好'])
