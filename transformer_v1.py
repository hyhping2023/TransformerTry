import torch
import torch.nn as nn
import math
from tqdm import tqdm
import functools
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
        self.W_Q = nn.Linear(d_model, d_k * n_head, bias = False, device = device)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias = False, device = device)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False, device=device)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False, device = device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.device = device
        self.attention = SelfAttention(scale_factor = d_k ** 0.5, dropout = dropout, device = device)
        self.layer_norm = nn.LayerNorm(n_head * d_v, device = device)
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
        self.normal = nn.LayerNorm(d_model, device = device)
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

    
    def forward(self, inputs:list, ignore:list = []):
        for inp in ignore:
            assert len(inp) == 1
        out = [self.src_embedding.input_embed(inp) for inp in inputs]
        out = torch.cat(out, dim=0)
        attention_mask = self.src_embedding.mask_useless_tokens(inputs, ignore, None, self.device)
        # attention_mask = None
        attentions = []
        for layer in self.layers:
            out, attention = layer(out, attention_mask)
            attentions.append(attention)
        return out, attentions


class DecoderLayer(nn.Module):
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.dec_self_attention = MultiHeadAttention(device=device)
        self.dec_enc_attention = MultiHeadAttention(device=device)
        self.feed_forward = FeedForward(device=device)
        self.device = device

    def forward(self, dec_inputs, enc_outputs, dec_self_mask, dec_enc_mask):
        out, self_attention = self.dec_self_attention(dec_inputs, dec_inputs, dec_inputs, dec_self_mask)
        out, enc_attention = self.dec_enc_attention(out, enc_outputs, enc_outputs, dec_enc_mask)
        out = self.feed_forward(out)
        return out, self_attention, enc_attention

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_layers, device = 'cuda'):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(device=device) for _ in range(n_layers)])
        self.tgt_embedding = Embedding(device = device)
        self.device = device

    
    def forward(self, dec_inputs, enc_inputs, enc_outputs, ignore = []):
        dec_outputs = torch.cat([self.tgt_embedding.input_embed(dec_input) for dec_input in dec_inputs])
        dec_self_ignore_mask = self.tgt_embedding.mask_useless_tokens(dec_inputs, ignore, None, self.device)
        dec_self_subseq_mask = self.tgt_embedding.get_attn_subsequence_mask(dec_inputs, self.device)

        dec_self_mask = torch.gt((dec_self_subseq_mask + dec_self_ignore_mask), 0)
        
        dec_enc_mask = self.tgt_embedding.mask_useless_tokens(dec_inputs, ignore, refer = enc_inputs, device=self.device)
        
        dec_attentions, dec_enc_attentions = [], []
        for layer in self.layers:
            dec_outputs, self_attention, enc_attention = layer(dec_outputs, enc_outputs, dec_self_mask, dec_enc_mask)
            dec_attentions.append(self_attention)
            dec_enc_attentions.append(enc_attention)
        return dec_outputs, dec_attentions, dec_enc_attentions      


class Embedding(nn.Module):
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.projection = Tokenizer(pre_model_file=True, pre_index_file=True,
                          load_model_file='model_norelate.pkl', load_index_file='chars.pkl', 
                          train_data='train1.txt', device=device)
        self.device = device
        # self.embedding = nn.Embedding(len(self.projection.chars), 512)

    @functools.lru_cache(maxsize=8196)
    def input_embed(self, text:str):
        return self.PositionalEmbedding(self.projection.input_embedding(text, 0, 0, self.device))
    
    def output_project(self, x, topk:int = 3, toChar: bool = False):
        return [self.projection.output_prediction(x[_, -1:, :], topk=topk, toChar=toChar) for _ in range(x.shape[0])]

    def mask_useless_tokens(self, sequence:list, ignore:list, refer:list = None, device = 'cuda'):
        '''
        sequence is usd for query
        refer us used for key
        '''
        len_Q = len(sequence[0])
        if refer is not None:
            len_R = len(refer[0])
        else:
            len_R = len_Q
            refer = sequence
        batch_size = len(refer)
        refer = torch.stack([torch.tensor(self.projection.tokenize(text)) for text in refer])
        ignore = [torch.tensor(self.projection.tokenize(text))for text in ignore]
        mask = torch.zeros(refer.shape)
        for text in ignore:
            mask += (refer == text).int()
        mask = (mask > 0).int().unsqueeze(1).expand(batch_size, len_Q, len_R)
        return mask.to(device)

    def get_attn_subsequence_mask(self, sequence, device = 'cuda'):
        sub_mask = torch.triu(torch.ones([len(sequence), len(sequence[0]), len(sequence[0])]), diagonal=1)
        return sub_mask.to(device)

    def PositionalEmbedding(self, text_vector):
        pe = torch.zeros(text_vector.shape, device=self.device)
        poition = torch.arange(0, text_vector.shape[0], dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, text_vector.shape[1], 2, device=self.device).float() * (-math.log(10000.0) / text_vector.shape[1]))
        pe[:, 0::2] = torch.sin(poition * div_term)
        pe[:, 1::2] = torch.cos(poition * div_term)
        text_vector = text_vector + pe
        return text_vector.unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.embedding = Embedding(device=device)
        self.device = device
        self.multi_head_attention = MultiHeadAttention(device=device)
        self.encoder = TransformerEncoderBlock(n_layer=6, device = self.device)
        self.decoder = TransformerDecoderBlock(n_layers=6, device= self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.01)

    def ignore_split(self, text) -> list:
            result = []
            for item in text:
                [result.append(_) for _ in item]
            return result

    def forward(self, enc_inputs, dec_inputs, ignore = []):
        ignore = self.ignore_split(ignore)
        
        enc_outputs, enc_attention = self.encoder.forward(enc_inputs, ignore)

        dec_outputs, dec_attention, dec_enc_attention = self.decoder.forward(dec_inputs, enc_inputs, enc_outputs, ignore)
        
        return self.embedding.projection.output(dec_outputs)
    
    def predict(self, enc_inputs:list, dec_inputs:list, step = 30, ignore = []):
        assert len(enc_inputs) == len(dec_inputs)
        ignore = self.ignore_split(ignore)
        for _ in range(step):
            enc_outputs, enc_attention = self.encoder.forward(enc_inputs, ignore)
            dec_outputs, dec_attention, dec_enc_attention = self.decoder.forward(dec_inputs, enc_inputs, enc_outputs, ignore)
            dec_outputs = self.embedding.projection.output(dec_outputs)
            dec_outputs = torch.argmax(dec_outputs, dim = -1)
            dec_inputs = self.addToDecode(dec_inputs, dec_outputs.unsqueeze(1))
        return dec_inputs
    
    def addToDecode(self, decode_input, decode_output):
        chars = self.embedding.projection.toChar(decode_output)
        for i in range(len(decode_input)):
            decode_input[i] += chars[i]
        return decode_input
    
    def dataLoader(self, raw_texts:str, seqence_length:int, start:int, iter:int, batch_size:int) -> list:
        texts = []
        inx = start
        while inx + seqence_length < len(raw_texts) and (inx - start)//seqence_length < iter:
            texts.append(raw_texts[inx:inx+seqence_length])
            inx += seqence_length
        if start + seqence_length > len(raw_texts):
            status = -1
        else:
            status = inx
        return [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)], status

    def train(self, train_data:str, epoches = 100, iter_size = 1000, batch_size = 128, seqence_length = 20,
              continue_training = False, output_file:str = 'transformer.pkl', device = 'cuda'): 
        status = 0
        with open(train_data, 'r', encoding='utf-8') as f:
            raw_texts = f.readlines()
            raw_texts = ''.join(raw_texts).replace('\n', '').replace('  ', '')
            while status != -1:
                train_batches, status = self.dataLoader(raw_texts, seqence_length, status, iter_size, batch_size)
                print(len(train_batches))
                for epoch in range(epoches):
                    for inx, batch in enumerate(train_batches):
                        decoder_inputs = ['\n']*len(batch)
                        for _ in tqdm(range(seqence_length)):
                            encoder_inputs = batch
                            self.optimizer.zero_grad()
                            outputs = self.forward(encoder_inputs, decoder_inputs)
                            target = torch.tensor([self.embedding.projection.tokenize(ba[_])[0] for ba in batch], device = self.device)
                            # print(outputs.shape, torch.tensor([self.embedding.projection.tokenize(ba[_]) for ba in batch], device = self.device).shape)
                            loss = self.loss(outputs, target)
                            loss.backward()
                            self.optimizer.step()
                            decoder_inputs = self.addToDecode(decoder_inputs, target.unsqueeze(-1))
                        # print(self.optimizer.param_groups[0])
                        # print(decoder_inputs)
                        print(f'Epoch {epoch+1}, Batch {inx} finished, loss: {loss}  ', end='\n')
                        print('你好，我叫阿尼亚\nPredict:', self.predict(['你好，我叫阿尼亚'], ['\n']))
                        torch.save(self.state_dict(), output_file)

if __name__ == '__main__':
    transformer = Transformer(device='cuda:1')
    # print(transformer.embedding.output_project(transformer.embedding.projection.input_embedding('你好，我叫阿尼亚', 0, 0).unsqueeze(0)))
    # transformer.test_function('你好，我叫阿尼亚')
    # result = transformer.forward(['你好，我叫阿尼亚', '我真的不是基佬啊'], ['\n你1', '\n我1'], ['<bos>', '<eos>'])
    # print(result)
    # print(transformer.optimizer.param_groups)
    # import cProfile
    # cProfile.run('transformer.train("./train1.txt", epoches=1, iter_size=1000, batch_size=512, seqence_length=20)', sort='cumtime')
    transformer.train('./train1.txt', epoches=100, iter_size=100000, batch_size=4096, seqence_length=30, device='cuda:1')
    
    transformer = Transformer(device='cuda:2')
    transformer.load_state_dict(torch.load('transformer.pkl'))
    results = transformer.predict(['你好，我叫阿尼亚', '我真的不是基佬啊'], ['\n', '\n'], 50)
    for result in results:
        print(result)
