import os
import pickle
import torch
import torch.nn as nn
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_length:int, hidden_size = 512, device = 'cuda'):
        super().__init__()
        self.device = device
        self.net1 = nn.Sequential(
            nn.Linear(vocab_length, hidden_size),
        ).to(device)
        self.net2 = nn.Sequential(
            nn.Linear(hidden_size, vocab_length),
        ).to(device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam((*self.net1.parameters(), *self.net2.parameters()), lr = 0.001)
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.net1.apply(init_weights)
        self.net2.apply(init_weights)
    def forward(self, x):
        return self.net2(torch.mean(self.net1(x), dim=1))
    
    def change_device(self, device):
        self.device = device
        self.net1 = self.net1.to(device)
        self.net2 = self.net2.to(device)

    def train(self, train_data:list, epoches = 100):
        device_train_data = []
        for idx, (x_iter, y_iter) in enumerate(train_data):
            device_train_data.append((x_iter.to(self.device), y_iter.to(self.device)))
        for epoch in range(epoches):
            for idx, (x_iter, y_iter) in enumerate(device_train_data):
                # x_iter = x_iter.to(self.device)
                # y_iter = y_iter.to(self.device)
                y_hat = self.forward(x_iter)
                loss = self.loss(y_hat, y_iter)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f'Epoch {epoch+1}, Batch {idx+1} finished', end='\r')
                # x_iter.cpu()
                # y_iter.cpu()
            if epoch % 10 == 9:
                print(f'epoch: {epoch+1}, loss: {loss}                 ')
        del device_train_data

class Tokenizer:
    model = None
    def __init__(self, pre_model_file:bool, pre_index_file:bool, 
                 load_pre_data_file = None, load_model_file = None, load_index_file = None, train_data = None) -> None:
        self.train_data = train_data
        if not pre_index_file:
            self.pre_loading(load_pre_data_file)
            with open('chars.pkl', 'wb') as f:
                pickle.dump(self.chars, f)
        else:
            with open(load_index_file, 'rb') as f:
                self.chars = pickle.load(f)
        if not pre_model_file:
            self.train_embedding(train_data, 3, 3)
        else:
            with open(load_model_file, 'rb') as f:
                self.model = pickle.load(f)
        self.reverseDict = {v: k for k, v in self.chars.items()}

    def change_device(self, device):
        self.device = device
        self.model.change_device(device)
            
    def get_hot_vector(self, inx:int, length:int) -> list:
        hot_vector = torch.zeros(length)
        hot_vector[inx] = 1
        return hot_vector
        
    def pre_loading(self, raw_text) -> None:
        chars = []
        with open(raw_text, 'r', encoding='utf-8') as f:
            raw_texts = f.readlines()
            for raw_text in raw_texts:
                if raw_text == '\n':
                    continue
                else:
                    raw_text = [_ for _ in raw_text]
                    chars.extend(raw_text)
        chars = list(set(chars))
        loadChars = {}
        for inx, char in enumerate(chars):
            loadChars[char] = inx
        loadChars['<unk>'] = max(loadChars.values())+1
        self.chars = loadChars

    def tokenize(self, text:str) -> list:
        tokens = []
        for char in text:
            if char in self.chars.keys():
                tokens.append(self.chars[char])
            else:
                tokens.append(max(self.chars.values()))
        return tokens
    
    def train_embedding(self, train_data:list, forward_window:int, backward_window:int, continue_training = False, deivce = 'cuda', output_file:str = 'model.pkl') -> list:
        def data_loader(data_dir:str, forward_window:int, backward_window:int, start:int, 
                        shape:int, batch_size:int) -> list:
            # def padding(x:list, forward_window:int, backward_window:int) -> list:
            #     if len(x) <= 2:
            #         return [self.tokenize(['\n' for _ in range(forward_window)]) + x_ + self.tokenize(['\n' for _ in range(backward_window)]) for x_ in x]
            #     result = [self.tokenize(['\n' for _ in range(forward_window)]) + x[0] + self.tokenize(['\n' for _ in range(backward_window)])]
            #     inx = 1
            #     while inx < len(x) - 1:
            #         result.append(self.tokenize([x[inx-1][-_-1] if len(x[inx-1]) > _ else '\n' for _ in range(forward_window)]) 
            #                       + x[inx] + [self.tokenize([x[inx+1][_] if len(x[inx+1]) > _ else '\n' for _ in range(backward_window)])])
            #         inx += 1
            #     result.append(self.tokenize([x[inx-1][-_-1] if len(x[inx-1]) > _ else '\n' for _ in range(forward_window)]) 
            #                   + x[inx] + self.tokenize(['\n' for _ in range(backward_window)]))
            #     return result
                

            with open(data_dir, 'r', encoding='utf-8') as f:
                ori_raw_texts = f.readlines()
                raw_tokens = []
                if len(ori_raw_texts) > shape:
                    raw_texts = ori_raw_texts[start:start+shape]
                else:
                    raw_texts = ori_raw_texts
                for raw_text in raw_texts:
                    raw_tokens.extend(self.tokenize(raw_text))
                raw_tokens = self.tokenize(['\n']*(forward_window-len(ori_raw_texts[start-forward_window:start]))+ori_raw_texts[start-forward_window:start]) + raw_tokens + self.tokenize(['\n']*(backward_window-len(ori_raw_texts[start+shape+1:start+shape+backward_window+1]))+ori_raw_texts[start+shape+1:start+shape+backward_window+1])
                raw_hot_vectors = [self.get_hot_vector(token, len(self.chars)) for token in raw_tokens]
                loader = []
                for idx in range(forward_window, len(raw_hot_vectors)-backward_window):
                    loader.append((torch.stack(raw_hot_vectors[max(0,idx-forward_window):idx+backward_window+1]), raw_hot_vectors[idx]))
                if start + shape >= len(ori_raw_texts):
                    status = -1
                else:
                    status = start + shape
                # print(len(loader))
                loader = torch.utils.data.DataLoader(loader, batch_size=batch_size, shuffle=True)
                return loader, status
                
        assert self.chars is not None
        if continue_training:
            model = self.model
        else:
            model = EmbeddingModel(len(self.chars), hidden_size=512, device='cuda')
            self.model = model
        self.change_device(deivce)
        status = 0
        while status != -1:
            train_data, status = data_loader(self.train_data, forward_window, backward_window, 
                                             status, shape=1000, batch_size=8192)
            model.train(train_data, epoches=200)
            # print(status)
            del train_data
            with open(output_file, 'wb') as f:
                pickle.dump(model, f)
        self.model = model

    def input_embedding(self, text:str, forward_window:int = 1, backward_window:int = 1, device = 'cuda') -> list:
        assert self.model is not None
        tokens = self.tokenize(text)
        hot_vectors = [self.get_hot_vector(token, len(self.chars)) for token in tokens]
        with torch.no_grad():
            embedding_vectors = [torch.mean(self.model.net1(torch.stack(hot_vectors[max(0,idx-forward_window):idx+backward_window+1]).to(device)), dim=0) for idx in range(len(tokens))]
        return torch.stack(embedding_vectors)
    
    def output_prediction(self, tensor, topk:int = 3) -> list:
        assert self.model is not None
        prediction = torch.argsort(self.model.net2(tensor), descending=True)[:, 0:topk].to('cpu').tolist()
        prediction = [[self.reverseDict[i] for i in pred] for pred in prediction]
        return prediction

def continue_training(model:Tokenizer, data_dir:str, device = 'cuda'):
    assert model.model is not None
    model.train_embedding(data_dir, 3, 3, continue_training=True, output_file='model.pkl')


if __name__ == '__main__':
    # tokenizer = Tokenizer(pre_model_file=False, pre_index_file=False, 
    #                       load_pre_data_file='Chinese7000.txt', train_data='train1.txt')
    tokenizer = Tokenizer(pre_model_file=True, pre_index_file=True,
                          load_model_file='model_norelate.pkl', load_index_file='chars.pkl', 
                          train_data='train1.txt')
    # print(tokenizer.input_embedding('你好, 我爱你中国,我最喜欢东风31甲改', device='cuda').shape)

    ni = tokenizer.output_prediction(tokenizer.input_embedding('你好，我叫阿尼亚', 0, 0, device='cuda'), topk=5)
    sentence = ''
    for _ in ni:
        sentence += _[0][0]
    print(ni)
    print(sentence)

    # tokenizer.change_device('cuda:1')
    # print(tokenizer.model.device)
    # continue_training(tokenizer, 'train1.txt')
