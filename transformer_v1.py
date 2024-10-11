import torch
import torch.nn as nn
from tokenizer import Tokenizer

class TransformerEncoderBlock(nn.Module):
    def __init__(self, ):
        super().__init__()

class TransformerDecoderBlock(nn.Module):
    def __init__(self, ):
        super().__init__()

class Transformer(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.embedding = Tokenizer(pre_model_file=True, pre_index_file=True,
                          load_model_file='model.pkl', load_index_file='chars.pkl', 
                          train_data='train1.txt')

if __name__ == '__main__':
    transformer = Transformer()
