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
        self.tokenizer = Tokenizer(True, 'Chinese7000.txt', 'chars.pkl', True)

if __name__ == '__main__':
    transformer = Transformer()
