import os
import pickle

class Tokenizer:
    def __init__(self, load_file:bool, raw_text:str, data_dir:str, save:bool) -> None:
        if not load_file:
            self.chars = self.pre_loading(raw_text)
            if save:
                with open(data_dir, 'wb') as f:
                    pickle.dump(self.chars, f)
        else:
            with open(data_dir, 'rb') as f:
                self.chars = pickle.load(f)
        
    def pre_loading(self, raw_text) -> list:
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
        chars = loadChars
        return chars

    def tokenize(self, text:str) -> list:
        tokens = []
        for char in text:
            if char in self.chars.keys():
                tokens.append(self.chars[char])
            else:
                tokens.append(max(self.chars.values())+1)
        return tokens

if __name__ == '__main__':
    tokenizer = Tokenizer(True, 'Chinese7000.txt', 'chars.pkl', True)
    # print(tokenizer.chars)
    print(tokenizer.tokenize('我爱北京天安门,'))
