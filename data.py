
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import sys
sys.path.insert(0, '../')

#modeldir = '/home/mpark/fvtranscribe/BertPunc/models/20191115_184654/'

#with open(modeldir+'hyperparameters.json', 'r') as f:
#    hyperparameters = json.load(f)
#hyperparameters

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

punctuation_enc = {
    'O': 0,
    'COMMA': 1,
    'PERIOD': 2,
    'QUESTION': 3
}

punctuation_dec = {
    1: ',',
    2: '.',
    3: '?'
}

punctuation_enc_new = {'O':0,
                       ',':1,
                      '.':2,
                      '?':3}

def split2sent(string, keeppunc = True, splitcomma=False):
    if splitcomma:
        split = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!\,)\s', string)
    else:
        split = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', string)
    
    if not keeppunc:
        split = [re.sub('\.|\?|\!|\,|\;', '', x) for x in split]
        split = [re.sub('\s+', ' ', x).strip() for x in split]
    return [x.strip() for x in split]

def split2word(string, keeppunc = False):
    if not keeppunc:
        string = re.sub('\.|\?|\!|\,|\;', '', string)
        string = re.sub('\s+', ' ', string).strip()
    return string.split()

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

def encode_data(data, tokenizer, punctuation_enc):
    """
    Converts words to (BERT) tokens and puntuation to given encoding.
    Note that words can be composed of multiple tokens.
    """
    X = []
    Y = []
    for line in data:
        if len(line.split('\t')) == 1: continue
        word, punc = line.split('\t')
        punc = punc.strip()
        tokens = tokenizer.tokenize(word)
        x = tokenizer.convert_tokens_to_ids(tokens)
        y = [punctuation_enc[punc]]
        if len(x) > 0:
            if len(x) > 1:
                y = (len(x)-1)*[0]+y
            X += x
            Y += y
    return X, Y

def insert_target(x, segment_size):
    """
    Creates segments of surrounding words for each word in x.
    Inserts a zero token halfway the segment.
    """
    X = []
    x_pad = x[-((segment_size-1)//2-1):]+x+x[:segment_size//2]

    for i in range(len(x_pad)-segment_size+2):
        segment = x_pad[i:i+segment_size-1]
        segment.insert((segment_size-1)//2, 0)
        X.append(segment)

    return np.array(X)

def preprocess_data(data, tokenizer, punctuation_enc, segment_size):
    X, y = encode_data(data, tokenizer, punctuation_enc)
    X = insert_target(X, segment_size)
    return X, y

def preprocess_train(diary, tokenizer, punctuation_enc, segment_size):
    """
    Converts words to (BERT) tokens and puntuation to given encoding.
    Note that words can be composed of multiple tokens.
    """
    X = []
    Y = []
    for isent in sorted(diary.keys()):
        for puncfrag in split2sent(diary[isent]['sentence'], keeppunc=True, splitcomma=True):
            puncfrag = puncfrag.strip()
            assert puncfrag[-1] in punctuation_enc_new.keys()
            word, punc = puncfrag[:-1].strip(), punctuation_enc_new[puncfrag[-1]]
        
            tokens = tokenizer.tokenize(word)
            x = tokenizer.convert_tokens_to_ids(tokens)
            y = [punctuation_enc[punc]]
            
            if len(x) > 0:
                if len(x) > 1:
                    y = (len(x)-1)*[0]+y
                X += x
                Y += y
    X_seg = insert_target(X, segment_size)
    return X_seg, Y

def preprocess_pred(diary, tokenizer, punctuation_enc, segment_size):
    """
    Converts words to (BERT) tokens and puntuation to given encoding.
    Note that words can be composed of multiple tokens.
    """
    X = []
    Y = []
    for isent in sorted(diary.keys()):
        for puncfrag in split2word(diary[isent]['sentence'], keeppunc=False):
            word, punc = puncfrag.strip(), 0
        
            tokens = tokenizer.tokenize(word)
            x = tokenizer.convert_tokens_to_ids(tokens)
            y = [punc]
            if len(x) > 0:
                if len(x) > 1:
                    y = (len(x)-1)*[0]+y
                X += x
                Y += y
    X_seg = insert_target(X, segment_size)
    return X_seg, Y

def create_data_loader(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader
