import os
import requests
import tiktoken
import numpy as np
import glob
from transformers import AutoModel,AutoTokenizer

# download the tiny shakespeare dataset
# for x in 
# input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# if not os.path.exists(input_file_path):
#     data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#     with open(input_file_path, 'w') as f:
#         f.write(requests.get(data_url).text)
import pickle
import io
buf = io.StringIO()
'''
head -n 50000 CNencyclopedia.csv.clean| split -l 10000 - --numeric-suffixes CNencyclopedia.csv. --additional-suffix=.txt
'''
pth = '/root/ccm/models/MiniCPM-1B-sft-bf16/'

# pth = '/root/autodl-tmp/LLaMA-Factory/models/llama-3-8b-Instruct-chinese_v2/'
tokenizer = AutoTokenizer.from_pretrained(pth,local_files_only=True)


# data = io.
import collections

offsets = [((0,0),'START')]
buf = []
# offsets = collections.OrderedDict()
for fn in glob.glob( os.path.join(os.path.dirname(__file__), '*.txt')):
    print(fn)
    with open(fn, 'rb') as f:
        data = None
        buff = f.read()
        for encoding in 'gbk utf8'.split():
            try:
                data = buff.decode(encoding)
                break
            except Exception as e:
                print(e)            
        if data is None:
            print(f'[error]{fn}')
        else:
            print(f'[succ]{fn}')
            # print(type(data))
            data = tokenizer([data]).input_ids[0]
            data =  np.array(data, dtype=np.uint32)
            buf.append(data)
            # buf.write(data)
            (start,end),_ = offsets[-1]
            start = end
            end   = start+len(data)
            offsets.append(((start,end), fn ))
# buf.seek(0)
data = np.concatenate(buf,0)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]



# get all the unique characters that occur in this text
# chars = sorted(list(set(data)))
# vocab_size = len(chars)
# print("all the unique characters:", ''.join(chars[:10]))
# print(f"vocab size: {vocab_size:,}")

#   breakpoint()

# create a mapping from characters to integers
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return s
    # return tokenizer([s]).input_ids[0]
# def decode(l):
#     return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint32)
val_ids = np.array(val_ids, dtype=np.uint32)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': tokenizer.vocab_size,
    'tokenizer':pth,
    # 'itos': itos,
    # 'stoi': stoi,
    'offsets':offsets,
}
print(offsets)

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)