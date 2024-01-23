import os
import requests
import tiktoken
import numpy as np
import glob
# download the tiny shakespeare dataset
# for x in 
# input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# if not os.path.exists(input_file_path):
#     data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#     with open(input_file_path, 'w') as f:
#         f.write(requests.get(data_url).text)

import io
buf = io.StringIO()

# data = io.
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
            buf.write(data)
buf.seek(0)
data = buf.read()        
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
# enc = tiktoken.get_encoding("gpt3.5-turbo")
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
