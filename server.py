import gradio as gr

def greet(name, intensity):
    return "Hello " * intensity + name + "!"


import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import model as _model
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def load_model(out_dir, model_name,dataset):

    """
    Sample from a trained model
    """
    # -----------------------------------------------------------------------------
    model = model_name
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    # out_dir = 'out' # ignored if init_from is not 'resume'
    start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 10 # number of samples to draw
    max_new_tokens = 500 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    GPT = getattr(_model,model)

    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    meta = None
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta.get('stoi',{}), meta.get('itos',{})
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    # ModelEncoding()
    return Env(model,encode,decode,dataset,meta)


import numpy as np
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
seed_offset=0
random_seed=0

torch.manual_seed(1337 + seed_offset + random_seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device='cuda:0'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

import os



class Env(object):
    def __init__(self,model,encode,decode,dataset,meta):
        self.model = model
        self.encode = encode
        self.decode = decode
        self.train_data, self.val_data = Env.get_dataset(dataset)
        self.meta = meta
        pass

    @staticmethod
    def get_dataset(
        dataset
        ):
        
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # poor man's data loader
        data_dir   = os.path.join('data', dataset)
        if dataset=='novel_char3':
            train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint32, mode='r')
            val_data   = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint32, mode='r')
        else:
            train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
            val_data   = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        return train_data, val_data


    def get_batch(self, 
            # data,
            split='train',
            block_size = 64,
            batch_size = 128,
        ):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def get_batch_with_text(self, 
            # data,
            split='train',
            block_size = 64,
            batch_size = 128,
        ):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        # start,ends = self.meta['offsets']
        # starts, ends = self.meta['offsets']
        rg = np.stack( [rg for (rg,fn) in  self.meta['offsets']], 0)
        ix = ix.numpy()[:,None]
        val = (ix >= rg[None,:,0])&(ix < rg[None,:,1])
        val = val.argmax(1)
        fns = [xx[1] for xx in  self.meta['offsets'] ]
        text = np.vectorize(fns.__getitem__)(val)
        # print((x,y,text))


        return x, y, text

fn = 'out-novel-word_CCM305_db6def9feb4d60d03dab9fea4fd9c42a'
fn = 'out-novel-word_CCM310_def9c4134bbe0932436e65a80e9a1774'
# fn = 'out-novel-word_CCM305_67a09bbf19b8f7f0ea7606769e6ad989'
fn = 'out-novel-word_CCM310_48e86b8c07e86e54f7af3b4db2c4740a'
fn = 'out-shakespeare-char_CCM305_10de15b1217383000a3a7854b0ad98c1'
fn = 'out-shakespeare-char_CCM336_43b6e518488015cefb410c653f5cae7b'
fn = 'out-novel-char_CCM336_85ef6ca9f63692a41bf8e4d53047cccc'
fn = 'out-novel-char_GPT_85ef6ca9f63692a41bf8e4d53047cccc'

# # fn = 'out-novel-char_CCM337_85ef6ca9f63692a41bf8e4d53047cccc'
# fn = 'out-novel-char_CCM305D_13247fc8c05cfd87962f3a8426527446'
# fn = 'out-novel-char_CCM305D_42dd657983b1ca34c14ecd1004ef4f96' ### L4N12
# # fn = 'out-novel-char_CCM305D3_42dd657983b1ca34c14ecd1004ef4f96'
# # fn = 'out-novel-char_CCM305D2_42dd657983b1ca34c14ecd1004ef4f96'
# # fn = 'out-novel-char_CCM305D2_85ef6ca9f63692a41bf8e4d53047cccc' ### L4N12R3

fn = 'out-novel-char_CCM305F2_42dd657983b1ca34c14ecd1004ef4f96'
fn = 'out-novel-char_CCM305F2_f3847d36640b12339a12c1c8183b35a6'

# fn = 'out-novel-char_CCM305F5_42dd657983b1ca34c14ecd1004ef4f96'

# fn = 'out-novel-char_CCM305F2_4065df5236496f34a2c0e9ad2ddd7fd7'

# fn = 'out-novel-char_CCM305F6_42dd657983b1ca34c14ecd1004ef4f96'
fn = 'out-novel-char_CCM305F7_42dd657983b1ca34c14ecd1004ef4f96'
fn = 'out-novel-char_CCM305F8_42dd657983b1ca34c14ecd1004ef4f96'

fn = 'out-novel-char_CCM305F14_31540fb5d1e100861eed430368403542'

# fn = 'out-novel-char_CCM305F8C_42dd657983b1ca34c14ecd1004ef4f96'
# fn = 'out-novel-char_CCM305F12B_42dd657983b1ca34c14ecd1004ef4f96'  # L4N12
# fn = 'out-novel-char_CCM305F12B_5ec2aac12c167edc2e22aff144218bfd'  ### L4N2

# fn = 'out-novel-char_CCM305F8_ab144478feffa96eaeba34b256e83a07'

# fn = 'out-novel-char_CCM400_42dd657983b1ca34c14ecd1004ef4f96'
# fn = 'out-novel-char_CCM305F9_42dd657983b1ca34c14ecd1004ef4f96'
# fn = 'out-novel-char_CCM305F11_42dd657983b1ca34c14ecd1004ef4f96'
# fn = 'out-novel-char_CCM305F11_4065df5236496f34a2c0e9ad2ddd7fd7'
fn = 'out-novel-char_CCM401_4065df5236496f34a2c0e9ad2ddd7fd7'

fn = 'out-novel-char_GPT_4065df5236496f34a2c0e9ad2ddd7fd7'

fn = 'out-novel-char_GPT248N_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
fn = 'out-novel-char_LGT001B_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
fn = 'out-novel-char_LGT001_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
fn = 'out-novel-char_LGT001D_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

# fn = 'out-novel-char_LGT001E_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT001_5f42cea4d6cf5b579c5219fb2b448174'
# fn = 'out-novel-char_LGT020_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
fn = 'out-novel-char_LGT025_08ff8420ecdc9c6a69b756bb835bf92d'
# fn = 'out-novel-char_LGT013D_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
fn = 'out-novel-char_LGT012D_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

fn = 'out-novel-char_LGT012D_08ff8420ecdc9c6a69b756bb835bf92d'


# fn = 'out-novel-char_LGT019D_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

# # fn = 'out-novel-char_LGT017_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# # fn = 'out-novel-char_LGT017_08ff8420ecdc9c6a69b756bb835bf92d'
# # fn = 'out-novel-char_LGT017_12a79b3ea9cdc978105976c879ac0a18'
# fn = 'out-novel-char_LGT018_08ff8420ecdc9c6a69b756bb835bf92d'


# # fn = 'out-novel-char_LGT012_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

# # fn = 'out-novel-char_LGT001M_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

# # fn = 'out-novel-char_LGT012_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn ='out-novel-char_LGT012_12a79b3ea9cdc978105976c879ac0a18' # R3
# # fn = 'out-novel-char_LGT012_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

# fn = 'out-novel-char_LGT025D_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT026E_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT025_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT030_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT001D_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

# fn = 'out-novel-char_LGT025_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

# fn  ='out-novel-char_LGT012E_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn  ='out-novel-char_LGT023E_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

# fn = 'out-novel-char_LGT029_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
fn = 'out-novel-char_LGT029D_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT029E_29fd8aaa7d6a706d3fe1fe4edaa59f0d'




# fn = 'out-novel-char_LGT031_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT031F_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

# fn = 'out-novel-char_LGT031E_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

# fn = 'out-novel-char_LGT029E_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# # fn = 'out-novel-char_LGT024_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT031_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT029E_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT031_29fd8aaa7d6a706d3fe1fe4edaa59f0d'

fn = 'out-novel-char_LGT029E_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
fn = 'out-novel-char_LGT029_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
fn = 'out-novel-char_LGT029D_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
fn = 'out-novel-char_LGT310F_73f5a42d89fb7da6ccfabd14d280c51b'
# fn = 'out-novel-char_LGT314F_73f5a42d89fb7da6ccfabd14d280c51b'
fn = 'out-novel-char_LGT310G_73f5a42d89fb7da6ccfabd14d280c51b'
fn = 'out-novel-char_LGT310G_2fa86314f213af3c914f2720f514e0f8'
# fn = 'out-novel-char_LGT310H_73f5a42d89fb7da6ccfabd14d280c51b'
fn = 'out-novel-char_LGT310J_73f5a42d89fb7da6ccfabd14d280c51b'
fn = 'out-novel-char_LGT314J_73f5a42d89fb7da6ccfabd14d280c51b'
# fn = 'out-novel-char_LGT314H_73f5a42d89fb7da6ccfabd14d280c51b'

# fn = 'out-novel-char_LGT552A_7a6ec853557c12d9f4715b5701f204e5'
# fn = 'out-novel-char_LGT553A_7a6ec853557c12d9f4715b5701f204e5'
# fn = 'out-novel-char_LGT548A_7a6ec853557c12d9f4715b5701f204e5'
# fn = 'out-novel-char_LGT552F_7a6ec853557c12d9f4715b5701f204e5'
# fn = 'out-novel-char_LGT552A_7a6ec853557c12d9f4715b5701f204e5'
# fn = 'out-novel-char_LGT552F_7a6ec853557c12d9f4715b5701f204e5'
fn = 'out-novel-char_LGT553A_7a6ec853557c12d9f4715b5701f204e5'
fn = 'out-novel-char_LGT553A_4d06c3ce0eb6b62233f4128fdb1c3b0f'  # R3 from scratch
fn = 'out-novel-char_LGT553A_7c359afca8dc913e1687eceea7f47c9e' ## R3 L12

fn = 'out-novel-char_LGT553F_4d06c3ce0eb6b62233f4128fdb1c3b0f'  # R3 from scratch
# fn = 'out-novel-char_LGT556A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT557A_7a6ec853557c12d9f4715b5701f204e5'

# fn = 'out-novel-char_LGT558A_7a6ec853557c12d9f4715b5701f204e5'

# fn = 'out-novel-char_LGT559A_4d06c3ce0eb6b62233f4128fdb1c3b0f'

# fn = 'out-novel-char_LGT555A_7a6ec853557c12d9f4715b5701f204e5'

fn = 'out-novel-char_LGT560A_7a6ec853557c12d9f4715b5701f204e5'
fn = 'out-novel-char_LGT560A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT561A_7a6ec853557c12d9f4715b5701f204e5'
# fn = 'out-novel-char_LGT553A_7c359afca8dc913e1687eceea7f47c9e' ## R3 L12
# fn = 'out-novel-char_LGT561K_7a6ec853557c12d9f4715b5701f204e5'
fn = 'out-novel-char_LGT561A_5fae2c6bfe0505ac9273df162b177fad'
# fn = 'out-novel-char_LGT552A_41a655002a3dcb619938fb5b47d4e72b'
fn = 'out-novel-char_LGT561A_947abccd7833bcfe98f0d5dc50ccc0e2'  ### R3 L2
fn = 'out-novel-char_LGT561B_15cd9352c323b2b6561286d1eadc9e7d'
# fn = 'out-novel-char_LGT552A_dc5c0a8265563a6f78880aee25541230'  ### 552A L2 N128
# fn = 'out-novel-char_LGT561B_21e30874783a89dde189b5002656ae22'
fn = 'out-novel-char_LGT561A_8aac12a7de59cbfba2778db270232960'  ### L4
fn = 'out-novel-char_LGT561B_21e30874783a89dde189b5002656ae22'  ### L4
# fn = 'out-novel-char_LGT552A_4f928a3dadba95848df868aeb507a20a'  ### L4
fn = 'out-novel-char_LGT555A_4f928a3dadba95848df868aeb507a20a'

# fn = 'out-novel-char_LGT553F_4d06c3ce0eb6b62233f4128fdb1c3b0f'  # R3 from scratch

# fn = 'out-novel-char_LGT553F_7a6ec853557c12d9f4715b5701f204e5' ### R2 migrated
# fn = 'out-novel-char_LGT561A_5fae2c6bfe0505ac9273df162b177fad'
# fn = 'out-novel-char_LGT552A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT553A_7c359afca8dc913e1687eceea7f47c9e' ## R3 L12
# fn = 'out-novel-char_LGT561A_5fae2c6bfe0505ac9273df162b177fad' ### 
# fn = 'out-novel-char_LGT561A_4d06c3ce0eb6b62233f4128fdb1c3b0f'  # R3 from scratch
# fn = 'out-novel-char_LGT570A_4f928a3dadba95848df868aeb507a20a'


# fn = 'out-novel-char_LGT553A_7c359afca8dc913e1687eceea7f47c9e' ## R3 L12


# fn = 'out-novel-char_LGT555A_7a6ec853557c12d9f4715b5701f204e5'
# fn = 'out-novel-char_LGT552A_7a6ec853557c12d9f4715b5701f204e5'
# fn = 'out-novel-char_LGT562A_4d06c3ce0eb6b62233f4128fdb1c3b0f'

# fn = 'out-novel-char_LGT561A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# fn = 'out-novel-char_LGT561A_4d06c3ce0eb6b62233f4128fdb1c3b0f'

# fn = 'out-novel-char_LGT554A_7a6ec853557c12d9f4715b5701f204e5' ## 554A R2
# fn = 'out-novel-char_LGT554A_4d06c3ce0eb6b62233f4128fdb1c3b0f' ### R3

# fn = 'out-novel-char_LGT527A_428f066b7c090666c9438da8000042c6'
# fn = 'out-novel-char_LGT552A_4d06c3ce0eb6b62233f4128fdb1c3b0f'  # R3 from scratch
# fn = 'out-novel-char_LGT553F_7a6ec853557c12d9f4715b5701f204e5'
fn = 'out-novel-char_LGT562A1_5fae2c6bfe0505ac9273df162b177fad'
fn = 'out-novel-char_LGT571A_7a6ec853557c12d9f4715b5701f204e5'
fn = 'out-novel-char_LGT555A_7a6ec853557c12d9f4715b5701f204e5'
fn = 'out-novel-char_LGT553A_7c359afca8dc913e1687eceea7f47c9e' ## R3 L12
# fn = 'out-novel-char_LGT561A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT561A_7a6ec853557c12d9f4715b5701f204e5' ###  R2 migrated
fn = 'out-novel-char_LGT553A_83f27233ce39d2b46b92fe7ca69ea64d' ### L16
# fn = 'out-novel-char_LGT561A_4d06c3ce0eb6b62233f4128fdb1c3b0f'  # R3 from scratch
fn = 'out-novel-char_LGT553A_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT561A_83f27233ce39d2b46b92fe7ca69ea64d'


fn = 'out-novel-char_LGT555A_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT555A_7a6ec853557c12d9f4715b5701f204e5'

# fn = 'out-novel-char_LGT552A_4d06c3ce0eb6b62233f4128fdb1c3b0f'  # R3 from scratch
# fn = 'out-novel-char_LGT542A_7a6ec853557c12d9f4715b5701f204e5' ### R2
# fn = 'out-novel-char_LGT542A_83f27233ce39d2b46b92fe7ca69ea64d' ### L16
fn = 'out-novel-char_LGT553A_83f27233ce39d2b46b92fe7ca69ea64d' ### L16

# fn = 'out-novel-char_LGT571A_7a6ec853557c12d9f4715b5701f204e5' 
# fn = 'out-novel-char_LGT561A_83f27233ce39d2b46b92fe7ca69ea64d'  ### L16, migrated from 553A
# fn = 'out-novel-char_LGT561A_4d06c3ce0eb6b62233f4128fdb1c3b0f'  ### L8R3 scratch

fn = 'out-novel-char_LGT563A_83f27233ce39d2b46b92fe7ca69ea64d' ### L22 R3 from scratch
# fn = 'out-novel-char_LGT552A_4d06c3ce0eb6b62233f4128fdb1c3b0f'  # R3 from scratch
# fn = 'out-novel-char_LGT552A_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT564A_4d06c3ce0eb6b62233f4128fdb1c3b0f'


# fn = 'out-novel-char_LGT563B_83f27233ce39d2b46b92fe7ca69ea64d' ### L16 migrated from 553A
# fn = 'out-novel-char_LGT563B_4d06c3ce0eb6b62233f4128fdb1c3b0f' ### L8 scratch

fn = 'out-novel-char_LGT555A_83f27233ce39d2b46b92fe7ca69ea64d'

# fn = 'out-novel-char_LGT553F_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# fn = 'out-novel-char_LGT561A_bd72b96868c73c1f2402958e595130ea' ### L10 R3 from scratch
# fn = 'out-novel-char_LGT553A_4d06c3ce0eb6b62233f4128fdb1c3b0f' ### L8 R3
# fn = 'out-novel-char_LGT553A_7c359afca8dc913e1687eceea7f47c9e'
fn = 'out-novel-char_LGT565A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT563B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT555C_83f27233ce39d2b46b92fe7ca69ea64d'

# fn = 'out-novel-char_LGT555A_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT555B_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT566A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT567A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT568A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT563C_a972333150a33e840a44365c81c7a3ac'
# fn = 'out-novel-char_LGT563F_fe07543e050de1801fdd5c398fec46bd'
# fn = 'out-novel-char_LGT563A_fe07543e050de1801fdd5c398fec46bd'
fn = 'out-novel-char_LGT563A_fe07543e050de1801fdd5c398fec46bd' ### L16 R4 T3000
# fn = 'out-novel-char_LGT563A_a972333150a33e840a44365c81c7a3ac' ### L26 R3
# fn = 'out-novel-char_LGT563A_a972333150a33e840a44365c81c7a3ac_T2300'
fn = 'out-novel-char_LGT563F8_a972333150a33e840a44365c81c7a3ac' ### L26 R3 step 1700
# fn = 'out-novel-char_LGT563F8_83f27233ce39d2b46b92fe7ca69ea64d' ### L16 R3

# fn = 'out-novel-char_LGT569A_a972333150a33e840a44365c81c7a3ac' ### L26 R3
# fn = 'out-novel-char_LGT563F12_70b087c7927ad0f50300b406e1004f2b' ### L36 L12

# fn = 'out-novel-char_LGT569A_a972333150a33e840a44365c81c7a3ac_T1700' ## L26 R3 scratch T1700
fn = 'out-novel-char_LGT569A_a972333150a33e840a44365c81c7a3ac' ## L26 R3 scratch T2300
fn = 'out-novel-char_LGT563F_a972333150a33e840a44365c81c7a3ac' ### L26 T2300

# fn = 'out-novel-char_LGT563B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# fn = 'out-novel-char_LGT563B_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT563B_67aac987f309ce7e0077c792a1089509' ### L26 scratch
# fn = 'out-novel-char_LGT555C_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT572F_a972333150a33e840a44365c81c7a3ac'
fn = 'out-novel-char_LGT574A_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT575A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT576A_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT576C_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT577A_83f27233ce39d2b46b92fe7ca69ea64d'

fn = 'out-novel-char_LGT577A_a972333150a33e840a44365c81c7a3ac'
fn = 'out-novel-char_LGT577E_a972333150a33e840a44365c81c7a3ac'
fn = 'out-novel-char_LGT578F_a972333150a33e840a44365c81c7a3ac'
fn = 'out-novel-char_LGT577F_a972333150a33e840a44365c81c7a3ac'
# fn = 'out-novel-char_LGT577E_a972333150a33e840a44365c81c7a3ac'
fn = 'out-novel-char_LGT577F2_83f27233ce39d2b46b92fe7ca69ea64d'

fn = 'out-novel-char_LGT577B_a972333150a33e840a44365c81c7a3ac'


# fn = 'out-novel-char_LGT577A_a972333150a33e840a44365c81c7a3ac'
# fn = 'out-novel-char_LGT579B_a972333150a33e840a44365c81c7a3ac'
# fn = 'out-novel-char_LGT579C_a972333150a33e840a44365c81c7a3ac'
fn = 'out-novel-char_LGT580A_a972333150a33e840a44365c81c7a3ac'
# fn  = 'out-novel-char_LGT555A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT555C_a972333150a33e840a44365c81c7a3ac'
# fn = 'out-novel-char_LGT581A_a972333150a33e840a44365c81c7a3ac'
# fn = 'out-novel-char_LGT583A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT582B_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT581A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT585A_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT586A_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT585B_83f27233ce39d2b46b92fe7ca69ea64d'

# fn = 'out-novel-char_LGT555A_a972333150a33e840a44365c81c7a3ac' ## L26 T6350
# fn = 'out-novel-char_LGT582A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT582B_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT584A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT582B_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT583A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT587A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT589A_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT588A_fe07543e050de1801fdd5c398fec46bd'
# fn = 'out-novel-char_LGT588A_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT590B_fe07543e050de1801fdd5c398fec46bd'

# fn = 'out-novel-char_LGT591A_83f27233ce39d2b46b92fe7ca69ea64d'
# # fn = 'out-novel-char_LGT592A_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT591B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# # fn = 'out-novel-char_LGT591A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT593B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# fn = 'out-novel-char_LGT590B_fe07543e050de1801fdd5c398fec46bd'


# # fn = 'out-novel-char_LGT582C_83f27233ce39d2b46b92fe7ca69ea64d'
# fn = 'out-novel-char_LGT582A_a972333150a33e840a44365c81c7a3ac'
fn = 'out-novel-char_LGT594A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# fn = 'out-novel-char_LGT594B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# fn = 'out-novel-char_LGT595B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT596A_41a655002a3dcb619938fb5b47d4e72b'
fn = 'out-novel-char_LGT597A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT595A_4d06c3ce0eb6b62233f4128fdb1c3b0f'

# fn = 'out-novel-char_LGT595A_41a655002a3dcb619938fb5b47d4e72b'
# fn = 'out-novel-char_LGT590A_83f27233ce39d2b46b92fe7ca69ea64d'
fn = 'out-novel-char_LGT598A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# fn = 'out-novel-char_LGT598B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# fn = 'out-novel-char_LGT598A_41a655002a3dcb619938fb5b47d4e72b'
# fn = 'out-novel-char_LGT599A_41a655002a3dcb619938fb5b47d4e72b'
fn = 'out-novel-char_LGT600A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT601A_41a655002a3dcb619938fb5b47d4e72b'
fn = 'out-novel-char_LGT599A_2eba875496dbc592fb7121ac81bd28c0'
fn = 'out-novel-char_LGT599A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# fn = 'out-novel-char_LGT599A_fe07543e050de1801fdd5c398fec46bd'
# fn = 'out-novel-char_LGT603A_41a655002a3dcb619938fb5b47d4e72b'
# fn = 'out-novel-char_LGT599A_fe07543e050de1801fdd5c398fec46bd'
# fn = 'out-novel-char_LGT603B_4d06c3ce0eb6b62233f4128fdb1c3b0f'

fn = 'out-novel-char_LGT603A_a972333150a33e840a44365c81c7a3ac'
fn = 'out-novel-char_LGT603A_41a655002a3dcb619938fb5b47d4e72b'

# fn = 'out-novel-char_LGT606A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# fn = 'out-novel-char_LGT599A_41a655002a3dcb619938fb5b47d4e72b'
fn = 'out-novel-char_LGT607A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT599B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT599A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT608A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT603A_658caab9f5066390a26e1d487cf8757e'
fn = 'out-novel-char_LGT603A_a972333150a33e840a44365c81c7a3ac' ## L26

# fn = 'out-novel-char_LGT599A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# fn = 'out-novel-char_LGT599A_fe07543e050de1801fdd5c398fec46bd'

# fn = 'out-novel-char_LGT599A_41a655002a3dcb619938fb5b47d4e72b'
# fn = 'out-novel-char_LGT606B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
fn = 'out-novel-char_LGT602A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'


fn = 'out-novel-char2_LGT603A_d9b15f0c0f74c9b6e1396050c83e2f3c'
dataset = 'novel_char2'

fn = 'out-novel-char2_LGT603A_a972333150a33e840a44365c81c7a3ac'
dataset = 'novel_char2'

fn = 'out-novel-char2_LGT610A_83f27233ce39d2b46b92fe7ca69ea64d'

# fn = 'out-novel-char2_LGT610A_83f27233ce39d2b46b92fe7ca69ea64d'
# dataset = 'novel_char2'


fn = 'out-novel-char_LGT607A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'


# fn = 'out-novel-char2_LGT611A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char2'


fn = 'out-novel-char2_LGT603A_d9b15f0c0f74c9b6e1396050c83e2f3c'  ### L8
dataset = 'novel_char2'

# fn ='out-novel-char2_LGT603C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char2'
fn = 'out-novel-char2_LGT603A_a972333150a33e840a44365c81c7a3ac'
dataset = 'novel_char2'

# fn = 'out-novel-char2_LGT603B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char2'

# fn = 'out-novel-char_LGT606A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char2_LGT612A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char2'

fn = 'out-novel-char_LGT603D_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

# fn = 'out-novel-char_LGT603B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT603E_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT614A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'


# fn = 'out-novel-char_LGT603A2_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT616A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'


# fn = 'out-novel-char_LGT615A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'


fn = 'out-novel-char_LGT617A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT618A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'


# fn = 'out-novel-char_LGT619A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT620A_41a655002a3dcb619938fb5b47d4e72b'
dataset = 'novel_char'



# fn = 'out-novel-char_LGT621A_41a655002a3dcb619938fb5b47d4e72b_T6200'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT622A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT621A_41a655002a3dcb619938fb5b47d4e72b_T3200'
# dataset = 'novel_char'


fn = 'out-novel-char_LGT617A_4d06c3ce0eb6b62233f4128fdb1c3b0f_T6200'
dataset = 'novel_char'



# fn = 'out-novel-char_LGT624A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT625A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT626A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

# fn = 'out-novel-char_LGT627A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT617A_83f27233ce39d2b46b92fe7ca69ea64d'  ### L16
# dataset = 'novel_char'


fn = 'out-novel-char_LGT617A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'


# fn = 'out-novel-char_LGT628A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT626A_83f27233ce39d2b46b92fe7ca69ea64d' ### L16 ## bad
# dataset = 'novel_char'

fn = 'out-novel-char_LGT629A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'


# fn = 'out-novel-char_LGT630A_83f27233ce39d2b46b92fe7ca69ea64d'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT623A_4d06c3ce0eb6b62233f4128fdb1c3b0f' ### L8
# dataset = 'novel_char'

fn = 'out-novel-char_LGT623A_83f27233ce39d2b46b92fe7ca69ea64d' ### L16
dataset = 'novel_char'


#fn = 'out-novel-char_LGT629A_83f27233ce39d2b46b92fe7ca69ea64d_T3000'
dataset = 'novel_char'

fn = 'out-novel-char_LGT629A_83f27233ce39d2b46b92fe7ca69ea64d_T3300' ### L16 R3
dataset = 'novel_char'


fn = 'out-novel-char_LGT629A_83f27233ce39d2b46b92fe7ca69ea64d' ### L16 R3
dataset = 'novel_char'
### 2500 2.19
### 3000 2.24
### 3300 2.67
### 3500 3.24
### 3950 2.42
### 4000 2.82

fn = 'out-novel-char_LGT623A_fe07543e050de1801fdd5c398fec46bd' ### L16 R4
dataset = 'novel_char'

### 3000 2.98
### 3300 3.31
### 3600 4.23

fn = 'out-novel-char_LGT631A_83f27233ce39d2b46b92fe7ca69ea64d'
dataset = 'novel_char'

fn = 'out-novel-char_LGT632A_83f27233ce39d2b46b92fe7ca69ea64d'
dataset = 'novel_char'

fn = 'out-novel-char_LGT623A_83f27233ce39d2b46b92fe7ca69ea64d'
dataset = 'novel_char'

# fn = 'out-novel-char_LGT632A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT633A_4d06c3ce0eb6b62233f4128fdb1c3b0f'  ### L8 R3/
dataset = 'novel_char'

# fn = 'out-novel-char_LGT634A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT633A_41a655002a3dcb619938fb5b47d4e72b'  ### L8 R4
dataset = 'novel_char'

# fn = 'out-novel-char_LGT633A_41a655002a3dcb619938fb5b47d4e72b'   ### L8
# dataset = 'novel_char'

fn = 'out-novel-char_LGT633A_83f27233ce39d2b46b92fe7ca69ea64d'
dataset = 'novel_char'

fn = 'out-novel-char_LGT635A_4d06c3ce0eb6b62233f4128fdb1c3b0f'  ### L8
dataset = 'novel_char'

fn = 'out-novel-char_LGT635B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT636A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'


# fn = 'out-novel-char_LGT636C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT637A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT638B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

# fn = 'out-novel-char_LGT638C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT636B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'


# fn = 'out-novel-char_LGT637A2_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT637A_83f27233ce39d2b46b92fe7ca69ea64d' ### L16
# dataset = 'novel_char'

fn = 'out-novel-char_LGT637B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT637C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT640C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT637C2_83f27233ce39d2b46b92fe7ca69ea64d'
dataset = 'novel_char'

# fn = 'out-novel-char_LGT637C2_83f27233ce39d2b46b92fe7ca69ea64d'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT637A_4d06c3ce0eb6b62233f4128fdb1c3b0f' ### L8
# dataset = 'novel_char'


# fn = 'out-novel-char_LGT637E_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT637C_83f27233ce39d2b46b92fe7ca69ea64d' ### L16
# dataset = 'novel_char'

fn = 'out-novel-char_LGT641A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT642A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'


# fn = 'out-novel-char_LGT651A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT650A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT650A_83f27233ce39d2b46b92fe7ca69ea64d' ## L16
dataset = 'novel_char'


# # fn = 'out-novel-char_LGT652A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# # dataset = 'novel_char'



# fn = 'out-novel-char_LGT653A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'


# fn = 'out-novel-char_LGT654A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT654B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT652B_83f27233ce39d2b46b92fe7ca69ea64d' ### L16
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT652B_4d06c3ce0eb6b62233f4128fdb1c3b0f' ### L8
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT653B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'


# fn = 'out-novel-char_LGT642C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT652B_83f27233ce39d2b46b92fe7ca69ea64d'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT655C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT655C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT657D_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT656D_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT650C2_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'



# fn = 'out-novel-char_LGT658A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# # fn = 'out-novel-char_LGT659A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# # dataset = 'novel_char'

# fn = 'out-novel-char_LGT659B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT660C_4d06c3ce0eb6b62233f4128fdb1c3b0f'

fn = 'out-novel-char_LGT670A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT670A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT662A_83f27233ce39d2b46b92fe7ca69ea64d'
dataset = 'novel_char'

fn = 'out-novel-char_LGT662A2_7c359afca8dc913e1687eceea7f47c9e'
dataset = 'novel_char'

fn = 'out-novel-char_LGT662A2_b3c72374e5dee2107a4e7fe9fbf01b28'
dataset = 'novel_char'

fn = 'out-novel-char_LGT662A2_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT673A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT673A_83f27233ce39d2b46b92fe7ca69ea64d'
dataset = 'novel_char'

# fn = 'out-novel-char_LGT674A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT673A_e06c6afd6e1519333f528472cf20aa7b'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT676A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT660C_83f27233ce39d2b46b92fe7ca69ea64d'
dataset = 'novel_char'

fn = 'out-novel-char_LGT660C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

# fn = 'out-novel-char_LGT658B_83f27233ce39d2b46b92fe7ca69ea64d' ## l16
# dataset = 'novel_char'

fn = 'out-novel-char_LGT658C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

# fn = 'out-novel-char_LGT650C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'


# fn = 'out-novel-char_LGT658B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT661B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT660C_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT680B_7c359afca8dc913e1687eceea7f47c9e'
dataset = 'novel_char'

# fn = 'out-novel-char_LGT681B_7c359afca8dc913e1687eceea7f47c9e'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT680B_83f27233ce39d2b46b92fe7ca69ea64d'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT682B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT682B_7c359afca8dc913e1687eceea7f47c9e'
dataset = 'novel_char'

fn = 'out-novel-char_LGT683B_7c359afca8dc913e1687eceea7f47c9e'
dataset = 'novel_char'

fn = 'out-novel-char_LGT683B_7c359afca8dc913e1687eceea7f47c9e'
dataset = 'novel_char'

fn = 'out-novel-char_LGT683B_4d06c3ce0eb6b62233f4128fdb1c3b0f' ## L8
dataset = 'novel_char'

fn = 'out-novel-char_LGT683B_e06c6afd6e1519333f528472cf20aa7b' ### L24
dataset = 'novel_char'
# 

fn = 'out-novel-char_LGT683B_992fe55c3643a7c71d3df6a662d744b6' ### L32
dataset = 'novel_char'

# fn = 'out-novel-char_LGT687B_7c359afca8dc913e1687eceea7f47c9e'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT683C_7c359afca8dc913e1687eceea7f47c9e'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT684B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT690D_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT691B_7c359afca8dc913e1687eceea7f47c9e' ### L12 R3
dataset = 'novel_char'


fn = 'out-novel-char_LGT700B_7c359afca8dc913e1687eceea7f47c9e'
dataset = 'novel_char'

fn = 'out-novel-char_LGT701B_7c359afca8dc913e1687eceea7f47c9e'
dataset = 'novel_char'


# fn = 'out-novel-char_LGT702B_7c359afca8dc913e1687eceea7f47c9e'
# dataset = 'novel_char'


# fn = 'out-novel-char_LGT694B_7c359afca8dc913e1687eceea7f47c9e'
# dataset = 'novel_char'




# fn = 'out-novel-char_LGT705B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT712B_7c359afca8dc913e1687eceea7f47c9e' ### L12
dataset = 'novel_char'

# fn = 'out-novel-char_LGT711B_4d06c3ce0eb6b62233f4128fdb1c3b0f' ### L8
# dataset = 'novel_char'

fn = 'out-novel-char_LGT711B_7c359afca8dc913e1687eceea7f47c9e' ### L12
dataset = 'novel_char'

# fn = 'out-novel-char_LGT711B_41a655002a3dcb619938fb5b47d4e72b' ### L8 R4
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT711B_e06c6afd6e1519333f528472cf20aa7b' ### L24
# dataset = 'novel_char'

fn = 'out-novel-char_LGT713B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT716B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT716B_7c359afca8dc913e1687eceea7f47c9e'
dataset = 'novel_char' 

fn = 'out-novel-char_LGT716B_7c359afca8dc913e1687eceea7f47c9e'
dataset = 'novel_char'


# fn = 'out-novel-char_LGT711B_41a655002a3dcb619938fb5b47d4e72b'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT717B_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT716B_41a655002a3dcb619938fb5b47d4e72b'
dataset = 'novel_char'

fn = 'out-novel-char_LGT717B_41a655002a3dcb619938fb5b47d4e72b'
dataset = 'novel_char'

fn = 'out-novel-char_LGT718D_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'


# fn = 'out-novel-char_LGT718D_7c359afca8dc913e1687eceea7f47c9e'  ### L12
# dataset = 'novel_char'


# fn = 'out-novel-char_LGT718E_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'



fn = 'out-novel-char_LGT718C_4d06c3ce0eb6b62233f4128fdb1c3b0f'  ### L8
dataset = 'novel_char'

fn = 'out-novel-char_LGT720A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

# fn = 'out-novel-char_LGT720F_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT720G_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'



# fn = 'out-novel-char_LGT720D_e06c6afd6e1519333f528472cf20aa7b'
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT720J_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'


fn = 'out-novel-char_LGT720K_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT720M_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT720N_41a655002a3dcb619938fb5b47d4e72b'
dataset = 'novel_char'

fn = 'out-novel-char_LGT723H_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT724H_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'





fn = 'out-novel-char_LGT720D_4d06c3ce0eb6b62233f4128fdb1c3b0f' ### L8
dataset = 'novel_char'


fn = 'out-novel-char_LGT720H_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT720H_41a655002a3dcb619938fb5b47d4e72b' ## L8 R4
dataset = 'novel_char'

fn = 'out-novel-char_LGT720H_7c359afca8dc913e1687eceea7f47c9e' ### L12
dataset = 'novel_char'


fn = 'out-novel-char_LGT723H_41a655002a3dcb619938fb5b47d4e72b' ## L8 R4
dataset = 'novel_char'

fn = 'out-novel-char_LGT725H_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

# fn = 'out-novel-char_LGT720H_83f27233ce39d2b46b92fe7ca69ea64d' ### L16
# dataset = 'novel_char'

# fn = 'out-novel-char_LGT722A_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char'

fn = 'out-novel-char_LGT725D_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT728D_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char'

fn = 'out-novel-char_LGT728D_41a655002a3dcb619938fb5b47d4e72b'
dataset = 'novel_char'

fn = 'out-novel-char3_LGT800D_4d06c3ce0eb6b62233f4128fdb1c3b0f'
dataset = 'novel_char3'

# fn = 'out-novel-char3_LGT723D_4d06c3ce0eb6b62233f4128fdb1c3b0f'
# dataset = 'novel_char3'

# fn = 'out-novel-char3_LGT723D_3d25ede0cfd080d9e35756b051dcef65'
# dataset = 'novel_char3'

fn = 'out-novel-char3_LGT723E_3d25ede0cfd080d9e35756b051dcef65'
dataset = 'novel_char3'

fn = 'out-novel-char3_LGT723F_3d25ede0cfd080d9e35756b051dcef65'
dataset = 'novel_char3'

# fn = 'out-novel-char_LGT723H_41a655002a3dcb619938fb5b47d4e72b'
# dataset = 'novel_char'
fn = "out-novel-char3_LGT740F_3d25ede0cfd080d9e35756b051dcef65"
dataset = 'novel_char3'

fn = 'out-novel-char3_LGT740F_b13991dc3a1d92fb6590a4db51c60170'
dataset = 'novel_char3'

fn = 'out-novel-char3_LGT740F_d34f65156182c545ea648093d502a95d'
dataset = 'novel_char3'


config =dict(server_port=6006,server_name="0.0.0.0")


'''

场，但至少在眼下，宫里已经坐实了自己谋杀陛下的谋逆大罪，自己已经成为了人人得而诛之的恶贼。

　　可他没有一丝畏惧，也没有受

落，这名手臂上插着匕首的年轻人取出了一封黄油皮信笺，恭敬的放在身落，这名手臂上插着匕首的年轻人取出了一封黄油皮信笺，恭敬的放在身

### 3300 1.27 629A
### 3600 1.92 629A
山盘古脉本是巫邪祭死之地，玉城中是藏纳祭器之地，而封师古又把此山山盘古脉本是巫邪祭死之地，玉城中是藏纳祭器之地，而封师古又把此山

现即隐，沉默而悲伤地从雪地里抬起那具尸体，踉跄着走进了神庙之中，那尸体上穿着一件人间常见的布衣。


　　第七十三章 范府的变化

现即隐，沉默而悲伤地从雪地里抬起那具尸体，踉跄着走进了庙中，那尸体上穿着一件人间常见的布衣。


　　第七十三章 范府的变
 
么说，这片四维空间对于你，或者说对于你的建造者，是类似于海洋的东么说，这片四维空间对于你，或者说对于你的建造者，是类似于海洋的东

可依的救治，长孙无疆已经到了五内俱衰的地步，在这种情形下，唯有先可依的救治，长孙无疆已经到了五内俱衰的地步，在这种情形下，唯有先

1998年　抗洪抢险电视画面中捕捉到球状闪电 
2001年  
1998年　抗洪抢险电视画面中捕捉到球状闪电 
2001年   

艰难无比地开口说道：“官船上岛的时候，正是黎明前的那一刻，岛周礁多，那么黑的天光下，能够强行登岛，应该是专业的水师，而不是借船的

以成匹夫之勇，安得为义？其罪三也。兄有此三罪，弟不得不告。”
公沉吟曰：“汝说我有三罪，欲我如何？”辽曰：“今四面皆曹公之兵，其

却说王濬班师，迁吴主皓赴洛阳面君。皓登殿稽首以见晋帝。帝赐坐曰：“朕设此座以待卿久矣。”皓对曰：“臣于南方亦设此座以待陛下。”帝大笑。贾充问皓曰：“闻君在南方，每凿人眼目，剥人面皮，此何等刑耶？”皓曰：“人臣弑君及奸回不忠者，则加此刑耳。”充默然甚愧。帝封皓为归命侯，子孙封中郎，随降宰辅皆封列侯。丞相张悌阵亡，封其子孙。封王濬为辅国大将军，其馀各加封赏。

　　“差不多了。”忽然，他的表情又严肃认真了起来，拉起了自己的长衫下摆，做了一个兜，全神贯注的抬头看着头顶的牌楼。
　　马车轮在碎石地上轻微作响，鹿林镇外一片黄杨小树林里几个嬉闹的孩童停了下来，好奇地看着这辆马车走出了鹿林镇，爬上了前面一个小土坡，最终消失在他们好奇的视线之中。
　　“原来这个世界，真是有高手的……”林夕站在风调雨顺牌楼下面，这个在鹿林镇有些名气的林二少爷此刻的神情有些古怪。他一副若有所思的神情，不时不自觉的摸着自己的额头，好像那里有一个包一样。

第一章 一路向北
　　鹿东陵刮了一场大风。
　　这年的雨水比往年少，所以即便在大风过后，这个偏安一隅的边城的天空还是有些发灰。
　　一辆刚刚换了新轮的普通马车出了鹿东陵城西的客栈，花了半天的时间，穿过了整个鹿东陵，出了东城门，又继续慢慢的往东而行，消失在了城门上持戈而立的兵卒的视线之中。
　　除了驾车的是一名十五六岁的清秀少女让兵卒有些一时的惊讶之外，这辆继续往东的马车并没有引起任何人更多的注意。
　　城中的鹿东陵府就像一个内城，用烧制的土砖建造的外墙高达五丈，陵府官员办公的府邸只占了朝北的三分之一建筑，其余都是兵营和操练场。
　　这并没有什么特色，自云秦帝国建国以来，所有的陵府除了占地大小要省府确定之外，所有的样式，却都是这样的格局。
　　此刻鹿东陵府北部正中的陵督府里，正点着数根红色的巨烛。
　　这几根红色巨烛驱走了这个铺着青色石板的幽深大厅的最后一丝阴暗，但是摇曳的烛光映射在李西平的脸上，却是恰如其分的昭示了他此刻摇曳不定的心情。


　　“差不多了。”忽然，他的表情又严肃认真了起来，全神贯注的抬头看着头顶的牌楼。
第一章 一路向北
　　鹿东陵刮了一场大风。
　　这年的雨水比往年少，所以即便在大风过后，这个偏安一隅的边城的天空还是有些发灰。


林夕平静地说道：“那你是谁的死士？”
　　薛万涛看了林夕一眼，道：“这你不必要知道。”
　　林夕道：“说实话我不喜欢被人砍”
　　薛万涛眼神一冷，拔出手中的长枪刺向林夕的胸口。此刻鹿东陵府北部正中的陵督府里，正点着数根红色的巨烛。


当然可以。”方池末马上接过了后方主动递上的一份军图，递给了林夕。
　　“谢谢你们的慷慨。”林夕躬身行礼，和云秦军方大举收缩之后

'''
mname = fn.split('_')[1]
model = load_model(fn,mname,dataset)
print(f'[loaded_model]{fn}')




def add_sample_interface(env):
    '''
    Adding a sampling interface for the model
    
    '''
    xvar = gr.State([])
    with gr.Row() as row:
        with gr.Column():
            model_name = gr.Textbox(label='model_name', value=env.model.__class__.__name__)
            c_temp = gr.Number(label='temperature',value=0.8)
            c_topk = gr.Number(label='topk',value=200,precision=0)
            c_max_token = gr.Number(label='max token',value=200,precision=0)
        with gr.Column():
            input_text = gr.Textbox(
                label="input_text ( model.text_generate )",
                value="(input text here)"
            )
            output_text = gr.Textbox(
                label="output_text",
                value="(input text here)"
            )
            btn = gr.Button("generate")

    def generate(input_text, c_temp, c_topk, c_max_token, var):
        enc_text = env.encode(input_text)
        ret = env.model.text_generate(enc_text, temperature=c_temp, top_k=(c_topk), max_new_tokens= (c_max_token))
        ret = env.decode(ret)
        return ret


    btn.click(
        generate, 
        [input_text,c_temp,c_topk, c_max_token, xvar],
        output_text
        )
    return


import plotly.graph_objects as go
import numpy as np
# fig.show()
def add_analyse_interface(env):
    '''
    Adding a sampling interface for the model
    
    '''
    model = env.model
    xvar = gr.State([])

    with gr.Row() as row:
        with gr.Column():
            model_name = gr.Textbox(label='model_name', value=model.__class__.__name__)
            input_text = gr.Textbox(
                label="input_text",
                value="(input text here)"
            )

            output_text = gr.Textbox(
                label="output_text",
                value="(input text here)"
            )
            btn = gr.Button("analyse")
            


    with gr.Row() as row:
        with gr.Column(scale=1,min_width=160):
            plot3a = gr.Plot(label="plot3a")         
        with gr.Column(scale=8):
            plot3b = gr.Plot(label="plot3b")         


    with gr.Row() as row:
        with gr.Column(scale=0.5):
            plot1 = gr.Plot(label="plot1")         
        with gr.Column(scale=4):
            plot2 = gr.Plot()         
        # with gr.Column(scale=6):
        #     plot3 = gr.Plot()         



    def callback(input_text, ar):
        c_temp = 0.8
        c_top_k=200
        c_max_token = 100
        # if not input_text.startswith('\n'):
        #     input_text = '\n'+input_text
        # ret = model.text_generate(input_text, temperature=c_temp, top_k=(c_top_k), max_new_tokens= (c_max_token))

        enc_text = env.encode(input_text)
        ret = env.model.text_generate(enc_text, temperature=c_temp, top_k=(c_top_k), max_new_tokens= (c_max_token))
        ret = env.decode(ret)
        dec_text = env.decode(enc_text)


        y = model.text_analyse(enc_text = enc_text)
        ### 1. plot the activation of head for each token


        # print([enc_x],enc_x.__len__(),[dec_x],[enc_x_sp],[atext])

        k='final_out_k'
        txt = [ '_'.join(env.decode(xx[:3])) for xx in  model._stats['final_out_k'][0].numpy()]
        # txt = np.array(txt)[:,:,:3]
        # txt = [['_'.join(xxx.tolist()) for xxx in xx] for xx in txt]
        # print(txt)
        text = np.array([[f'{x}->{y}:{z}'] for (x,y,z) in zip(dec_text[:-1],dec_text[1:],txt)])
        # text = ['_' for (x,y) in zip(dec_text,txt)]
        # print(text)
        # text = np.array([list(dec_text)]).T

        fig1 = go.Figure(data=go.Heatmap(
                    z=model._stats['logp'].T.numpy(),
                    text = text,
                    texttemplate="%{text}",
                    textfont={"size":15},
                    colorscale='Viridis',
                    ),
                    )
        fig1.update_layout(
                autosize=False,
                    width=250,
                    # height=200,
                    height=20*len(text),

        )
        fig1.update_yaxes(autorange="reversed")


        # print(model._stats['final_gates'][0].numpy())
        z = model._stats['final_gates'].squeeze(0).numpy()
        # print(z.shape,len(dec_text))
        ytext = [ xx if xx else ' ' for xx in list(dec_text)]
        # ()
        gate_outputs = [[list(env.decode(xxx)) for xxx in (xx)] for xx in  model._stats['gate_out_k'][0].numpy()]
        gate_outputs = np.array(gate_outputs)[:,:,:3]
        gate_outputs_str = [['_'.join(xxx.tolist()) for xxx in xx] for xx in gate_outputs]
        # print(gate_outputs)
        # print(gate_outputs_str)

        fig2 = go.Figure(data=go.Heatmap(
                    z=z,

                    text = gate_outputs_str,
                    texttemplate="%{text}",
                    textfont={"size":10},
                    colorscale='Viridis',

                    # y = ytext[:-1],
                    #     (list(dec_text)[:-1]),
                    # texttemplate="%{text}",
                    # textfont={"size":20},
                    ),
                    )
        fig2.update_layout(
            yaxis=dict(
                tickmode='array',
                ticktext=ytext[:-1],
                tickvals=list(range(len(ytext[:-1]))),
                autorange='reversed',
            ),
                autosize=False,
                    width=20*3.5*z.shape[1],
                    height=20*len(dec_text),
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="LightSteelBlue",
                # side="top",
        )


        ### annotate the context units of all layers
        ### btkl -> tkl
        z = model._stats['int_states'].squeeze(0).square().mean(-1).numpy()
        # print([z.shape])

        v = model._stats['int_states_k'].squeeze(0).numpy()
        
        text = [[ '_'.join(env.decode(xx[:3])) for xx in x] for x in v]
        # z = v
        # z = z[:,:,0]*0
        # z[0,0]=100

        #  np.array([[f'{x}->{y}:{z}'] for (x,y,z) in zip(dec_text[:-1],dec_text[1:],txt)])
        fig3 = go.Figure(data=go.Heatmap(
                    z=z,

                    text = text,
                    texttemplate="%{text}",
                    textfont={"size":10},
                    colorscale='Viridis',

                    # y = ytext[:-1],
                    #     (list(dec_text)[:-1]),
                    # texttemplate="%{text}",
                    # textfont={"size":20},
                    ),
                    )
        fig3.update_layout(
            yaxis=dict(
                tickmode='array',
                ticktext=ytext[:-1],
                tickvals=list(range(len(ytext[:-1]))),
                autorange='reversed',
            ),
                autosize=False,
                    width=20*3.8*z.shape[1],
                    height=20*len(dec_text),
                paper_bgcolor="LightSteelBlue",
                # side="top",
        )

        z = model._stats['atts'].squeeze(0).numpy()
        fig4 = go.Figure(data=go.Heatmap(
                    z=z[:,0],

                    # text = text,
                    # texttemplate="%{text}",
                    # textfont={"size":10},
                    colorscale='Viridis',

                    ),
                    )
        fig4.update_layout(
            yaxis=dict(
                tickmode='array',
                ticktext=ytext[:-1],
                tickvals=list(range(len(ytext[:-1]))),
                autorange='reversed',
            ),
                autosize=False,
                    width=20*4*z.shape[1],
                    height=20*len(dec_text),
                paper_bgcolor="LightSteelBlue",
                # side="top",
        )

        return [ret, fig1, fig2, fig1, fig3, ]


    btn.click(
        fn=callback, 
        inputs=[input_text,  xvar],
        outputs=[output_text, plot1,  plot2, plot3a,plot3b],
        )





    return
         




env = model
with gr.Blocks() as demo:    
    env.model.add_sample_interface(env,gr)
    # add_sample_interface(model)
    # add_analyse_interface(model)
    
demo.launch(**config)

# demo.launch(server_port=6006,server_name="0.0.0.0")
