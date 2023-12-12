"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # self.config.
        self.share_kv = config.share_kv
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        self.is_causal = config.is_causal
        self.use_dropout = config.use_dropout
        nrep = getattr(config,'nrep', 1)
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size*nrep, config.block_size*nrep))
                                        .view(1, 1, config.block_size*nrep, config.block_size*nrep))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        ### lower rank matrix for the internal representation
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if self.share_kv:
            v = k

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            assert 0
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0, 
            is_causal=self.is_causal)
        else:
            # assert 0
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.is_causal:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # att = F.softmax(att, dim=-1)
            att = F.softmax(att, dim=-1)
            if self.use_dropout:
                att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        if self.use_dropout:
            y = self.resid_dropout(y)
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Block13(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention13(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
        return x



class CausalSelfAttention13(nn.Module):

    def __init__(self, config):
        super().__init__()
        # assert 0
        assert config.n_embd % config.n_head == 0
        # self.config.
        self.share_kv = config.share_kv
        # self.share_kv  = 1
        self.share_kv  = 1
        # if self.share_kv:
        #     self.c_attn_q = nn.Parameter(nn.Linear(config.n_embd, config.n_embd, bias=config.bias).weight)
        #     self.c_attn_k = nn.Parameter(nn.Linear(config.n_embd, config.n_embd, bias=config.bias).weight)
        #     self.c_attn_v = nn.Parameter(nn.Linear(config.n_embd, config.n_embd, bias=config.bias).weight)
        # else:

        # key, query, value projections for all heads, but in a batch
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attn_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attn_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
    
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        self.is_causal = config.is_causal
        self.use_dropout = config.use_dropout
        nrep = getattr(config,'nrep', 1)
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size*nrep, config.block_size*nrep))
                                        .view(1, 1, config.block_size*nrep, config.block_size*nrep))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        if self.share_kv:

            q = x.matmul(self.c_attn_q.weight)
            k = x.matmul(self.c_attn_k.weight)
            v = k.matmul(self.c_attn_q.weight.T)
            # v = k.matmul(self.c_attn_v)
        else:
            q = self.c_attn_q(x)
            k = self.c_attn_k(x)
            v = self.c_attn_v(x)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # if self.share_kv:
        #     v = k

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0, 
            is_causal=self.is_causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.is_causal:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # att = F.softmax(att, dim=-1)
            att = F.softmax(att, dim=-1)
            if self.use_dropout:
                att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        if not self.share_kv:
            y = self.c_proj(y)

        if self.use_dropout:
            y = self.resid_dropout(y)
        return y

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    share_kv: bool= False
    optimizer:str='adamw'
    method: int = -1
    is_causal: int = 1
    use_dropout:int = 1
    
class GPT_01(nn.Module):
    @classmethod
    def add_config(cls, config):
        config.suffix=""
        return 

    @classmethod
    def get_out_dir(cls, out_dir,config):
        cls.add_config(config)
        return f'{out_dir}-{cls.__name__}-{config.suffix}'        

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        # breakpoint()
        if self.config.optimizer =='rmsprop':
            optimizer = torch.optim.RMSprop(optim_groups, lr=learning_rate)
            print(f"using optimizer:{optimizer!r}")
        elif self.config.optimizer=='adamw':
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
        else:
            raise Exception(self.config.optimizer)

        return optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # breakpoint()
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class GPT_02(GPT_01):


    def __init__(self, config):
        super(GPT_01, self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        config.r = r = 5
        print(config.block_size)
        print(id(config))
        config.block_size = config.block_size*r
        print(config.block_size)

        self.config = config
        # super().__init__(config)


        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size+1, config.n_embd),
            # wte = nn.Embedding((config.vocab_size)*r + 1, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            wre  = nn.Embedding(config.r, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            
            # v_lin = nn.Linear(config.n_embd ,config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        # self
        return 



    def forward(self, idx, targets=None):
        b,t = idx.shape
        r = self.config.r
        idx=idx[:,:,None].expand((b,t,r)).reshape((b,t*r))
        targets=targets[:,:,None].expand((b,t,r)).reshape((b,t*r))
        logits, loss_arr = self._forward(idx,targets)
        # if 
        if loss_arr is None:
            loss = None
        else:
            loss_arr = (-loss_arr.reshape((b,t,r))).logsumexp(-1) - math.log(r)
            loss = (-loss_arr).mean()

            
        #     logitsr = logits.reshape((b,t,r,-1))
        #     lp = logitsr[:,:,:,-1].log_softmax(-1)
        #     loss_arr = (-loss_arr.reshape((b,t,r)) + lp ).logsumexp(-1)
        #     loss = (-loss_arr).mean()

        logits = logits[:,::r]
        # loss = loss.reshape((b,t,r)).logsumexp(-1)
        return logits, loss


    def _forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        # self.tok_emb = 
        r = self.config.r
        r_emb = self.transformer.wre(pos%r) # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb + r_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # loss_arr = -torch.gather(logits,index=targets.unsqueeze(-1),dim=-1).squeeze(-1)

            loss_arr = F.cross_entropy(logits.transpose(1,2),targets, ignore_index=-1,reduce=False)
            # print(loss_arr.shape)
            # loss_arr = loss_arr.mean()

            # loss_arr = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss_arr = None

        return logits, loss_arr





class GPT_03(GPT_01):


    def __init__(self, config):
        super(GPT_01, self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        config.r = r = 4
        print(config.block_size)
        print(id(config))
        config.block_size = config.block_size*r
        print(config.block_size)

        self.config = config
        # super().__init__(config)

        config.vocab_size = config.vocab_size+1  
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding((config.vocab_size)*r+1, config.n_embd),
            wte_2 = nn.Embedding((config.vocab_size)*r + 1, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            wre = nn.Embedding(config.r, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            
            # v_lin = nn.Linear(config.n_embd ,config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, (config.vocab_size)*r+1, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        # self
        return 



    def forward(self, idx, targets=None):
        b,t = idx.shape
        r = self.config.r
        idx=idx[:,:,None].expand((b,t,r)).reshape((b,t*r))
        targets=targets[:,:,None].expand((b,t,r)).reshape((b,t*r))
        logits, loss_arr = self._forward(idx,targets)
        # if 
        if loss_arr is None:
            loss = None
        else:
            loss_arr = (-loss_arr.reshape((b,t,r))).logsumexp(-1) - math.log(r)
            loss = (-loss_arr).mean()

            
            # logitsr = logits.reshape((b,t,r,-1))
            # lp = logitsr[:,:,:,-1].log_softmax(-1)
            # loss_arr = (-loss_arr.reshape((b,t,r)) + lp ).logsumexp(-1)
            # loss = (-loss_arr).mean()

        logits = logits[:,::r]
        # loss = loss.reshape((b,t,r)).logsumexp(-1)
        return logits, loss


    def _forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        r = self.config.r
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        pos_r = pos%r
        tok_emb = self.transformer.wte(idx ) # token embeddings of shape (b, t, n_embd)
        tok_emb = tok_emb+ self.transformer.wte_2(idx + pos_r * self.config.vocab_size) # token embeddings of shape (b, t, n_embd)

        # self.tok_emb = 
        r_emb = self.transformer.wre(pos_r) # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb + r_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # loss_arr = -torch.gather(logits.reshape((b,)),index=targets.unsqueeze(-1),dim=-1).squeeze(-1)

            loss_arr = F.cross_entropy(logits.transpose(1,2),targets, ignore_index=-1,reduce=False)
            # print(loss_arr.shape)
            # loss_arr = loss_arr.mean()

            # loss_arr = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss_arr = None

        return logits, loss_arr


class GPT_04(GPT_01):


    def __init__(self, config):
        super(GPT_01, self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        config.r = r = 4
        print(config.block_size)
        print(id(config))
        config.block_size = config.block_size*r
        print(config.block_size)

        self.config = config
        # super().__init__(config)

        config.vocab_size = config.vocab_size+1  
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size+1, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            wre = nn.Embedding(config.r, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            
            # v_lin = nn.Linear(config.n_embd ,config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size+1, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        # self
        return 



    def forward(self, idx, targets=None):
        b,t = idx.shape
        r = self.config.r
        r = 1
        idx=idx[:,:,None].expand((b,t,r)).reshape((b,t*r))
        targets=targets[:,:,None].expand((b,t,r)).reshape((b,t*r))
        logits, loss_arr = self._forward(idx,targets)
        # if 
        if loss_arr is None:
            loss = None
        else:
            # loss_arr = (-loss_arr.reshape((b,t,r))).logsumexp(-1) - math.log(r)
            # loss = (-loss_arr).mean()

            
            logitsr = logits.reshape((b,t,r,-1))
            lp = logitsr[:,:,:,-1].log_softmax(-1)
            loss_arr = (-loss_arr.reshape((b,t,r)) + lp ).logsumexp(-1)
            loss = (-loss_arr).mean()

        logits = logits[:,::r]
        # loss = loss.reshape((b,t,r)).logsumexp(-1)
        return logits, loss


    def _forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        r = self.config.r
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        pos_r = pos%r
        tok_emb = self.transformer.wte(idx ) # token embeddings of shape (b, t, n_embd)
        tok_emb = tok_emb+ self.transformer.wte_2(idx + pos_r * self.config.vocab_size) # token embeddings of shape (b, t, n_embd)

        # self.tok_emb = 
        r_emb = self.transformer.wre(pos_r) # position embeddings of shape (t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb + r_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # loss_arr = -torch.gather(logits,index=targets.unsqueeze(-1),dim=-1).squeeze(-1)

            loss_arr = F.cross_entropy(logits.transpose(1,2),targets, ignore_index=-1,reduce=False)
            # print(loss_arr.shape)
            # loss_arr = loss_arr.mean()

            # loss_arr = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss_arr = None

        return logits, loss_arr 



class GPT_05(GPT_01):


    def __init__(self, config):
        config.share_kv = True
        super().__init__(config)




class GPT_06(GPT_01):
    '''
    adding noise into gpt
    '''

            
    def forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
            x = x + torch.normal(0.,  5*1./self.config.n_embd**0.5,x.shape,device=device)
            x,lp = self.transformer.ln_f(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if gradient_only:
            return logits, loss, None
        else:
            return logits, None, loss







class SamplingBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SamplingCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, is_sampling=True):
        dx,lp = self.attn(self.ln_1(x),is_sampling)
        x = x + dx
        x = x + self.mlp(self.ln_2(x))
        return x,lp


class SamplingCausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # self.config.
        self.share_kv = config.share_kv
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        self.use_dropout = config.use_dropout
 
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, is_sampling=True):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if self.share_kv:
            v = k

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # torch.distributions
        # y = 
        if not is_sampling:
            ### non-sampling average
            att = F.softmax(att, dim=-1)
            ### (B,nh,T,T)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            lp= 0.



        else:
            # att_logit = att.log_softmax()
            att_dist = torch.distributions.Categorical(logits=att)
            ### (B,nh,T)
            att_sample = att_dist.sample()
            # (B, nh, T, hs) -> (B, nh, T, hs)
            y = torch.gather(v, index=att_sample[:,:,:,None].expand(v.size()),dim=2)
            ### (B,nh,T)
            lp = att_dist.log_prob(att_sample).sum((1,))
            
            ### (B,T)
            



        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        if self.use_dropout:
            y = self.resid_dropout(y)
        return y,lp


class GPT_07(GPT_01):
    '''
    adding noise into gpt
    '''


    def __init__(self, config):
        super(GPT_01,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.config.mask_index = -1

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([SamplingBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
            
    def forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        lp_internal = torch.zeros((b,),device=device)


        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        target_not_null = targets != self.config.mask_index 

        for block in self.transformer.h:
            x, lp_internal_diff = block(x,is_sampling=True)
            ## lp_internal_diff  ## (b,)
            lp_internal = lp_internal + (lp_internal_diff*target_not_null).sum(-1)

            x     = self.transformer.ln_f(x)
            # if gradient_only:
            #     ### only construct gradient estimator
            #     pass
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        ### (b,t)
        lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        # loss = -lp_external.sum(1).mean(0)
        ct = target_not_null.sum(1)
        lp_total   = lp_external.sum(1) + lp_internal
        # if gradient_only:
        #     # loss_grad = -lp_total.sum(0)/ct.sum(0)
        #     # loss_grad  = (lp_external.sum(1).exp().detach() * -lp_total).sum(0)/ct.sum(0)
        #     loss_grad  = (lp_external.mean(1).exp().detach() * -lp_total).sum(0)/ct.sum(0)
        #     loss_valid = None
        # else:
        #     loss_grad = None
        #     loss_valid = -lp_total.sum(0)/ct.sum(0)

        method = 2 ### model 2 is the best
        # method = 3
        # method = 4

        ### sampling from posterior is difficult https://www.cs.ubc.ca/~schmidtm/Courses/540-W20/L30.pdf
        
        loss_valid = -lp_total.sum(0)/ct.sum(0)
        if method==1:
            loss_grad  = (lp_external.mean(1).exp().detach() * -lp_total).sum(0)/ct.sum(0)
        elif method == 2:
            loss_grad  = -lp_total.sum(0)/ct.sum(0)
        elif method == 3:
            loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10
        elif method == 4:
            loss_grad  = -lp_external.sum(1).sum(0)/ct.sum(0)
            pass
        else:
            assert 0

        # loss_grad  = (lp_external.mean(1).exp().detach() * -lp_total).sum(0)/ct.sum(0)
        # loss_valid = -lp_total.sum(0)/ct.sum(0)
        # loss_grad  = loss_valid




        ### loss_grad:  to construct gradient estimator
        ### loss_valid: to construct the loss estimator.
        ### 2 estimator is separate due to the differentiation on discrete state space
        
        ### [method1]
        ### step 500: train loss 104.7144, val loss 104.8990
        ### step 750: step 750: train loss 103.0445, val loss 103.3315
        ### step 1000: train loss 99.0758, val loss 99.6617

        ### [method2]
        ### step 1000: train loss 109.4333, val loss 108.8561
        ### step 2000: train loss 83.2934, val loss 83.3724
        ### step 2250: train loss 87.2787, val loss 87.2733
        ### step 2500: train loss 79.6755, val loss 79.8309

        ### [method4]
        ### step 1000: train loss 119.3508, val loss 120.1341
        ### step 1250: train loss 119.2215, val loss 120.0744


        ### [method3]
        ### step 1000: train loss 110.0972, val loss 110.2709
        ### step 2250: train loss 96.9372, val loss 97.0283
        ### step 3000: train loss 95.7277, val loss 95.7019




        return logits, loss_grad, loss_valid



class GPT_08(GPT_01):
    '''
    better estimator to sample from posterior
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        # config.optimizer ='adamw'
        # config.method = 5
        config.use_dropout = 0
        config.suffix = f'{config.optimizer}-method{config.method}'
        assert config.method in [5,6,7,8,9]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT_01,self).__init__()

        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        print(config)
        # breakpoint()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([SamplingBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        # ns = 15
        ns = 6
        idx     = idx[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()

        if targets is not None:
            target_not_null = targets != self.config.mask_index 
            targets = targets[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()
        else:
            target_not_null = torch.ones_like(idx)
        ct = target_not_null.sum(1)


        (logits, lp_internal, lp_external) = self._forward(idx,targets,gradient_only)


        lp_total   = lp_external.sum(1) + lp_internal
        # method = 6
        method = self.config.method
        logits = logits.reshape(((ns,b,t,-1)))
        
        lp_internal = lp_internal.reshape((ns,b))
        lp_total    = lp_total.reshape((ns,b,))
        lp_external = lp_external.reshape((ns,b,t))   

        ### sampling from posterior is difficult
        ### https://www.cs.ubc.ca/~schmidtm/Courses/540-W20/L30.pdf
        
        # loss_valid = -(lp_total.sum(-1).logsumexp(0) - math.log(ns))/ct.sum(0)
        loss_valid = -(lp_total.logsumexp(0).sum(0))/ct.sum(0)
        if 0:
            pass
        elif method == 5:
            ### equally weighted sample from p(z|h)
            # loss_grad  = -(lp_total.logsumexp(0) - math.log(ns))/ct.sum(0)
            loss_grad  = -(lp_total.logsumexp(0).sum(0))/ct.sum(0)
        elif method == 6:
            ### re weighted sample from p(z|h)
            ##(ns,b)
            # lp_total_sum = lp_total.sum(-1)
            reweight_lp  = lp_total + lp_total.log_softmax(0)
            loss_grad    = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            pass
            # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10
        elif method == 7:
            ### re weighted sample using q()
            ##(ns,b)
            # lp_external = 
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 

            reweight_lp  = lp_total + reweight_lp
            loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            pass
            # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10

        elif method == 8:
            ### re weighted sample using q()
            ##(ns,b)
            # lp_external = 
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 

            reweight_lp  = lp_total + reweight_lp.detach()
            loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            # loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            pass
            # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10

        elif method == 9:
            ### re weighted sample using q()
            ##(ns,b)
            # lp_external = 
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 

            reweight_lp  = lp_total + reweight_lp.detach()
            loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            pass
            # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10
        else:
            assert 0
        ### (b,v)
        # breakpoint()
        # print(logits.shape)
        logits0 = logits
        logits = (lp_internal[:,:,None,None] + logits.log_softmax(-1)).logsumexp(0)
        # breakpoint()
        # if logits.shape.__len__()==2:
        #     logits = logits[:,None]
        # breakpoint()




        return logits, loss_grad, loss_valid


    def _forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        lp_internal = torch.zeros((b,),device=device)


        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # x = self.transformer.drop(tok_emb + pos_emb)
        x = tok_emb + pos_emb
        if self.config.use_dropout:
            x = self.transformer.drop(x)

        target_not_null = targets != self.config.mask_index 

        for block in self.transformer.h:
            x, lp_internal_diff = block(x,is_sampling=True)
            ## lp_internal_diff  ## (b,)
            lp_internal = lp_internal + (lp_internal_diff*target_not_null).sum(-1)

            x     = self.transformer.ln_f(x)
            # if gradient_only:
            #     ### only construct gradient estimator
            #     pass
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        ### (b,t)
        if targets is not None:
            lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        else:
            lp_external = torch.zeros_like(logits[:,:,0])
        # ct = target_not_null.sum(1)
        return logits, lp_internal,lp_external








class GPT_09(GPT_01):
    '''
    compress the internal representation to discrete representations
    g200 
iter 240: loss 7.9497, loss_eval:7.9357  time 702.09ms, mfu 1.02%
step 250: train loss 7.9466, val loss 8.0475
iter 490: loss 7.6413, loss_eval:7.6197  time 731.53ms, mfu 1.00%
step 500: train loss 7.7671, val loss 7.9329

step 250: train loss 7.5792, val loss 7.7099
iter 490: loss 7.4772, loss_eval:7.4628  time 714.63ms, mfu 1.01%
step 500: train loss 7.5697, val loss 7.7299

step 1000: train loss 7.7695, val loss 7.9453

method0,g200
iter 240: loss 7.8003, loss_eval:7.7760  time 727.53ms, mfu 1.02%
step 250: train loss 7.7858, val loss 7.9133
iter 490: loss 7.6439, loss_eval:7.6230  time 713.47ms, mfu 1.00%
step 500: train loss 7.7632, val loss 7.9285


method9,g200
iter 240: loss 6.3232, loss_eval:7.7767  time 714.52ms, mfu 1.02%
step 250: train loss 7.7861, val loss 7.9164
iter 490: loss 6.1698, loss_eval:7.6274  time 712.11ms, mfu 1.01%
 step 500: train loss 7.7606, val loss 7.9289

method10,g200
iter 240: loss 4.8718, loss_eval:7.7997  time 726.34ms, mfu 1.02%
step 250: train loss 7.7960, val loss 7.9173
iter 490: loss 4.6816, loss_eval:7.6239  time 688.22ms, mfu 1.01%
step 500: train loss 7.7682, val loss 7.9183


method10,g50
iter 240: loss 5.2655, loss_eval:7.6712  time 725.48ms, mfu 1.02%
step 250: train loss 7.7201, val loss 7.7941
iter 490: loss 4.9916, loss_eval:7.4936  time 726.84ms, mfu 1.00%
step 500: train loss 7.5804, val loss 7.7097

method10,g50
iter 240: loss 5.3594, loss_eval:7.7166  time 717.04ms, mfu 1.02%
step 250: train loss 7.6947, val loss 7.7751
iter 490: loss 4.9512, loss_eval:7.4518  time 730.31ms, mfu 1.00%
step 500: train loss 7.5686, val loss 7.7386




method11,g50
iter 240: loss 6.4530, loss_eval:7.6488  time 710.61ms, mfu 1.01%
step 250: train loss 7.6761, val loss 7.7710


method11,g50
iter 240: loss 6.5146, loss_eval:7.6650  time 716.95ms, mfu 1.02%
step 250: train loss 7.6280, val loss 7.7259

method12,g50
iter 240: loss 5.8304, loss_eval:7.6571  time 716.91ms, mfu 1.01%
step 250: train loss 7.6177, val loss 7.7205
iter 490: loss 5.6404, loss_eval:7.4676  time 719.60ms, mfu 1.01%
step 500: train loss 7.5693, val loss 7.7210

method12,g200
iter 240: loss 5.6181, loss_eval:7.8786  time 710.15ms, mfu 1.01%
step 250: train loss 7.8585, val loss 7.9670
iter 490: loss 5.3650, loss_eval:7.6248  time 723.61ms, mfu 1.00%
step 500: train loss 7.7636, val loss 7.9249
iter 740: loss 5.4536, loss_eval:7.7338  time 732.31ms, mfu 1.00%
step 750: train loss 7.7872, val loss 7.9550
iter 990: loss 5.4332, loss_eval:7.7669  time 722.89ms, mfu 1.01%
step 1000: train loss 7.7685, val loss 7.9637
iter 1240: loss 5.2694, loss_eval:7.5897  time 726.83ms, mfu 1.00%
step 1250: train loss 7.7734, val loss 7.9823


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.k = 15
        config.g = 50
        # config.g = 20 0
        # config.method = 8
        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=False
        config.n_layer= 1
        assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT_01,self).__init__()

        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        print(config)
        # breakpoint()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            k_query = nn.Linear(config.n_embd,config.k, ),
            g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h_enc = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            h_dec = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        ns = 5
        idx     = idx[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()

        if targets is not None:
            target_not_null = targets != self.config.mask_index 
            targets = targets[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()
        else:
            target_not_null = torch.ones_like(idx)
        ct = target_not_null.sum(1)


        (logits, lp_internal, lp_external) = self._forward(idx,targets,gradient_only)

        ### (nsb,t)
        g = self.config.g
        k = self.config.k
        lp_k = k*math.log(1./t) + k*math.log(1/g)  ### adding prior on pk and gk var
        lp_total   = lp_external.sum(1)  + lp_k

        method = self.config.method
        logits = logits.reshape(((ns,b,t,-1)))

        lp_internal = lp_internal.reshape((ns,b))
        lp_total    = lp_total.reshape((ns,b,))
        lp_external = lp_external.reshape((ns,b,t))   

        ### sampling from posterior is difficult
        ### https://www.cs.ubc.ca/~schmidtm/Courses/540-W20/L30.pdf
        
        # loss_valid = -(lp_total.sum(-1).logsumexp(0) - math.log(ns))/ct.sum(0)
        loss_valid = -(lp_total.logsumexp(0).sum(0))/ct.sum(0)

        ### re weighted sample using q()
        ##(ns,b)

        #### need to reweight with
        if method==8:
            reweight_lp  = (lp_external.sum(2) - lp_internal).log_softmax(0) 
            reweight_lp  = lp_total + reweight_lp
            loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
        elif method==9:
            ### adding p(c)p(x|c)q(c|x)
            # lp_raw = lp_external.sum(-1) + (1+ lp_internal.exp()).log() 
            lp_raw = lp_external.sum(-1) + F.softplus(lp_internal)
            reweight_lp  = (lp_raw - lp_internal).log_softmax(0) 
            reweight_lp  = lp_raw + reweight_lp.detach()
            loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            pass
        elif method==10:
            ### lp_k
            ### p(c)p(x|c) +  p(x|c)q(c|x)
            lp_raw = lp_external.sum(-1) +  F.softplus(lp_internal - lp_k)
            reweight_lp  = (lp_raw - lp_internal).log_softmax(0) 
            reweight_lp  = lp_raw + reweight_lp.detach()
            loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            pass 
        elif method==11:
            ### using only p(x|c)q(c|x)
            ### lp_k
            lp_raw = lp_external.sum(-1) +  lp_internal
            reweight_lp  = lp_external.sum(-1).log_softmax(0) 
            reweight_lp  = lp_raw + reweight_lp.detach()
            loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            pass 

        elif method==12:
            ### p(c)p(x|c) +  p(x|c)q(c|x)  fitting both generative loss and autoencoder loss
            ### using average of log
            lp_raw = lp_external.sum(-1) +  F.softplus(lp_internal - lp_k)
            reweight_lp  = (lp_raw - lp_internal).log_softmax(0) 
            reweight_lp  = lp_raw + reweight_lp.detach()
            loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            pass 
        elif method==13:
            ### p(c)p(x|c) +  p(x|c)q(c|x)  fitting both generative loss and autoencoder loss
            ### using average of log
            # lp_raw = lp_external.sum(-1) +  F.softplus(lp_internal - lp_k)
            # reweight_lp  = (lp_raw - lp_internal).log_softmax(0) 
            # reweight_lp  = lp_raw + reweight_lp.detach()
            # loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # pass 
            loss_grad = loss_valid
        else:
            assert 0
        # pass
        # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10

        logits0 = logits
        logits = (lp_internal[:,:,None,None] + logits.log_softmax(-1)).logsumexp(0)


        return logits, loss_grad, loss_valid


    def _forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        lp_internal = torch.zeros((b,),device=device)

        md = self.transformer ### model dict


        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        target_not_null = targets != self.config.mask_index 

        for block in self.transformer.h_enc:
            x     = block(x) 
            x     = self.transformer.ln_f(x)

    
        # (b, t, n_embd)
        # x = self.transformer.ln_f(x)

        ### encode the distributed vectors into discrete representations
        ### for each of k component
        ### find the vector that best match the prototype, record p_k .
        ### then find the best matching vector from the vector database 
        ### and record the index g_k
        k        = self.config.k
        k_query  = md.k_query ### (k, n_embed)
        g_vects  = self.transformer.ln_f(md.g_vects.weight) ### (g, n_embed)        

        ### (b,k,t)
        pk_dist = torch.distributions.Categorical(logits=k_query(x-pos_emb).transpose(1,2))
        pk      = pk_dist.sample()

        ### (b, k, ne)
        idx_b   = torch.arange(b,device=device)[:,None].expand(b,k)
        vk      = (x-pos_emb)[idx_b, pk, :]
        
        ### (b, k,  g)
        gk_dist = torch.distributions.Categorical(logits = vk.matmul( g_vects.T )/self.config.n_embd**0.5)
        gk      = gk_dist.sample() 

        ### lp_internal  (b, )
        lp_internal = (pk_dist.log_prob(pk) + gk_dist.log_prob(gk)).sum(-1)

        ### pk           (b, k)
        ### gk           (b, k)

        # n_embd  = self.config.n_embd
        # x_recon = torch.zeros((b,t,n_embd),device=device)
        # print(pos_emb.shape)
        # print(pk[0:2])
        # print(gk[0:2])
        # .sum(0))

        x_recon = pos_emb[None].repeat((b,1,1))
        x_recon[:,pk,:] = x_recon[:,pk,:] + g_vects[gk,:]

        ### then given p_k and g_k, reconstruct the representation 
        ### by putting the vector to the corresponding position

        x = x_recon
        for block in self.transformer.h_dec:
        # for block in self.transformer.h_enc[::-1]:
            x     = block(x) 
            x     = self.transformer.ln_f(x)
        # # x     = self.transformer.ln_f(x)

        logits = self.lm_head(x - pos_emb[None].repeat((b,1,1)))

        ### (b,t)
        if targets is not None:
            lp_external  = -F.cross_entropy(logits.transpose(1,2), idx, ignore_index=self.config.mask_index,reduction='none')
        else:
            lp_external = torch.zeros_like(logits[:,:,0])
        # ct = target_not_null.sum(1)
        return logits, lp_internal,lp_external


class GPT_10(GPT_01):
    '''
    compress the internal representation to discrete representations
    g200 
iter 240: loss 7.9497, loss_eval:7.9357  time 702.09ms, mfu 1.02%
step 250: train loss 7.9466, val loss 8.0475


layer1,k5,g200
iter 240: loss 6.4848, loss_eval:6.4761  time 514.69ms, mfu 0.81%
step 250: train loss 6.3367, val loss 6.4752

iter 490: loss 2.5496, loss_eval:2.5452  time 510.20ms, mfu 0.80%
step 500: train loss 1.9330, val loss 2.1382


layer1,k2,g2
iter 240: loss 6.0271, loss_eval:6.0157  time 496.76ms, mfu 0.82%
step 250: train loss 5.6433, val loss 5.8436
iter 490: loss 1.7409, loss_eval:1.7320  time 481.57ms, mfu 0.82%
step 500: train loss 1.1680, val loss 1.3470



additive
step 500: train loss 5.4136, val loss 5.8899
iter 500: loss 5.5735, loss_eval:5.3794  time 11853.54ms, mfu -100.00%
iter 510: loss 5.8282, loss_eval:5.5868  time 620.74ms, mfu 1.17%
iter 490: loss 5.9515, loss_eval:5.7734  time 556.78ms, mfu 1.19%
step 500: train loss 5.4961, val loss 5.9659
iter 740: loss 5.8171, loss_eval:5.5836  time 634.95ms, mfu 1.19%
step 750: train loss 5.1832, val loss 5.7880
iter 990: loss 5.2952, loss_eval:5.0550  time 597.41ms, mfu 1.17%
step 1000: train loss 4.7811, val loss 5.5304
iter 2240: loss 4.7365, loss_eval:4.5003  time 566.07ms, mfu 1.17%
step 2250: train loss 4.1148, val loss 5.7818


extract
iter 240: loss 5.8068, loss_eval:5.6415  time 633.79ms, mfu 1.18%
step 250: train loss 5.6000, val loss 5.8979
iter 490: loss 5.4432, loss_eval:5.2642  time 557.39ms, mfu 1.20%
step 500: train loss 5.0229, val loss 5.5521
iter 740: loss 5.2986, loss_eval:5.0846  time 576.89ms, mfu 1.22%
step 750: train loss 4.6732, val loss 5.3812
iter 2240: loss 4.3540, loss_eval:4.0936  time 610.82ms, mfu 1.19%
step 2250: train loss 3.7010, val loss 5.4492
iter 2490: loss 4.4014, loss_eval:4.1881  time 625.22ms, mfu 1.17%
step 2500: train loss 3.6815, val loss 5.5955

extract g200
iter 1740: loss 4.8787, loss_eval:4.5646  time 1088.49ms, mfu 0.85%
 step 1750: train loss 4.3611, val loss 5.7227


### maybe overfitting?


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        config.g = 200
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0

        # config.k = 2
        # config.g = 2

        # config.n_layer= 1

        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT_01,self).__init__()

        # config.block_size = config.block_size*

        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        print(config)
        # breakpoint()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # k_query = nn.Linear(config.n_embd,config.k, ),
            k_query = nn.Linear(config.n_embd,config.n_embd, ),
            k_vects = nn.Embedding(config.g, config.n_embd),
            g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h_enc = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            h_dec = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        ns = 5
        idx     = idx[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()

        if targets is not None:
            target_not_null = targets != self.config.mask_index 
            targets = targets[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()
        else:
            target_not_null = torch.ones_like(idx)
        ct = target_not_null.sum(1)


        (logits, lp_internal, lp_external) = self._forward(idx,targets,gradient_only)

        ### (nsb,t)
        g = self.config.g
        k = self.config.k
        lp_k = k*math.log(1./t) + k*math.log(1/g)  ### adding prior on pk and gk var
        lp_total   = lp_external.sum(1)  + lp_internal

        method = self.config.method
        logits = logits.reshape(((ns,b,t,-1)))

        lp_internal = lp_internal.reshape((ns,b))
        lp_total    = lp_total.reshape((ns,b,))
        lp_external = lp_external.reshape((ns,b,t))   

        ### sampling from posterior is difficult
        ### https://www.cs.ubc.ca/~schmidtm/Courses/540-W20/L30.pdf
        
        # loss_valid = -(lp_total.sum(-1).logsumexp(0) - math.log(ns))/ct.sum(0)
        loss_valid = -(lp_total.logsumexp(0).sum(0))/ct.sum(0)

        ##(ns,b)        
        if 0:
            pass
        elif method in (8,10):
            ### re weighted sample using q()
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 

            reweight_lp  = lp_total + reweight_lp.detach()
            # loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10
        elif method == 12:
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 
            reweight_lp  = lp_total + reweight_lp.detach()
            loss_grad    = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
        else:
            assert 0
        ### (b,v)
        logits = (lp_internal[:,:,None,None] + logits.log_softmax(-1)).logsumexp(0)

        return logits, loss_grad, loss_valid


    def _forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        lp_internal = torch.zeros((b,),device=device)

        md = self.transformer ### model dict

        k        = self.config.k
        k_query  = md.k_query ### (k, n_embed)
        g_vects  = self.transformer.ln_f(md.g_vects.weight) ### (g, n_embed)        
        k_vects  = self.transformer.ln_f(md.k_vects.weight) ### (g, n_embed)        

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        if self.config.use_dropout:
            x = self.transformer.drop(x)

        target_not_null = targets != self.config.mask_index 

        lp_internal = torch.zeros((b,),device=device)
        #### the problem here is to insert context vector
        ####
        e = self.config.n_embd
        l = self.config.n_layer
        if self.config.method==8:
            for i,block in enumerate(self.transformer.h_dec):
                # if i==5:
                if i==4:
                        ## (b, t, e)
                        xk = md.ln_f(k_query(x))
                        ### (b, t, g)
                        logits  = xk.matmul( g_vects.T )/self.config.n_embd**0.5
                        gk_dist = torch.distributions.Categorical(logits = logits )
                        ### (b, t)
                        gk      = gk_dist.sample() 
                        # breakpoint()
                        ### (b, t, e)
                        # breakpoint()
                        gv      = g_vects[gk,:]
                        xx      = torch.cat([gv[:,:,None],x[:,:,None]],dim=2)
                        xx = xx.reshape((b,2*t,e))

                        xx     = block(xx)
                        
                        xx     = xx.reshape((b,t,2,e))[:,:,-1]
                    #     x     = x[:,k:] 
                        x      = self.transformer.ln_f(xx)
                        lp_internal += (gk_dist.log_prob(gk)).sum(-1)

                        # print(gk[0:2].float().std().item())
                        # print(gk[0:2].float().mean().item())


                    # #### additive
                    # ### (b, t, e)
                    # # xk = md.ln_f(k_query(x-pos_emb))
                    # ### (b, t, g)
                    # logits  = x.matmul( k_vects.T )/self.config.n_embd**0.5
                    # gk_dist = torch.distributions.Categorical(logits = logits )
                    # ### (b, t)
                    # gk      = gk_dist.sample() 
                    # # breakpoint()
                    # ### (b, t, e)
                    # # breakpoint()
                    # gv      = g_vects[gk,:]
                    # # xx      = torch.cat([gv[:,:,None],x[:,:,None]],dim=2)
                    # # xx      = xx.reshape((b,2*t,e))
                    # xx     = x
                    # xx     = (block(xx) + gv)*0.5
                    
                    # # xx     = xx.reshape((b,t,2,e))[:,:,-1]
                    # x      = self.transformer.ln_f(xx)
                    # lp_internal += (gk_dist.log_prob(gk)).sum(-1)

                    # # print(gk[0:2].float().std().item())
                    # # pass                    

                else:
                    x     = block(x)
                    # x     = x[:,k:] 
                    x     = self.transformer.ln_f(x)            

                # if i==5:
                #     # print(gk[0:2])
                #     pass
        else:
    
            ### interleaving the context vector with token vector
            ### accessing the vector db 
            ### (b,t,2,e)


            for block in self.transformer.h_dec:
                x     = block(x)
                # x     = x[:,k:] 
                x     = self.transformer.ln_f(x)            


        #### lets extract some vector from vector database

        # (b, t, n_embd)

        ### encode the distributed vectors into discrete representations
        ### for each of k component
        ### find the vector that best match the prototype, record p_k .
        ### then find the best matching vector from the vector database 
        ### and record the index g_k


        ### pk           (b, k)
        ### gk           (b, k)

        # n_embd  = self.config.n_embd
        # x_recon = torch.zeros((b,t,n_embd),device=device)
        # print(pk[0:2])
        # print(gk[0:2])


        # logits = self.lm_head(x - pos_emb[None].repeat((b,1,1)))
        # logits = self.lm_head(x - pos_emb[None].repeat((b,1,1)))
        # x = self.transformer.ln_f((x - pos_emb[None].repeat((b,1,1))))
        logits = self.lm_head(x)

        ### (b,t)
        if targets is not None:
            lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        else:
            lp_external = torch.zeros_like(logits[:,:,0])
        # ct = target_not_null.sum(1)
        return logits, lp_internal,lp_external




class GPT_11(GPT_01):
    '''
    compress the internal representation to discrete representations
    g200 


### maybe overfitting?


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        config.g = 200
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0


        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT_01,self).__init__()

        # config.block_size = config.block_size*

        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        print(config)
        # breakpoint()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # k_query = nn.Linear(config.n_embd,config.k, ),
            k_query = nn.Linear(config.n_embd,config.n_embd, ),
            k_vects = nn.Embedding(config.g, config.n_embd),
            g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h_enc = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            h_dec = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        ns = 1
        idx     = idx[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()

        if targets is not None:
            target_not_null = targets != self.config.mask_index 
            targets = targets[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()
        else:
            target_not_null = torch.ones_like(idx)
        ct = target_not_null.sum(1)


        (logits, lp_internal, lp_external) = self._forward(idx,targets,gradient_only)

        ### (nsb,t)
        g = self.config.g
        k = self.config.k
        lp_k = k*math.log(1./t) + k*math.log(1/g)  ### adding prior on pk and gk var
        lp_total   = lp_external.sum(1)  + lp_internal

        method = self.config.method
        logits = logits.reshape(((ns,b,t,-1)))

        lp_internal = lp_internal.reshape((ns,b))
        lp_total    = lp_total.reshape((ns,b,))
        lp_external = lp_external.reshape((ns,b,t))   

        ### sampling from posterior is difficult
        ### https://www.cs.ubc.ca/~schmidtm/Courses/540-W20/L30.pdf
        
        # loss_valid = -(lp_total.sum(-1).logsumexp(0) - math.log(ns))/ct.sum(0)
        loss_valid = -(lp_total.logsumexp(0).sum(0))/ct.sum(0)

        ##(ns,b)        
        if 0:
            pass
        elif method in (8,10):
            ### re weighted sample using q()
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 

            reweight_lp  = lp_total + reweight_lp.detach()
            # loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10
        elif method == 12:
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 
            reweight_lp  = lp_total + reweight_lp.detach()
            loss_grad    = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
        else:
            assert 0
        ### (b,v)
        logits = (lp_internal[:,:,None,None] + logits.log_softmax(-1)).logsumexp(0)

        return logits, loss_grad, loss_valid


    def _forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        lp_internal = torch.zeros((b,),device=device)

        md = self.transformer ### model dict

        k        = self.config.k
        k_query  = md.k_query ### (k, n_embed)
        g_vects  = self.transformer.ln_f(md.g_vects.weight) ### (g, n_embed)        
        k_vects  = self.transformer.ln_f(md.k_vects.weight) ### (g, n_embed)        

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)

        x = tok_emb + pos_emb
        # x = tok_emb 
        if self.config.use_dropout:
            x = self.transformer.drop(x)

        target_not_null = targets != self.config.mask_index 

        lp_internal = torch.zeros((b,),device=device)
        #### the problem here is to insert context vector
        ####
        e = self.config.n_embd
        l = self.config.n_layer
        '''
        ### pos enc + / residual +
        step 250: train loss 5.6127, val loss 5.8632
        step 500: train loss 4.9467, val loss 5.3748
        step 750: train loss 4.5654, val loss 5.1806
        step 1000: train loss 4.2439, val loss 5.0947

        ### pos enc - / residual + 0.5
        step 250: train loss 5.4925, val loss 5.8086
        step 500: train loss 4.8301, val loss 5.3399
        step 750: train loss 4.4383, val loss 5.1964

        ### pos enc - / residual -
        step 250: train loss 4.8621, val loss 5.3303
        step 500: train loss 4.2734, val loss 4.9331
        step 750: train loss 3.9459, val loss 4.9137


        ### pos enc - / residual + 0.8


        ### pos enc + / residual -
        step 250: train loss 4.9676, val loss 5.3853


        ### pos enc + / residual - / middle_layer_norm -
        step 250: train loss 5.0719, val loss 5.4513
        step 500: train loss 4.3034, val loss 4.8873
        step 750: train loss 3.9363, val loss 4.7851


        ### pos enc + / residual - / middle_layer_norm - / remove_pos_emb
        step 250: train loss 5.0620, val loss 5.4487
        step 500: train loss 4.2928, val loss 4.8785
        step 750: train loss 3.9345, val loss 4.7823


        ### pos enc + / residual0.8 / middle_layer_norm - / remove_pos_emb
        step 250: train loss 5.1025, val loss 5.4564
        step 500: train loss 4.3291, val loss 4.8816
        step 750: train loss 3.9590, val loss 4.7685
        step 1000: train loss 3.6484, val loss 4.7441
        step 1250: train loss 3.3402, val loss 4.8668


        ### pos enc -/ residual0.8 / middle_layer_norm - / 
        step 250: train loss 5.0334, val loss 5.4276
        step 500: train loss 4.2829, val loss 4.8840
        step 750: train loss 3.9195, val loss 4.8110
        step 1000: train loss 3.6188, val loss 4.8142
        step 1250: train loss 3.3348, val loss 4.9116



        ### pos enc -/ residual -  / middle_layer_norm - / 
        step 250: train loss 4.9824, val loss 5.4127
        step 500: train loss 4.2946, val loss 4.9320
        step 750: train loss 3.9480, val loss 4.8661





        ''' 

        # block0 = self.transformer.h_dec[0]
        for i in range(3):
            # enumerate(self.transformer.h_dec):
            # if i==5:
            # random.ch
            # block0 = random.choice(self.transformer.h_dec)
            x0    = x[:,:-1]
            dx = 0. 
            for ii,block in enumerate(self.transformer.h_dec):
                dx     += block(x)
            x = dx / (ii+1)

            # x[:,1:] = 0.2 * x0 + 0.8*x[:,1:]
            # x     = self.transformer.ln_f(x)            
        x = x- pos_emb
        x     = self.transformer.ln_f(x)            


        logits = self.lm_head(x)

        ### (b,t)
        if targets is not None:
            lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        else:
            lp_external = torch.zeros_like(logits[:,:,0])
        # ct = target_not_null.sum(1)
        return logits, lp_internal,lp_external        

class GPT_13(GPT_01):
    '''
    compress the internal representation to discrete representations
    g200 


### maybe overfitting?

    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        config.g = 200
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0


        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT_01,self).__init__()

        # config.block_size = config.block_size*

        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        print(config)
        # breakpoint()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # k_query = nn.Linear(config.n_embd,config.k, ),
            # k_query = nn.Linear(config.n_embd,config.n_embd, ),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h_enc = nn.ModuleList([Block13(config) for _ in range(config.n_layer)]),
            h_dec = nn.ModuleList([Block13(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        ns = 1
        idx     = idx[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()

        if targets is not None:
            target_not_null = targets != self.config.mask_index 
            targets = targets[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()
        else:
            target_not_null = torch.ones_like(idx)
        ct = target_not_null.sum(1)


        (logits, lp_internal, lp_external) = self._forward(idx,targets,gradient_only)

        ### (nsb,t)
        g = self.config.g
        k = self.config.k
        lp_k = k*math.log(1./t) + k*math.log(1/g)  ### adding prior on pk and gk var
        lp_total   = lp_external.sum(1)  + lp_internal

        method = self.config.method
        logits = logits.reshape(((ns,b,t,-1)))

        lp_internal = lp_internal.reshape((ns,b))
        lp_total    = lp_total.reshape((ns,b,))
        lp_external = lp_external.reshape((ns,b,t))   

        ### sampling from posterior is difficult
        ### https://www.cs.ubc.ca/~schmidtm/Courses/540-W20/L30.pdf
        
        # loss_valid = -(lp_total.sum(-1).logsumexp(0) - math.log(ns))/ct.sum(0)
        loss_valid = -(lp_total.logsumexp(0).sum(0))/ct.sum(0)

        ##(ns,b)        
        if 0:
            pass
        elif method in (8,10,12):
            ### re weighted sample using q()
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 

            reweight_lp  = lp_total + reweight_lp.detach()
            # loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10
        else:
            assert 0
        ### (b,v)
        logits = (lp_internal[:,:,None,None] + logits.log_softmax(-1)).logsumexp(0)

        return logits, loss_grad, loss_valid


    def _forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        lp_internal = torch.zeros((b,),device=device)

        md = self.transformer ### model dict


        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)

        x = tok_emb + pos_emb
        # x = tok_emb 
        if self.config.use_dropout:
            x = self.transformer.drop(x)

        target_not_null = targets != self.config.mask_index 

        lp_internal = torch.zeros((b,),device=device)
        #### the problem here is to insert context vector
        ####
        e = self.config.n_embd
        l = self.config.n_layer
        '''
        ### pos enc + / residual - / middle_layer_norm - / remove_pos_emb
        step 250: train loss 5.0620, val loss 5.4487
        step 500: train loss 4.2928, val loss 4.8785
        step 750: train loss 3.9345, val loss 4.7823


        ###
        iter 240: loss 5.3922, loss_eval:5.3922  time 466.20ms, mfu 1.56%
        step 250: train loss 5.3472, val loss 5.6183
        step 500: train loss 4.4483, val loss 4.9606
        step 750: train loss 4.0301, val loss 4.8203
        step 1000: train loss 3.7105, val loss 4.8578


### GPT_13 performs good without v matrix
iter 480: loss 5.2782, loss_eval:5.2782  time 397.97ms, mfu 1.82%
iter 490: loss 5.1611, loss_eval:5.1611  time 396.14ms, mfu 1.82%
step 500: train loss 5.3144, val loss 5.5583



iter 720: loss 4.8826, loss_eval:4.8826  time 408.22ms, mfu 1.79%
iter 730: loss 4.9931, loss_eval:4.9931  time 417.93ms, mfu 1.79%
iter 740: loss 4.7547, loss_eval:4.7547  time 411.86ms, mfu 1.78%
step 750: train loss 4.7335, val loss 5.2412


iter 990: loss 4.3063, loss_eval:4.3063  time 414.63ms, mfu 1.80%
step 1000: train loss 4.4130, val loss 5.1087

iter 1240: loss 4.2099, loss_eval:4.2099  time 408.33ms, mfu 1.82%
step 1250: train loss 4.2325, val loss 5.0573

iter 1490: loss 4.1146, loss_eval:4.1146  time 322.41ms, mfu 1.84%
step 1500: train loss 4.0676, val loss 5.0974


### with less attention head
iter 470: loss 5.2525, loss_eval:5.2525  time 407.72ms, mfu 1.77%
iter 480: loss 5.3915, loss_eval:5.3915  time 401.58ms, mfu 1.78%
iter 490: loss 5.2891, loss_eval:5.2891  time 402.85ms, mfu 1.78%
step 500: train loss 5.4244, val loss 5.6316

iter 720: loss 5.0812, loss_eval:5.0812  time 421.52ms, mfu 1.83%
iter 730: loss 5.1902, loss_eval:5.1902  time 409.62ms, mfu 1.82%
iter 740: loss 4.9658, loss_eval:4.9658  time 385.92ms, mfu 1.83%
step 750: train loss 4.9113, val loss 5.3787

iter 990: loss 4.5471, loss_eval:4.5471  time 400.76ms, mfu 1.81%
step 1000: train loss 4.6388, val loss 5.2777
saving checkpoint to out-shakespeare-word-GPT_13-adamw-method12

        ''' 

        block0 = self.transformer.h_dec[0]

        h2 = torch.zeros((b,t,e),device=device)
        x_init = x
        # h2 = torch.zeros((b,t,e),device=device) + x_init
        for i in range(3):
            # enumerate(self.transformer.h_dec):
            # block0 = random.choice(self.transformer.h_dec)
            x0    = x[:,:-1]
            dx = 0. 
            for ii,block in enumerate(self.transformer.h_dec):
                # dx     += block(h2)
                dx     += block0(h2)
            # h2 = 0.5 * (h2 + dx) 
            # h2 = (0.5*(x_init + dx / (ii+1)))
            h2 = (0.5*(x_init + dx / (ii+1)) + h2)*0.5
            # x = dx / (ii+1)
        x = h2

        # x = x- pos_emb
        x     = self.transformer.ln_f(x)            


        logits = self.lm_head(x)

        ### (b,t)
        if targets is not None:
            lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        else:
            lp_external = torch.zeros_like(logits[:,:,0])
        # ct = target_not_null.sum(1)
        return logits, lp_internal,lp_external                


class CM01(GPT_01):
    '''
    compress the internal representation to discrete representations
    g200 


### maybe overfitting?

    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        config.g = 200
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//16


        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT_01,self).__init__()

        # config.block_size = config.block_size*

        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        print(config)
        # breakpoint()
        ne = config.n_embd
        nh = config.n_head
        nei = config.n_internal

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # k_query = nn.Linear(config.n_embd,config.k, ),
            # k_query = nn.Linear(config.n_embd,config.n_embd, ),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            lin_output   = nn.Linear(ne, nh*nei,bias=config.bias),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,nh*nei,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,nei,bias=config.bias),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t    = idx.size()
        ns      = 1
        idx     = idx[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()

        if targets is not None:
            target_not_null = targets != self.config.mask_index 
            targets = targets[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()
        else:
            target_not_null = torch.ones_like(idx)
        ct = target_not_null.sum(1)


        (logits, lp_internal, lp_external) = self._forward(idx,targets,gradient_only)

        ### (nsb,t)
        g = self.config.g
        k = self.config.k
        lp_k = k*math.log(1./t) + k*math.log(1/g)  ### adding prior on pk and gk var
        lp_total   = lp_external.sum(1)  + lp_internal

        method = self.config.method
        logits = logits.reshape(((ns,b,t,-1)))

        lp_internal = lp_internal.reshape((ns,b))
        lp_total    = lp_total.reshape((ns,b,))
        lp_external = lp_external.reshape((ns,b,t))   

        ### sampling from posterior is difficult
        ### https://www.cs.ubc.ca/~schmidtm/Courses/540-W20/L30.pdf
        
        # loss_valid = -(lp_total.sum(-1).logsumexp(0) - math.log(ns))/ct.sum(0)
        loss_valid = -(lp_total.logsumexp(0).sum(0))/ct.sum(0)

        ##(ns,b)        
        if 0:
            pass
        elif method in (8,10,12):
            ### re weighted sample using q()
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 

            reweight_lp  = lp_total + reweight_lp.detach()
            # loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10
        else:
            assert 0
        ### (b,v)
        logits = (lp_internal[:,:,None,None] + logits.log_softmax(-1)).logsumexp(0)

        return logits, loss_grad, loss_valid


    def _forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        lp_internal = torch.zeros((b,),device=device)

        md = self.transformer ### model dict


        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)

        x = tok_emb + pos_emb
        # x = tok_emb 
        if self.config.use_dropout:
            x = self.transformer.drop(x)


        lp_internal = torch.zeros((b,),device=device)
        #### the problem here is to insert context vector
        ####
        e = self.config.n_embd
        l = self.config.n_layer

        config = self.config

        ne = config.n_embd
        nh = config.n_head

        ### cpt is the post context at position 
        # cpt = torch.zeros((b,t,e),device=device) 
        cpt = torch.zeros((b,t,e),device=device)+ x
        x_init = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        ### need to concat a single prefix to avoid
        att_mask = torch.tril(att_mask, diagonal=1)
        # ### (tp,tc)
        nei = config.n_internal
        md = self.transformer
        # att_mask = att_mask.T

        def get_xpred(cpt):
            ### (b,tp,nh,nei)
            # xpred = cpt.matmul(wo).reshape((b,t,nh,ne))
            # xpred = md.lin_output(cpt).sigmoid().reshape((b,t,nh,ne))
            xpred = md.lin_output(cpt).reshape((b,t,nh,nei))
            return xpred


        def get_cpred(x,cpt):
            ### (b,tp,nei,nh,tc)
            ### no skip connection for the momenta
            cpred = md.lin_transit(cpt).reshape((b,t,nei,nh,1)).expand((b,t,nei,nh,t)) 
            cpred = cpred + md.lin_input(x).transpose(1,2)[:,None,:,None,:].expand((b,t,nei,nh,t)) 
            cpred = 0.5 * cpred
            return cpred

        for i in range(4):
            # dx = 0. 
            ### calculate the prediction of different models
            # wo = md.lin_output.weight

            ### tp for past t
            ### (b,tp,ne,tc)
            # xrep = x.transpose(1,2)[:,None].expand(b,t,ne,t)

            ## (b,tp,ne,nh,tc)
            ### (b,tp,ne,tc)
            # crep = cpt.transpose(1,2)[:,None].expand(b,t,ne,t)

            xpred = get_xpred(cpt)
            cpred = get_cpred(x,cpt)


            ### (b,tp,nh,tc)
            lp = 0.
            # lp += xpred.matmul(xrep) 
            lp += torch.einsum('bphe,bce->bphc',xpred,x.matmul(md.lin_internal_output.weight)) 
            lp += torch.einsum('bpehc,bce->bphc',cpred,cpt.matmul(md.lin_internal.weight))

            ### (b,tc,nh,tp)
            lp = lp.transpose(1,3)
            
            lp = lp.masked_fill(att_mask[None,:,None,:]==0,float('-inf'))

            ### (b,tc,nh,tp)
            # att = lp.reshape9.softmax((2,3))
            att = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)
            # 2,3))
            dcpt = torch.einsum('bchp,bpehc->bce', att, cpred).matmul(md.lin_internal.weight.T)
            cpt = cpt + 0.1*dcpt

        # xpred = md.lin_output(dcpt).reshape((b,t,nh,nei))
        xpred = get_xpred(cpt)
        xpred = xpred.matmul(md.lin_internal_output.weight.T).reshape((b,t,nh,ne))
        x = xpred

        x     = self.transformer.ln_f(x)            
# 

        logits = self.lm_head(x).log_softmax(-1).logsumexp(2)-math.log(nh)

        # print(logits.shape)
        # breakpoint()

        ### (b,t)
        if targets is not None:
            lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        else:
            lp_external = torch.zeros_like(logits[:,:,0])
        # ct = target_not_null.sum(1)
        return logits, lp_internal,lp_external                


        
class GPT_RNN01(GPT_01):
    '''
    sample a binary internal representation


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        config.g = 200
        # config.g = 10
        config.nrep=2

        # config.k = 2
        # config.g = 2

        # config.n_layer= 1

        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT_01,self).__init__()

        # config.block_size = config.block_size*

        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # print(config)
        # breakpoint()
        # config.nh = n_h = config.n_embd * 2
        config.nh = n_h = config.n_embd 
        ne = config.n_embd
        self.transformer = nn.ModuleDict(dict(
            v_input   = nn.Linear(config.n_embd, n_h),
            v_transit = nn.Linear(n_h,n_h,),
            v_output = nn.Linear(n_h,ne,),

            c_input   = nn.Linear(config.n_embd, n_h),
            c_transit = nn.Linear(n_h,n_h,),


            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # # k_query = nn.Linear(config.n_embd,config.k, ),
            # k_query = nn.Linear(config.n_embd,config.n_embd, ),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # h_enc = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # h_dec = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        ns = 5
        idx     = idx[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()

        if targets is not None:
            target_not_null = targets != self.config.mask_index 
            targets = targets[None].expand((ns,b,t)).reshape((ns*b,t))#.contiguous()
        else:
            target_not_null = torch.ones_like(idx)
        ct = target_not_null.sum(1)


        (logits, lp_internal, lp_external) = self._forward(idx,targets,gradient_only)

        ### (nsb,t)
        g = self.config.g
        k = self.config.k
        lp_k = k*math.log(1./t) + k*math.log(1/g)  ### adding prior on pk and gk var
        lp_total   = lp_external.sum(1)  + lp_internal

        method = self.config.method
        logits = logits.reshape(((ns,b,t,-1)))

        lp_internal = lp_internal.reshape((ns,b))
        lp_total    = lp_total.reshape((ns,b,))
        lp_external = lp_external.reshape((ns,b,t))   

        ### sampling from posterior is difficult
        ### https://www.cs.ubc.ca/~schmidtm/Courses/540-W20/L30.pdf
        
        # loss_valid = -(lp_total.sum(-1).logsumexp(0) - math.log(ns))/ct.sum(0)

        ### use a lower bound by summing over the hidden variable
        # loss_valid = -(lp_total.logsumexp(0).sum(0))/ct.sum(0)

        ### use variational approximation from the samples
        loss_valid = -(lp_external.sum(-1).logsumexp(0) - math.log(ns)).sum(0)/ct.sum(0)

        ##(ns,b)        
        if 0:
            pass
        elif method in (8,10):
            ### re weighted sample using q()
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 

            reweight_lp  = lp_total + reweight_lp.detach()
            loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            # loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10
        elif method == 12:
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 
            reweight_lp  = lp_total + reweight_lp.detach()
            loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
        else:
            assert 0
        ### (b,v)
        logits = (lp_internal[:,:,None,None] + logits.log_softmax(-1)).logsumexp(0)

        return logits, loss_grad, loss_valid


    def _forward(self, idx, targets=None, gradient_only=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        lp_internal = torch.zeros((b,),device=device)

        md = self.transformer ### model dict

        # forward the GPT model itself
        
        nh = self.config.nh
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        h  = torch.zeros((b, t+1, nh),device=device).float()
        lp_internal = torch.zeros((b,),device=device)

        p_update = 0.5
        def get_h2():
            h2_prop = 0.
            # h2_prop = (md.v_input(tok_emb) + md.v_transit(h1) + h2)*0.5
            # h2_prop = (md.v_input(tok_emb) + md.v_transit(h1) + h1)*0.5
            # h2_prop = (md.v_input(tok_emb) + md.v_transit(h1))
            h2_prop = (md.v_input(tok_emb) + md.v_transit(h1)).sigmoid()
            # h2_prop  = (h2_prop  + h2)*0.5
            h2_prop  = (h2_prop  + h1)*0.5
            return h2_prop

        # for i in range(5):
        for i in range(10):

            h1     = h[ :, :-1]
            
            
            #### p_copy 
            # p_prop  = (p_copy * h1) + (1-p_copy) * p_rand 

            h2      = h[:,1:].float()
            
            h2_prop = get_h2()
            # h2_prop = h2_prop + torch.normal(0.,1.,size=h2_prop.shape,device=device)
            # h2_prop = h2_prop + torch.normal(0., 0.1, size=h2_prop.shape,device=device)
            # h2_prop = h2_prop + torch.normal(0., 0.05, size=h2_prop.shape,device=device)
            h2_prop = h2_prop + torch.normal(0., 0.1, size=h2_prop.shape,device=device)
            

            # # prop = (md.v_input(tok_emb) + md.v_transit(h1)).sigmoid()
            # updater = torch.rand_like(p_prop[:,:,:1])< p_update
            # h[:,1:] = h2_prop * updater + h2 * ~updater 
            
            h[:,1:] = h2_prop.detach()
            
            # lp_internal_d  = (p_prop.log()*h2 + (1-p_prop).log() * (1-h2))            
            # lp_internal_d  = (lp_internal_d*updater).sum(-1).sum(-1)
            # lp_internal   += lp_internal_d


        h1   = h[ :, :-1]
        h2   = h[:,1:].float()
        # h2_prop = (md.v_input(tok_emb) + md.v_transit(h1) + h1)*0.5
        h2_prop = get_h2()
        # h2_prop = (md.v_input(tok_emb) + md.v_transit(h1) + h2)*0.5

        # ### (b,t)
        # h1   = h[ :, :-1]
        # p_copy = (md.c_input(tok_emb) + md.c_transit(h1)).sigmoid()
        # p_rand = (md.v_input(tok_emb) + md.v_transit(h1)).sigmoid()
        # p_copy = (md.c_input(tok_emb) + md.c_input(h1)).sigmoid()
        # p_rand = (md.v_input(tok_emb) + md.v_transit(h1)).sigmoid()
        
        lp_internal = -(h2_prop - h2).square().sum(-1).sum(-1)
        ### p_copy 
        # p_prop      = (p_copy * h1) + (1-p_copy) * p_rand 
        # lp_internal = (p_prop.log()*h2 + (1-p_prop).log() * (1-h2)).sum(-1).sum(-1)
        # lp_internal = lp_internal


        ### (b,t,ne)
        
        # target_not_null = targets != self.config.mask_index 

        # lp_internal = torch.zeros((b,),device=device)

        x = md.v_output(h2)
        # x      = self.transformer.ln_f(x)            
        logits = self.lm_head(x)

        ### (b,t)
        if targets is not None:
            lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        else:
            lp_external = torch.zeros_like(logits[:,:,0])
        # ct = target_not_null.sum(1)
        return logits, lp_internal,lp_external




class GPT_GRU(GPT_01):


    def __init__(self, config):
        super(GPT_01, self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        config.r = r = 4
        print(config.block_size)
        print(id(config))
        config.block_size = config.block_size*r
        print(config.block_size)

        self.config = config
        # super().__init__(config)

        D = config.n_layer
        Z = config.n_embd
        config.D = D
        config.Z = Z

        config.vocab_size = config.vocab_size+1  
        # self.v_input = nn.Linear(config.n_embd, )
        self.transformer = nn.ModuleDict(dict(

            wte = nn.Embedding(config.vocab_size+1, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # rnn
            rnn = nn.GRU(Z, Z, D, batch_first=True,),
            # wre = nn.Embedding(config.r, config.n_embd),
            # h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            
            # v_lin = nn.Linear(config.n_embd ,config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size+1, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # self.rnn  



        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        # self
        return 



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)


        x = self.transformer.drop(tok_emb + pos_emb)


        # for block in self.transformer.h:
        #     x = block(x)

        D = self.config.D
        Z = self.config.Z
        h0 = torch.zeros((b,D,Z),device=device).transpose(0,1).contiguous()
        # tok_emb_par = torch.cat([torch.zeros((B*M,1,Z),device=device), tok_emb[:,:-1]],dim=1)
        # ##(BM,T,Z)
        output_emb, hn = self.transformer.rnn(x, h0 )
        # self.transformer.ln_f

        x = self.transformer.ln_f(output_emb)
        # x = output_emb

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss



class GGRU02(GPT_01):


    def __init__(self, config):
        super(GPT_01, self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        config.r = r = 4
        print(config.block_size)
        print(id(config))
        config.block_size = config.block_size*r
        print(config.block_size)

        self.config = config
        # super().__init__(config)

        D = config.n_layer
        Z = config.n_embd
        config.D = D
        config.Z = Z

        config.vocab_size = config.vocab_size+1  
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size+1, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # rnn
            rnn = nn.GRU(Z, Z, D, batch_first=True,),
            kmat = (nn.Linear(Z,Z, bias=False)),
            vmat = (nn.Linear(Z,Z, bias=False)),
            # wre = nn.Embedding(config.r, config.n_embd),
            # h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            
            # v_lin = nn.Linear(config.n_embd ,config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size+1, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying



        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        # self
        return 



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)


        x = tok_emb + pos_emb
        # x = self.transformer.drop(x)   ### (b,t,e)
        x_emb = x


        # for block in self.transformer.h:
        #     x = block(x)

        D = self.config.D
        Z = self.config.Z
        h0 = torch.zeros((b,D,Z),device=device).transpose(0,1).contiguous()


        # self.device= device
        # tok_emb_par = torch.cat([torch.zeros((B*M,1,Z),device=device), tok_emb[:,:-1]],dim=1)
        # ##(BM,T,Z)
        md = self.transformer ### (moduleDict)

        # # h0 = torch.zeros((B,D,Z),device=self.device).transpose(0,1).contiguous()
        v = self.config.vocab_size
        hnext = torch.zeros((b,D,Z),device=device).transpose(0,1).contiguous()
        # tok_emb_par = torch.cat([torch.zeros((b,1,Z),device=device)],dim=1)
        # x_emb[:,0:1]
        # ##(BM,T,Z)
        act_emb = torch.ones((b,1,Z),device=device)/ Z
        lp_act = torch.zeros((b,t),device=device)
        logits = torch.zeros((b,t,v+1),device=device)

        ct_b = torch.zeros((b,),device=device)
        loss_b = torch.zeros((b,),device=device)

        for t in range(t):
            h = hnext 

            ### inject the action into dependency
            # print(x_emb.shape)
            # print(act_emb.shape)




            ### (b,)
            target_not_null = (targets[:,t]!=-1)
            idx_b = torch.arange(b)

            # x_emb[:,t:t+1] = x_emb[:,t:t+1] + act_emb



            ### (b,1,z)
            act_emb, hnext = md.rnn( x_emb[:,t:t+1], h )
            act_emb = md.ln_f(act_emb)

            ### (b,t+2,z)
            ### action is kv operation of any vector including the 
            h_emb = torch.cat([x_emb[:,:t+1],act_emb],dim=1)
            ### (b,t+1)
            #### this is a xWy energy function
            lp_internal = torch.einsum('bTz,btz->bTt', md.ln_f(md.kmat(h_emb)),act_emb).squeeze(-1).log_softmax(-1)

            ### (b,t,z)
            # ret_emb = h_emb[idx_b, act,:][:,None]
            ret_emb_v_all = md.vmat(h_emb)
            xold = x_emb[:,t:t+1]                
            ### (b,t,z)
            xnew = (xold + (target_not_null[:,None,None] * ret_emb_v_all + ~target_not_null[:,None,None] *xold) ) 
            xnew = md.ln_f(xnew)

            logits_k = self.lm_head(xnew)

            ### t option 
            ### (b,t)
            # print(logits_k.shape)
            # print( targets[:,t:t+1].expand(b,t+1).shape)
            # breakpoint()
            lp_external = -F.cross_entropy(logits_k.transpose(1,2), targets[:,t:t+1].expand(b,t+2), ignore_index=-1,reduction='none')
            lp_internal = lp_internal
            lp_post     = lp_internal + lp_external


            # if 1:
            ### sample from posterior
            ### (b,t)
            ### expand the possibility of action
            ### ignore chains where targets are finished
            post = torch.distributions.Categorical(logits=lp_post)
            ### (b,)
            act = post.sample()
            lp_act_t = lp_post[idx_b,act]



            # xnew_act = xnew[idx_b, act,:][:,None]
            x_emb[:,t]     = xnew[idx_b, act,:]
            # breakpoint()
            logits[:,t,:]  = logits_k[idx_b,act,:]
            #.log_softmax(-1)
            loss_b = loss_b - lp_act_t * target_not_null
            ct_b   = ct_b + target_not_null

            # print(act_emb.shape)


        loss = (loss_b.sum()/ct_b.sum())
        # print(loss.shape)

        return logits, loss



'''
GGRU02



GPT_GRU
step 250: train loss 6.2685, val loss 6.3870
iter 250: loss 6.4440, time 3417.51ms, mfu 6.65%
iter 260: loss 6.3914, time 289.56ms, mfu 6.62%
iter 270: loss 6.3874, time 313.37ms, mfu 6.54%
iter 280: loss 6.3287, time 299.79ms, mfu 6.50%
iter 290: loss 6.0823, time 221.35ms, mfu 6.67%
iter 300: loss 6.2214, time 247.16ms, mfu 6.75%
iter 310: loss 6.3253, time 285.69ms, mfu 6.71%
iter 290: loss 6.0823, time 221.35ms, mfu 6.67%
iter 300: loss 6.2214, time 247.16ms, mfu 6.75%
iter 310: loss 6.3253, time 285.69ms, mfu 6.71%
iter 320: loss 6.2148, time 298.70ms, mfu 6.66%
iter 330: loss 6.4362, time 211.18ms, mfu 6.86%
iter 340: loss 6.3102, time 194.33ms, mfu 7.11%
iter 350: loss 6.3097, time 266.47ms, mfu 7.09%
iter 360: loss 6.2109, time 263.69ms, mfu 7.07%
iter 370: loss 6.5440, time 253.77ms, mfu 7.09%
iter 380: loss 6.2494, time 222.26ms, mfu 7.20%
iter 390: loss 6.4656, time 255.97ms, mfu 7.20%
iter 400: loss 6.3440, time 253.47ms, mfu 7.20%
iter 410: loss 6.4275, time 255.50ms, mfu 7.20%
iter 420: loss 6.4242, time 235.41ms, mfu 7.25%
iter 430: loss 6.4152, time 291.76ms, mfu 7.16%
iter 440: loss 6.2270, time 237.16ms, mfu 7.21%
iter 450: loss 6.2076, time 239.51ms, mfu 7.26%
iter 460: loss 6.2151, time 284.54ms, mfu 7.17%
iter 470: loss 6.3893, time 235.79ms, mfu 7.23%
iter 480: loss 6.3648, time 240.75ms, mfu 7.27%
iter 490: loss 6.4582, time 287.32ms, mfu 7.18%
step 500: train loss 6.3215, val loss 6.4999
iter 500: loss 6.4113, time 3071.81ms, mfu 6.52%
iter 510: loss 6.3862, time 223.60ms, mfu 6.69%
iter 520: loss 6.3828, time 296.69ms, mfu 6.64%
iter 530: loss 6.2244, time 233.18ms, mfu 6.76%
iter 540: loss 6.4128, time 239.85ms, mfu 6.84%
iter 550: loss 6.1900, time 225.02ms, mfu 6.97%
iter 560: loss 6.3864, time 253.32ms, mfu 7.00%
iter 570: loss 6.3414, time 238.58ms, mfu 7.07%
iter 580: loss 6.3688, time 292.97ms, mfu 6.98%
iter 590: loss 6.3706, time 198.02ms, mfu 7.21%
iter 600: loss 6.6097, time 270.94ms, mfu 7.16%
iter 610: loss 6.3662, time 265.69ms, mfu 7.14%
iter 620: loss 6.4944, time 254.77ms, mfu 7.14%
iter 630: loss 6.2286, time 179.70ms, mfu 7.45%
iter 640: loss 6.2262, time 289.82ms, mfu 7.33%
iter 650: loss 6.3870, time 230.06ms, mfu 7.40%
iter 940: loss 4.3829, time 264.38ms, mfu 7.26%
iter 950: loss 5.0370, time 275.00ms, mfu 7.20%
iter 960: loss 4.6188, time 186.87ms, mfu 7.46%
iter 970: loss 4.7217, time 249.72ms, mfu 7.44%
iter 980: loss 4.9550, time 315.16ms, mfu 7.28%
iter 990: loss 5.0286, time 305.55ms, mfu 7.15%
step 1000: train loss 4.7031, val loss 5.2330
saving checkpoint to out-shakespeare-word -GPT_GRU
iter 1000: loss 4.5306, time 3555.02ms, mfu 6.49%
iter 1010: loss 4.2995, time 218.97ms, mfu 6.68%
iter 1450: loss 4.5092, time 311.07ms, mfu 6.79%
iter 1460: loss 4.1906, time 312.55ms, mfu 6.70%
iter 1470: loss 4.6550, time 197.31ms, mfu 6.95%
iter 1480: loss 4.5293, time 205.07ms, mfu 7.15%
iter 1490: loss 4.4696, time 204.21ms, mfu 7.33%
step 1500: train loss 4.4083, val loss 5.0424
saving checkpoint to out-shakespeare-word -GPT_GRU
iter 1500: loss 4.3218, time 3238.42ms, mfu 6.66%
iter 1980: loss 4.4177, time 198.00ms, mfu 8.05%
iter 1990: loss 4.1405, time 202.10ms, mfu 8.15%
step 2000: train loss 4.2244, val loss 4.9733
saving checkpoint to out-shakespeare-word -GPT_GRU
iter 2000: loss 4.4170, time 3168.52ms, mfu 7.39%
step 4750: train loss 3.8208, val loss 4.9715


GPT_06_withnoise
iter 480: loss 4.5305, time 256.15ms, mfu 2.38%
iter 490: loss 4.8258, time 223.96ms, mfu 2.38%
step 500: train loss 4.5667, val loss 5.0177
saving checkpoint to out-shakespeare-word -GPT_06
iter 500: loss 4.7698, time 2726.69ms, mfu 2.16%
iter 1490: loss 3.8106, time 224.27ms, mfu 2.68%
step 1500: train loss 3.6710, val loss 4.7207
iter 1500: loss 4.1303, time 2118.29ms, mfu 2.44%
iter 1510: loss 3.7200, time 277.98ms, mfu 2.39%


GPT_06_nonoise
iter 490: loss 4.7474, time 297.94ms, mfu 1.80%
step 500: train loss 4.4803, val loss 4.9705
saving checkpoint to out-shakespeare-word -GPT_06
iter 500: loss 4.7048, time 3343.87ms, mfu 1.64%
iter 740: loss 4.3017, time 330.73ms, mfu 1.69%
step 750: train loss 4.1286, val loss 4.7974
saving checkpoint to out-shakespeare-word -GPT_06


GGRU02
iter 280: loss 6.1518, time 5877.69ms, mfu 0.29%
iter 290: loss 6.0164, time 5768.83ms, mfu 0.30%
iter 300: loss 6.2889, time 5912.09ms, mfu 0.30%
iter 310: loss 6.1400, time 5929.15ms, mfu 0.30%
iter 320: loss 5.9971, time 5796.65ms, mfu 0.30%
iter 330: loss 5.8516, time 8571.98ms, mfu 0.29%

step 1000: train loss 4.7509, val loss 5.3021
saving checkpoint to out-shakespeare-word 
iter 1000: loss 4.5941, time 104089.50ms, mfu 0.28%


### shakespear_char


GPT = GPT_01    

iter 470: loss 1.6655, time 34.62ms, mfu 11.58%
iter 480: loss 1.6237, time 34.54ms, mfu 11.50%
iter 490: loss 1.6062, time 34.49ms, mfu 11.43%
step 500: train loss 1.5356, val loss 1.7292
saving checkpoint to out-shakespeare-char
iter 500: loss 1.6059, time 3489.89ms, mfu 10.30%
iter 510: loss 1.6218, time 28.92ms, mfu 10.56%

iter 990: loss 1.3393, time 30.38ms, mfu 11.84%
 step 1000: train loss 1.2762, val loss 1.5216
saving checkpoint to out-shakespeare-char
iter 1000: loss 1.3399, time 3549.95ms, mfu 10.66%
iter 1010: loss 1.3393, time 30.52ms, mfu 10.82%

'''


'''
GPT_02 r2
step 500: train loss 1.5253, val loss 1.7187
saving checkpoint to out-shakespeare-char
iter 500: loss 1.5311, time 5268.98ms, mfu 15.69%

r4
iter 970: loss 1.2757, time 83.32ms, mfu 22.93%
iter 980: loss 1.2960, time 83.42ms, mfu 22.96%
iter 990: loss 1.2594, time 83.53ms, mfu 22.98%
step 1000: train loss 1.2613, val loss 1.5919
saving checkpoint to out-shakespeare-char
iter 1000: loss 1.3092, time 9242.22ms, mfu 20.71%
iter 1010: loss 1.2633, time 83.11ms, mfu 20.97%


step 1000: train loss 1.2638, val loss 1.5775


step 1000: train loss 1.2668, val loss 1.5762




iter 980: loss 1.7136, time 83.51ms, mfu 22.90%
iter 990: loss 1.6945, time 84.18ms, mfu 22.91%
step 1000: train loss 1.6533, val loss 1.8427
saving checkpoint to out-shakespeare-char
iter 1000: loss 1.6830, time 9180.28ms, mfu 20.64%
iter 1010: loss 1.7260, time 82.84ms, mfu 20.91%
iter 1020: loss 1.7125, time 83.67ms, mfu 21.13%

'''


'''
### shakespear_word


GPT = GPT_01    
iter 240: loss 4.2459, time 37.79ms, mfu 12.78%
step 250: train loss 3.9874, val loss 4.8423
saving checkpoint to out-shakespeare-word 
iter 250: loss 4.1194, time 10027.09ms, mfu 11.51%


iter 490: loss 3.4173, time 37.84ms, mfu 12.64%
step 500: train loss 3.1934, val loss 4.8607
iter 500: loss 3.4633, time 5928.70ms, mfu 11.38%

iter 980: loss 2.3854, time 37.60ms, mfu 12.65%
iter 990: loss 2.3591, time 37.53ms, mfu 12.67%
step 1000: train loss 1.9279, val loss 5.5394
iter 1000: loss 2.3939, time 5838.57ms, mfu 11.41%

'''


'''
GPT_02 

lrE-3

B256
r3
iter 230: loss 4.0536, time 181.27ms, mfu 8.26%
iter 240: loss 4.1023, time 181.23ms, mfu 8.26%
step 250: train loss 3.9085, val loss 4.8213
saving checkpoint to out-shakespeare-word 
iter 250: loss 3.8696, time 12514.82ms, mfu 7.44%
iter 260: loss 3.8855, time 175.90ms, mfu 7.55%


iter 490: loss 3.1722, time 182.17ms, mfu 8.15%
step 500: train loss 2.9196, val loss 5.1402
iter 500: loss 3.0291, time 12020.70ms, mfu 7.35%

iter 740: loss 2.3153, time 181.57ms, mfu 8.15%
step 750: train loss 2.0096, val loss 5.9016
iter 750: loss 2.3473, time 12041.70ms, mfu 7.34%

iter 990: loss 1.6498, time 180.81ms, mfu 8.15%
step 1000: train loss 1.2633, val loss 6.6371
iter 1000: loss 1.6263, time 12017.25ms, mfu 7.35%



lrE-3
r3 with weighting

iter 210: loss 4.2029, time 189.29ms, mfu 7.96%
iter 220: loss 3.9756, time 188.51ms, mfu 7.96%
iter 230: loss 3.9498, time 189.01ms, mfu 7.96%
iter 240: loss 3.9361, time 188.63ms, mfu 7.96%
step 250: train loss 3.9163, val loss 4.8544
saving checkpoint to out-shakespeare-word 
iter 250: loss 3.9908, time 12525.77ms, mfu 7.17%
iter 260: loss 3.9991, time 184.67ms, mfu 7.27%

lrE-4

step 250: train loss 5.1145, val loss 5.4790
saving checkpoint to out-shakespeare-word 
iter 250: loss 5.1890, time 12508.12ms, mfu 7.17%
iter 260: loss 5.2370, time 184.85ms, mfu 7.27%
step 500: train loss 4.4270, val loss 4.9995
step 750: train loss 4.0626, val loss 4.8513
step 1000: train loss 3.8183, val loss 4.7621
step 1250: train loss 3.5932, val loss 4.7719
step 1500: train loss 3.3778, val loss 4.8803



02_r4
B256
lrE4-E4
step 1500: train loss 3.7783, val loss 4.7970
step 2000: train loss 3.4958, val loss 4.7612

lrE4-E5
step 250: train loss 5.0626, val loss 5.4245
step 500: train loss 4.3433, val loss 5.0199
step 750: train loss 3.9291, val loss 4.8071
step 1000: train loss 3.6464, val loss 4.8130
iter 1000: loss 3.6232, time 8308.73ms, mfu 6.38%


02_r5
step 500: train loss 4.3405, val loss 5.0079
step 750: train loss 3.9389, val loss 4.8104
step 1000: train loss 3.6365, val loss 4.7966


03_r4
step 250: train loss 4.9504, val loss 5.3720
step 500: train loss 4.2411, val loss 4.9700
step 750: train loss 3.8260, val loss 4.7953

GPT_01
lrE4-E5
step 750: train loss 3.9766, val loss 4.7861
step 1000: train loss 3.7364, val loss 4.7123


step 250: train loss 4.2056, val loss 5.0620
step 500: train loss 3.3775, val loss 5.0759
 

'''



'''
GPT05
step 500: train loss 1.6412, val loss 1.8312
step 750: train loss 1.4219, val loss 1.6335
step 1000: train loss 1.3153, val loss 1.5379
step 1250: train loss 1.2429, val loss 1.5137


GPT01
step 500: train loss 1.5356, val loss 1.7292
step 750: train loss 1.3693, val loss 1.5967


GPT05
step 250: train loss 4.3171, val loss 5.0140
iter 490: loss 4.2497, time 117.39ms, mfu 2.57%
step 500: train loss 3.7677, val loss 4.9310
iter 740: loss 3.3328, time 119.89ms, mfu 2.58%
step 750: train loss 3.2992, val loss 4.9362
iter 990: loss 3.1626, time 120.66ms, mfu 2.56%
step 1000: train loss 2.9482, val loss 5.1283


GPT01 E3
iter 240: loss 4.2298, time 122.03ms, mfu 2.56%
step 250: train loss 4.3144, val loss 5.0122
iter 490: loss 4.1698, time 118.47ms, mfu 2.56%
step 500: train loss 3.6980, val loss 4.8768
iter 740: loss 3.2621, time 119.94ms, mfu 2.57%
step 750: train loss 3.2034, val loss 4.9740
iter 990: loss 3.0816, time 116.09ms, mfu 2.60%
step 1000: train loss 2.8090, val loss 5.2355


GPT01 E4
iter 240: loss 5.2015, time 112.95ms, mfu 2.65%
step 250: train loss 5.2901, val loss 5.5895
iter 490: loss 5.1227, time 120.12ms, mfu 2.54%
step 500: train loss 4.6771, val loss 5.1282
iter 740: loss 4.2987, time 117.23ms, mfu 2.58%
step 750: train loss 4.3765, val loss 4.9851
iter 990: loss 4.3548, time 120.56ms, mfu 2.58%
step 1000: train loss 4.1101, val loss 4.9137
iter 1240: loss 3.9425, time 119.74ms, mfu 2.56%
step 1250: train loss 3.9641, val loss 4.7947
iter 1490: loss 4.2959, time 118.81ms, mfu 2.54%
step 1500: train loss 3.8475, val loss 4.7376
iter 1740: loss 3.8988, time 124.47ms, mfu 2.57%
step 1750: train loss 3.6954, val loss 4.7223
iter 1990: loss 3.6848, time 119.38ms, mfu 2.56%
step 2000: train loss 3.6057, val loss 4.6854
iter 2990: loss 3.4369, time 119.23ms, mfu 2.54%
step 3000: train loss 3.3069, val loss 4.6511

GPT05_E4
iter 240: loss 5.1957, time 113.12ms, mfu 2.65%
step 250: train loss 5.2818, val loss 5.5868
iter 490: loss 5.1318, time 113.43ms, mfu 2.62%
step 500: train loss 4.6836, val loss 5.1385
iter 740: loss 4.3240, time 123.07ms, mfu 2.57%
step 750: train loss 4.3989, val loss 5.0014
iter 990: loss 4.3875, time 122.02ms, mfu 2.52%
step 1000: train loss 4.1733, val loss 4.9609
step 1250: train loss 4.0152, val loss 4.8452
iter 1490: loss 4.3213, time 114.37ms, mfu 2.60%
step 1500: train loss 3.8962, val loss 4.7701
iter 1740: loss 3.9220, time 116.07ms, mfu 2.53%
step 1750: train loss 3.7471, val loss 4.7695
iter 1990: loss 3.7334, time 123.85ms, mfu 2.50%
step 2000: train loss 3.6682, val loss 4.7271
iter 2990: loss 3.4788, time 113.94ms, mfu 2.54%
step 3000: train loss 3.3999, val loss 4.6811



'''
# GPT = GPT_02    
# GPT = GPT_03 
# GPT = GPT_01    
# GPT = GPT_07    
# GPT = GPT_08    
# GPT = GPT_10    
GPT = GPT_RNN01
GPT = GPT_11    
GPT = GPT_13    
GPT = CM01
# GPT = GPT_01    
# GPT = GPT_09    
# GPT = GPT_05  
# GPT = GPT_GRU
# GPT = GPT_06    
# GPT = GGRU02
# 
# torch._dynamo.config.suppress_errors = True
