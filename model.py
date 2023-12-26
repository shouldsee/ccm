"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
from dataclasses import dataclass

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
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
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


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    optimizer: str='adamw'

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # config.dropout = 0.0
        # print (config)

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

        return logits, loss

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
        if self.config.optimizer=='adamw':
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
        elif self.config.optimizer =='rmsprop':
            optimizer = torch.optim.RMSprop(optim_groups, lr=learning_rate,)
        elif self.config.optimizer=='adam':
            optimizer = torch.optim.Adam(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused Adam: {use_fused}")
        else:
            raise NotImplementedError(self.config.optimizer)


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
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
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




import torch.nn as nn


# class CCM01(nn.Module):
class CCM01(GPT):
    '''
    compress the internal representation to discrete representations
    g200 


iter 1480: loss 4.1011, time 94.83ms, mfu 3.64%
iter 1490: loss 4.1817, time 94.83ms, mfu 3.65%
step 1500: train loss 4.1789, val loss 5.0421
saving checkpoint to out-shakespeare-word

step 2000: train loss 4.0252, val loss 5.0605
step 1500: train loss 4.0410, val loss 4.9773



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
        # config.n_internal = config.n_embd//16
        config.n_internal = config.n_embd//4


        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2
        config.method = -1
        config.n_head = 1

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__(config)

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
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            lin_output   = nn.Linear(ne, nh*nei,bias=config.bias),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit   = nn.Linear(ne,nh*nei,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input     = nn.Linear(ne,nei,bias=config.bias),
            lin_input_2   = nn.Linear(ne,ne,bias=config.bias),
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
        cpt      = torch.zeros((b,t,e),device=device)+ x
        x_init   = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### need to concat a single prefix to avoid
        # att_mask = torch.tril(att_mask, diagonal=1)

        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer

        # ### (tp,tc)        
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
            # cpred = cpred.sigmoid()
            return cpred

        for i in range(4):
            xpred = get_xpred(cpt)
            cpred = get_cpred(x,cpt)

            ### (b,tp,nh,tc)
            lp = 0.
            # lp += xpred.matmul(xrep) 
            lp += torch.einsum('bphe,bce->bphc', xpred, x.matmul(md.lin_internal_output.weight)) 
            lp += torch.einsum('bpehc,bce->bphc',cpred, cpt.matmul(md.lin_internal.weight))

            ### (b,tc,nh,tp)
            lp = lp.transpose(1,3)
            
            lp = lp.masked_fill(att_mask[None,:,None,:]==0,float('-inf'))

            ### (b,tc,nh,tp)
            # att = lp.reshape.softmax((2,3))
            att = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)
            # 2,3))
            # breakpoint()    
            dcpt = torch.einsum('bchp,bpehc->bce', att, cpred).matmul(md.lin_internal.weight.T)
            cpt = cpt + 0.1*dcpt

        # xpred = md.lin_output(dcpt).reshape((b,t,nh,nei))
        xpred = get_xpred(cpt)
        xpred = xpred.matmul(md.lin_internal_output.weight.T).reshape((b,t,nh,ne))
        x     = xpred

        x     = self.transformer.ln_f(x)            

        logits= self.lm_head(x).log_softmax(-1).logsumexp(2)-math.log(nh)


        # if targets is not None:
        #     lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        #     # lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        # else:
        #     lp_external = torch.zeros_like(logits[:,:,0])

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        else:
            loss  = None
        return logits, loss         

    def forward_random(self, idx, targets=None, gradient_only=True):
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
        if 1:
            ### re weighted sample using q()
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 

            reweight_lp  = lp_total + reweight_lp.detach()
            # loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10

        ### (b,v)
        logits = (lp_internal[:,:,None,None] + logits.log_softmax(-1)).logsumexp(0)

        return logits, loss_grad, loss_valid



import torch.nn as nn


# class CCM01(nn.Module):
class CCM02(GPT):
    '''

calculating the rnn transition to use atttended vector



iter 1480: loss 4.1011, time 94.83ms, mfu 3.64%
iter 1490: loss 4.1817, time 94.83ms, mfu 3.65%
step 1500: train loss 4.1789, val loss 5.0421

step 2000: train loss 4.0252, val loss 5.0605


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
        config.method = -1
        config.n_head = 1

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__(config)

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
        cpt      = torch.zeros((b,t,e),device=device)+ x
        x_init   = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### need to concat a single prefix to avoid
        # att_mask = torch.tril(att_mask, diagonal=1)

        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer

        # ### (tp,tc)        
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
            # cpred = cpred.sigmoid()
            return cpred

        for i in range(4):
            xpred = get_xpred(cpt)
            cpred = get_cpred(x,cpt)

            ### (b,tp,nh,tc)
            lp = 0.
            # lp += xpred.matmul(xrep) 
            lp += torch.einsum('bphe,bce->bphc', xpred, x.matmul(md.lin_internal_output.weight)) 
            lp += torch.einsum('bpehc,bce->bphc',cpred, cpt.matmul(md.lin_internal.weight))

            ### (b,tc,nh,tp)
            lp = lp.transpose(1,3)
            
            lp = lp.masked_fill(att_mask[None,:,None,:]==0,float('-inf'))

            ### (b,tc,nh,tp)
            # att = lp.reshape.softmax((2,3))
            att = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)
            # 2,3))
            # breakpoint()    
            dcpt = torch.einsum('bchp,bpehc->bce', att, cpred).matmul(md.lin_internal.weight.T)
            cpt = cpt + 0.1*dcpt

        # xpred = md.lin_output(dcpt).reshape((b,t,nh,nei))
        xpred = get_xpred(cpt)
        xpred = xpred.matmul(md.lin_internal_output.weight.T).reshape((b,t,nh,ne))
        x     = xpred

        x     = self.transformer.ln_f(x)            

        logits= self.lm_head(x).log_softmax(-1).logsumexp(2)-math.log(nh)


        # if targets is not None:
        #     lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        #     # lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        # else:
        #     lp_external = torch.zeros_like(logits[:,:,0])

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        else:
            loss  = None
        return logits, loss         

    def forward_random(self, idx, targets=None, gradient_only=True):
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
        if 1:
            ### re weighted sample using q()
            ##(ns,b)
            reweight_lp  = lp_external.sum(2).log_softmax(0) 

            reweight_lp  = lp_total + reweight_lp.detach()
            # loss_grad  = -(reweight_lp.logsumexp(0).sum(0))/ct.sum(0)
            loss_grad  = -(reweight_lp.mean(0).sum(0))/ct.sum(0)
            # loss_grad = -( lp_external.sum(1) + lp_external.sum(1).detach() * lp_internal).sum(0)/ct.sum(0)# * 10

        ### (b,v)
        logits = (lp_internal[:,:,None,None] + logits.log_softmax(-1)).logsumexp(0)

        return logits, loss_grad, loss_valid


import torch.nn as nn


# class CCM01(nn.Module):
class CCM02(GPT):
    '''
    CCM02 insert an attention module in the token genration of the underlying machine


iter 1480: loss 4.1011, time 94.83ms, mfu 3.64%
iter 1490: loss 4.1817, time 94.83ms, mfu 3.65%
step 1500: train loss 4.1789, val loss 5.0421
saving checkpoint to out-shakespeare-word

step 2000: train loss 4.0252, val loss 5.0605
step 1500: train loss 4.0410, val loss 4.9773

step 1250: train loss 4.0988, val loss 4.9496

step 1500: train loss 3.9799, val loss 4.9250
step 1750: train loss 3.8957, val loss 4.9027


2 layer
step 1750: train loss 3.8957, val loss 4.9026


3 layer
step 1750: train loss 3.9735, val loss 4.9492
step 2000: train loss 3.8794, val loss 4.9326


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
        # config.n_internal = config.n_embd//16
        config.n_internal = config.n_embd//4


        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2
        config.method = -1
        config.n_head = 1

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__(config)

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
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            lin_output   = nn.Linear(ne, nh*nei,bias=config.bias),
            # lin_output_2   = nn.Linear(ne, nh*nei,bias=config.bias),
            lin_output_ctx   = nn.Linear(ne, nh*nei,bias=config.bias),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,nh*nei,bias=config.bias),
            lin_transit_ctx = nn.Linear(ne,nh*nei,bias=config.bias),
            lin_cpt_attention = nn.Linear(ne,ne,bias=config.bias),
            lin_cpt_attention_2 = nn.Linear(ne,ne,bias=config.bias),
            lin_ctx = nn.Linear(ne,ne,bias=config.bias),
            lin_ctx_2 = nn.Linear(ne,ne,bias=config.bias),
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
        cpt      = torch.zeros((b,t,e),device=device)+ x
        x_init   = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### need to concat a single prefix to avoid
        # att_mask = torch.tril(att_mask, diagonal=1)

        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        def get_xpred(cpt,ctx):
            ### (b,tp,nh,nei)
            # xpred = cpt.matmul(wo).reshape((b,t,nh,ne))
            # xpred = md.lin_output(cpt).sigmoid().reshape((b,t,nh,ne))
            #### adding a context
            xpred = (md.lin_output(cpt) + md.lin_output_ctx(ctx)).reshape((b,t,nh,nei))
            return xpred


        def get_cpred(x,cpt,ctx):
            ### (b,tp,nei,nh,tc)
            ### no skip connection for the momenta
            cpred = (md.lin_transit(cpt) + md.lin_transit_ctx(ctx)).reshape((b,t,nei,nh,1)).expand((b,t,nei,nh,t)) 
            cpred = cpred + md.lin_input(x).transpose(1,2)[:,None,:,None,:].expand((b,t,nei,nh,t)) 
            cpred = 0.5 * cpred
            # cpred = cpred.sigmoid()
            return cpred

        def get_ctx_2(cpt):
            return cpt
            cpt = (cpt)
            watt = md.lin_cpt_attention_2.weight
            lp  = cpt.matmul(watt).matmul(cpt.transpose(1,2))
            lp  = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            att = lp.softmax(2)

            ctx = att.matmul( cpt.matmul(watt.T) )
            ctx = md.lin_ctx_2(ctx).relu()
            return (ctx + cpt) * 0.5 

        def get_ctx(cpt):
            ### (b,tc,tp)
            cpt = get_ctx_2(cpt)
            lp  = cpt.matmul(md.lin_cpt_attention.weight).matmul(cpt.transpose(1,2))
            lp  = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            att = lp.softmax(2)

            ctx = att.matmul( cpt.matmul(md.lin_cpt_attention.weight.T) )
            ctx = md.lin_ctx(ctx).relu()
            return (ctx + cpt)*0.5

        for i in range(4):
            ### (b,tp,e)

            # cptc = cpt
            # for i in range(3):
            #     cptc = cptc + 0.1 *  get_ctx(cpt)
            ctx = get_ctx(cpt)
            
            # .matmul()
            xpred = get_xpred(cpt,ctx)
            cpred = get_cpred(x,cpt,ctx)

            ### (b,tp,nh,tc)
            lp = 0.
            # lp += xpred.matmul(xrep) 
            lp += torch.einsum('bphe,bce->bphc', xpred, x.matmul(md.lin_internal_output.weight)) 
            lp += torch.einsum('bpehc,bce->bphc',cpred, cpt.matmul(md.lin_internal.weight))

            ### (b,tc,nh,tp)
            lp = lp.transpose(1,3)
            
            lp = lp.masked_fill(att_mask[None,:,None,:]==0,float('-inf'))

            ### (b,tc,nh,tp)
            # att = lp.reshape.softmax((2,3))
            att = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)
            # 2,3))
            # breakpoint()    
            dcpt = torch.einsum('bchp,bpehc->bce', att, cpred).matmul(md.lin_internal.weight.T)
            cpt = cpt + 0.1*dcpt

        # xpred = md.lin_output(dcpt).reshape((b,t,nh,nei))

        ctx = get_ctx(cpt)
        
        # .matmul()
        # xpred = get_xpred(cpt,ctx)
        # xpred = (md.lin_output_2(cpt)).reshape((b,t,nh,nei))
            
        xpred = get_xpred(dcpt,ctx)
            # cpred = get_cpred(x,cpt,ctx)

        xpred = xpred.matmul(md.lin_internal_output.weight.T).reshape((b,t,nh,ne))
        x     = xpred

        x     = self.transformer.ln_f(x)            

        logits= self.lm_head(x).log_softmax(-1).logsumexp(2)-math.log(nh)


        # if targets is not None:
        #     lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        #     # lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        # else:
        #     lp_external = torch.zeros_like(logits[:,:,0])

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        else:
            loss  = None
        return logits, loss         


class CCM03(GPT):
    '''
    CCM02 extend the token generation with a key value vector pair


iter 1480: loss 4.1011, time 94.83ms, mfu 3.64%
iter 1490: loss 4.1817, time 94.83ms, mfu 3.65%
step 1500: train loss 4.1789, val loss 5.0421
saving checkpoint to out-shakespeare-word

step 2000: train loss 4.0252, val loss 5.0605
step 1500: train loss 4.0410, val loss 4.9773

step 1250: train loss 4.0988, val loss 4.9496

step 1500: train loss 3.9799, val loss 4.9250
step 1750: train loss 3.8957, val loss 4.9027


### tying k=v
step 1250: train loss 4.1273, val loss 4.9890

###separate kv
step 1250: train loss 4.1374, val loss 4.9755
step 1500: train loss 4.0350, val loss 4.9298

### k=v, but double the k dimension
step 1250: train loss 4.1478, val loss 4.9866


### g400, double k dimension
# step 1250: train loss 4.1483, val loss 4.9636

step 1250: train loss 4.1472, val loss 4.9646

step 1250: train loss 4.1489, val loss 4.9862


### maybe overfitting?

    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        config.g = 400
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        # config.n_internal = config.n_embd//16
        config.n_internal = config.n_embd//4


        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2
        config.method = -1
        config.n_head = 1

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__(config)

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
        # config.g = 200

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # k_query = nn.Linear(config.n_embd,config.k, ),
            k_vects = nn.Embedding(config.g, ne),
            v_vects = nn.Embedding(config.g, ne),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            lin_output   = nn.Linear(ne, nh*nei,bias=config.bias),
            lin_output_ctx   = nn.Linear(ne, nh*nei,bias=config.bias),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,nh*nei,bias=config.bias),
            lin_transit_ctx = nn.Linear(ne,nh*nei,bias=config.bias),
            lin_cpt_attention = nn.Linear(ne,ne,bias=config.bias),
            lin_cpt_attention_2 = nn.Linear(ne,ne,bias=config.bias),
            lin_cpt_attention_3 = nn.Linear(ne,ne,bias=config.bias),
            # lin_ctx = nn.Linear(ne,ne,bias=config.bias),
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
        cpt      = torch.zeros((b,t,e),device=device)+ x
        x_init   = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### need to concat a single prefix to avoid
        # att_mask = torch.tril(att_mask, diagonal=1)

        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        def get_xpred(cpt,ctx):
            ### (b,tp,nh,nei)
            # xpred = cpt.matmul(wo).reshape((b,t,nh,ne))
            # xpred = md.lin_output(cpt).sigmoid().reshape((b,t,nh,ne))
            #### adding a context
            xpred = (md.lin_output(cpt) + md.lin_output_ctx(ctx)).reshape((b,t,nh,nei))
            return xpred


        def get_cpred(x,cpt,ctx):
            ### (b,tp,nei,nh,tc)
            ### no skip connection for the momenta
            cpred = (md.lin_transit(cpt) + md.lin_transit_ctx(ctx)).reshape((b,t,nei,nh,1)).expand((b,t,nei,nh,t)) 
            cpred = cpred + md.lin_input(x).transpose(1,2)[:,None,:,None,:].expand((b,t,nei,nh,t)) 
            cpred = 0.5 * cpred
            # cpred = cpred.sigmoid()
            return cpred

        # def get_ctx(cpt):
        #     ### (nk,ne)
        #     kv = md.k_vects.weight
        #     vv = md.v_vects.weight

        #     ### (b,tc,nk)
        #     lp  = cpt.matmul(md.lin_cpt_attention.weight).matmul(kv.T)
        #     # lp  = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
        #     att = lp.softmax(2)

        #     # ctx = att.matmul( vv.matmul(md.lin_cpt_attention.weight.T) )
        #     ctx = att.matmul( vv[:,:ne])
        #     # .matmul(md.lin_cpt_attention.weight.T) )
        #     return ctx


        def get_ctx(cpt):
            ### (nk,ne)
            kv = md.k_vects.weight

            ### (b,tc,nk)
            lp  = cpt.matmul(md.lin_cpt_attention.weight).matmul(kv.T)
            att = lp.softmax(2)
            ctx = att.matmul( kv.matmul(md.lin_cpt_attention.weight.T) )
            return ctx            


        def get_ctx(cpt):
            ctx = cpt
            ### (nk,ne)
            kv = md.k_vects.weight
            # vv = md.v_vects.weight
            ctx = 0.
            ### (b,tc,nk)
            lp  = cpt.matmul(md.lin_cpt_attention.weight).matmul(kv.T)
            att = (lp).softmax(2)
            ctx += (att.matmul( kv.matmul(md.lin_cpt_attention.weight.T) ))

            # lp  = cpt.matmul(md.lin_cpt_attention_2.weight).matmul(kv.T)
            # att = (lp).softmax(2)
            # ctx += (att.matmul( kv.matmul(md.lin_cpt_attention_2.weight.T) ))

            # lp  = cpt.matmul(md.lin_cpt_attention_3.weight).matmul(kv.T)
            # att = (lp).softmax(2)
            # ctx += (att.matmul( kv.matmul(md.lin_cpt_attention_3.weight.T) ))
            return ctx    


        # for i in range(8):
        for i in range(4):
            ### (b,tp,e)

            # cptc = cpt
            # for i in range(3):
            #     cptc = cptc + 0.1 *  get_ctx(cpt)
            ctx = get_ctx(cpt)
            
            # .matmul()
            xpred = get_xpred(cpt,ctx)
            cpred = get_cpred(x,cpt,ctx)

            ### (b,tp,nh,tc)
            lp = 0.
            # lp += xpred.matmul(xrep) 
            lp += torch.einsum('bphe,bce->bphc', xpred, x.matmul(md.lin_internal_output.weight)) 
            lp += torch.einsum('bpehc,bce->bphc',cpred, cpt.matmul(md.lin_internal.weight))

            ### (b,tc,nh,tp)
            lp = lp.transpose(1,3)
            
            lp = lp.masked_fill(att_mask[None,:,None,:]==0,float('-inf'))

            ### (b,tc,nh,tp)
            # att = lp.reshape.softmax((2,3))
            att = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)
            # 2,3))
            # breakpoint()    
            dcpt = torch.einsum('bchp,bpehc->bce', att, cpred).matmul(md.lin_internal.weight.T)
            cpt = cpt + 0.1*dcpt
            # cpt = cpt + torch.normal(0,0.01,cpt.shape,device=device)

        # xpred = md.lin_output(dcpt).reshape((b,t,nh,nei))

        ctx = get_ctx(cpt)
        
        # .matmul()
        xpred = get_xpred(cpt,ctx)
            # cpred = get_cpred(x,cpt,ctx)

        xpred = xpred.matmul(md.lin_internal_output.weight.T).reshape((b,t,nh,ne))
        x     = xpred
        x     = self.transformer.ln_f(x)            

        logits= self.lm_head(x).log_softmax(-1).logsumexp(2)-math.log(nh)


        # if targets is not None:
        #     lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        #     # lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        # else:
        #     lp_external = torch.zeros_like(logits[:,:,0])

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        else:
            loss  = None
        return logits, loss         

class CCM04(GPT):
    '''
    CCM04 extend the hidden state to have pre and post context 
    
step 2000: train loss 3.9162, val loss 4.9884
step 1250: train loss 4.2499, val loss 5.0134
step 2000: train loss 3.9712, val loss 4.9935

    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        config.g = 400
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        # config.n_internal = config.n_embd//16
        config.n_internal = config.n_embd//4
        config.n_internal = config.n_embd//4


        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2
        config.method = -1
        config.n_head = 1

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__(config)

        # config.block_size = config.block_size*

        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        print(config)
        # breakpoint()
        ne  = config.n_embd
        nh  = config.n_head
        nei = config.n_internal
        # config.g = 200

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # k_query = nn.Linear(config.n_embd,config.k, ),
            k_vects = nn.Embedding(config.g, ne),
            v_vects = nn.Embedding(config.g, ne),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            lin_init     = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, nh*nei,bias=config.bias),
            lin_output_ctx   = nn.Linear(ne, nh*nei,bias=config.bias),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,nh*nei,bias=config.bias),
            lin_transit_ctx = nn.Linear(ne,nh*nei,bias=config.bias),
            lin_cpt_attention = nn.Linear(ne,ne,bias=config.bias),
            lin_cpt_attention_2 = nn.Linear(ne,ne,bias=config.bias),
            lin_cpt_attention_3 = nn.Linear(ne,ne,bias=config.bias),
            # lin_ctx = nn.Linear(ne,ne,bias=config.bias),
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
        cpt      = torch.zeros((b,t,e),device=device)  + md.lin_init(x)
        ### context before 
        cbt      = torch.zeros((b,t,e),device=device)  + md.lin_init(x)
        # cpt = torch.normal(0,0.1,size=(b,t,e),device=device)
        x_init   = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### need to concat a single prefix to avoid
        # att_mask = torch.tril(att_mask, diagonal=1)

        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        def get_xpred(cbt,ctx):
            ### (b,tp,nh,nei)
            #### adding a context
            #### cpt 
            xpred = (md.lin_output(cbt)).reshape((b,t,nh,nei))
            return xpred

        def get_cpred(x,cbt,ctx):            
            ### (b,tp,nei,nh,tc)
            ### no skip connection for the moment
            # cpred = (md.lin_transit(cbt)).reshape((b,t,nei,nh,1)).expand((b,t,nei,nh,t)) 
            cpred = cbt.matmul(md.lin_transit.weight.T)
            cpred = cpred + md.lin_input(x)
            # cpred = 
            cpred = 0.5 * cpred
            cpred = cpred.matmul(md.lin_internal.weight.T)
            # cpred = cpred.sigmoid()
            return cpred


        for i in range(4):
            ### (b,tp,e)

            #### calculate alternative hypotheses w.r.t. the word emission.
            # xpred   = get_xpred(cbt,ctx)
            cptpred = get_cpred(x,cbt,None)

            ### (b,tp,nh,tc)
            lp  = 0.
            lp += torch.einsum('bpe,hbce->bphc', cpt, cbt[None])

            ### (b,tc,nh,tp)
            lp = lp.transpose(1,3)            
            lp = lp.masked_fill(att_mask[None,:,None,:]==0,float('-inf'))

            ### (b,tc,nh,tp)
            att = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)

            # breakpoint()    
            dcbt = 0.
            dcbt += torch.einsum('bchp,hbpe->bce',  att, cpt[None]) 
            dcbt += cpt.matmul(md.lin_internal.weight).matmul(md.lin_transit.weight)
            dcpt = cptpred
            # dcpt = torch.einsum('bchp,bpehc->bce', att, cptpred).matmul(md.lin_internal.weight.T)
            # cpt = cpt + 0.1*dcptstep 2000: train loss 3.9712, val loss 4.9935

            cpt = cpt + 0.2*dcpt
            cbt = cbt + 0.2*dcbt


        xpred   = get_xpred(cbt,None)


        xpred = xpred.matmul(md.lin_internal_output.weight.T).reshape((b,t,nh,ne))
        x     = xpred
        x     = self.transformer.ln_f(x)            

        logits= self.lm_head(x).log_softmax(-1).logsumexp(2)-math.log(nh)


        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        else:
            loss  = None
        return logits, loss         


class CCM05(GPT):
    '''
    CCM04 extend the hidden state to have pre and post context 
    
    step 1750: train loss 4.0262, val loss 4.9741

    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        config.g = 400
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        # config.n_internal = config.n_embd//16
        config.n_internal = config.n_embd//4

        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2
        config.method = -1
        config.n_head = 1

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__(config)

        # config.block_size = config.block_size*

        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        print(config)
        # breakpoint()
        ne  = config.n_embd
        nh  = config.n_head
        nei = config.n_internal
        # config.n_head = 
        # nh = 4
        config.nc = 4
        nc = config.nc

        # config.g = 200

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # k_query = nn.Linear(config.n_embd,config.k, ),
            k_vects = nn.Embedding(config.g, ne),
            v_vects = nn.Embedding(config.g, ne),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            lin_init     = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne//nc, ne,bias=config.bias),
            # lin_output_ctx   = nn.Linear(ne, nh*nei,bias=config.bias),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne//nc,ne//nc,bias=config.bias),
            # lin_transit_ctx = nn.Linear(ne,nh*nei,bias=config.bias),
            lin_cpt_attention = nn.Linear(ne,ne,bias=config.bias),
            lin_cpt_attention_2 = nn.Linear(ne,ne,bias=config.bias),
            lin_cpt_attention_3 = nn.Linear(ne,ne,bias=config.bias),
            # lin_ctx = nn.Linear(ne,ne,bias=config.bias),
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
        cpt      = torch.zeros((b,t,e),device=device)  + md.lin_init(x)
        ### context before 
        cbt      = torch.zeros((b,t,e),device=device)  + md.lin_init(x)
        # cpt = torch.normal(0,0.1,size=(b,t,e),device=device)
        x_init   = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### need to concat a single prefix to avoid
        # att_mask = torch.tril(att_mask, diagonal=1)

        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T
        nc = config.nc
        def get_xpred(cbt,):
            ### (b,tp,nh,nei)
            #### adding a context
            #### cpt 
            cbts = cbt.reshape((b,t,nc,ne//nc))
            # v1,v2 = cbt.split(ne//2,dim=2)
            xpred = cbts.matmul(md.lin_output.weight.T)
            # (cbts)
            return xpred



        for i in range(6):
            ### (b,tp,e)

            #### calculate alternative hypotheses w.r.t. the word emission.
            xpred   = get_xpred(cbt,)
            # torch.spol


            ### (b,tp,nh,tc)
            lp  = 0.
            lp += torch.einsum('bpe,hbce->bphc', cpt, cbt[None])

            ### (b,tc,nh,tp)
            lp = lp.transpose(1,3)            
            lp = lp.masked_fill(att_mask[None,:,None,:]==0,float('-inf'))

            ### (b,tc,nh,tp)
            att = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)

            ### (b,tc,nc,ne)
            cbts = cbt.reshape((b,t,nc,ne//nc))
            cpts = cpt.reshape((b,t,nc,ne//nc))
            cpts_pred = cbts.matmul(md.lin_transit.weight)

            ### 
            lp   =0.
            lpc  = torch.einsum('btce,btce->btc', cbts, cpts)
            lp   += lpc.sum(-1,keepdim=True) - lpc
            lp   += torch.einsum('btce,btce->btc',cpts_pred,cpts)            
            lp   += torch.einsum('btce,bte->btc',get_xpred(cbt),x)
            att2 = lp.softmax(-1)
            # att2 = lp
            # breakpoint()    
            dcbt = 0.
            dcbt += torch.einsum('bchp,hbpe->bce',  att, cpt[None])
            dcbt += torch.einsum('btc,btck->btck',att2, 
                (x.matmul(md.lin_output.weight).unsqueeze(2) + cpts.matmul(md.lin_transit.weight.T)) 
                ).reshape((b,t,ne))
            dcpt = 0.
            dcpt += (cpts_pred * att2.unsqueeze(-1)).reshape((b,t,ne))
            # dcpt += 


            cpt = cpt + 0.1*dcpt
            cbt = cbt + 0.1*dcbt

        xpred   = get_xpred(cbt,)

        x     = xpred
        x     = self.transformer.ln_f(x)            

        logits= self.lm_head(x).log_softmax(-1).logsumexp(2)-math.log(nc)


        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        else:
            loss  = None
        return logits, loss         



import torch.nn as nn


# class CCM01(nn.Module):
class CCM06(GPT):
    '''

calculating the rnn transition to use atttended vector



iter 1480: loss 4.1011, time 94.83ms, mfu 3.64%
iter 1490: loss 4.1817, time 94.83ms, mfu 3.65%
step 1500: train loss 4.1789, val loss 5.0421

step 2000: train loss 4.0252, val loss 5.0605


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
        config.n_internal = config.n_embd//4


        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2
        config.method = -1
        config.n_head = 1

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__(config)

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
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            lin_output   = nn.Linear(ne, nh*nei,bias=config.bias),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,nh*nei,bias=config.bias),
            log_prior   = nn.Linear(10,2,bias=config.bias),
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
        # x        = torch.cat([torch.zeros((b,1,e),device=device),x],dim=1)
        # t = t+1
        cpt      = torch.zeros((b,t,e),device=device)+ x
        # x_init   = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### need to concat a single prefix to avoid
        # att_mask = torch.tril(att_mask, diagonal=1)

        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        def get_xpred(cpt):
            ### (b,tp,nh,nei)
            # xpred = md.lin_output(cpt).sigmoid().reshape((b,t,nh,ne))
            xpred = md.lin_output(cpt).reshape((b,t,nh,nei))
            return xpred


        def get_cpred(x,cpt):
            ### (b,tp,nei,nh,tc)
            ### no skip connection for the momenta
            cpred = md.lin_transit(cpt).reshape((b,t,nei,nh,1)).expand((b,t,nei,nh,t)) 
            cpred = cpred + md.lin_input(x).transpose(1,2)[:,None,:,None,:].expand((b,t,nei,nh,t)) 
            cpred = 0.5 * cpred
            # cpred = cpred.sigmoid()
            return cpred

        for i in range(4):
            xpred = get_xpred(cpt)
            cpred = get_cpred(x,cpt)

            ### (b,tp,nh,tc)
            lp = 0.
            # lp += xpred.matmul(xrep) 
            lp += torch.einsum('bphe,bce->bphc', xpred, x.matmul(md.lin_internal_output.weight)) 
            lp += torch.einsum('bpehc,bce->bphc',cpred, cpt.matmul(md.lin_internal.weight))

            ### (b,tc,nh,tp)
            lp = lp.transpose(1,3)
            
            lp = lp.masked_fill(att_mask[None,:,None,:]==0,float('-inf'))

            ### (b,tc,nh,tp)
            # att = lp.reshape.softmax((2,3))
            att = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)
            # 2,3))
            # breakpoint()    
            dcpt = torch.einsum('bchp,bpehc->bce', att, cpred).matmul(md.lin_internal.weight.T)
            # cpt2 = 
            cpt = cpt + 0.1*dcpt

        # xpred = md.lin_output(dcpt).reshape((b,t,nh,nei))
        xpred = get_xpred(cpt)
        xpred = xpred.matmul(md.lin_internal_output.weight.T).reshape((b,t,nh,ne))
        x     = xpred

        x     = self.transformer.ln_f(x)            

        ###step 1750: train loss 4.5279, val loss 5.3416

        logits= self.lm_head(x).log_softmax(-1)
        # lprior =  md.log_prior.weight.T[:1,None,:,None].log_softmax(2)
        # # lprior = 0.
        # logits = (torch.cat([logits[:,:-1],logits[:,1:]],dim=2)+lprior).logsumexp(2)
        # logits = torch.cat([logits[:,:-1],logits[:,1:]],dim=2)
        logits = logits.logsumexp(2)


        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        else:
            loss  = None
        return logits, loss         

class CCM06(GPT):
    '''

calculating the rnn transition to use atttended vector



iter 1480: loss 4.1011, time 94.83ms, mfu 3.64%
iter 1490: loss 4.1817, time 94.83ms, mfu 3.65%
step 1500: train loss 4.1789, val loss 5.0421

step 2000: train loss 4.0252, val loss 5.0605

step 1500: train loss 4.0372, val loss 4.9759
step 1500: train loss 4.0372, val loss 4.9759


1-layer

2-layer
step 1000: train loss 4.0673, val loss 4.8889
step 1250: train loss 3.9270, val loss 4.8307
step 1500: train loss 3.7982, val loss 4.8263

2-layer other initialisation
step 1000: train loss 4.1762, val loss 4.9463
step 1250: train loss 4.0637, val loss 4.9493
step 1500: train loss 3.9368, val loss 4.9163
step 1750: train loss 3.8000, val loss 4.8896





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
        config.n_internal = config.n_embd//4


        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        # config.block_size = config.block_size*2
        config.method = -1
        config.n_head = 1

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__(config)

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
            wb1 = nn.Embedding(config.vocab_size, config.n_embd),
            wb2 = nn.Embedding(config.vocab_size, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output_2   = nn.Linear(ne, ne,bias=config.bias),
            lin_output_3   = nn.Linear(ne, ne,bias=config.bias),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,nei,bias=config.bias),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,nei,bias=config.bias),
            lin_input_2   = nn.Linear(ne,ne,bias=config.bias),
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

        cpt       = (torch.zeros((b,t,e),device=device)+ (x))
        cpt2      = (torch.zeros((b,t,e),device=device)+ (x))

        # cpt       = (torch.zeros((b,t,e),device=device)+ md.wb1(idx) + x.matmul(md.lin_output.weight.T)) * 0.5
        # cpt2      = (torch.zeros((b,t,e),device=device)+ md.wb2(idx) + md.lin_input_2(cpt)) * 0.5

        # cpt       = (torch.zeros((b,t,e),device=device)+ pos_emb)
        # cpt2      = (torch.zeros((b,t,e),device=device)+ pos_emb)
        
        
        # cpt       = (torch.zeros((b,t,e),device=device)+ x.matmul(md.lin_output.weight.T)).detach()
        # cpt2      = (torch.zeros((b,t,e),device=device)+ md.lin_input_2(cpt)).detach()

        # x_init   = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### need to concat a single prefix to avoid
        # att_mask = torch.tril(att_mask, diagonal=1)

        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        def get_xpred(cpt):
            ### (b,tp,nh,nei)
            # xpred = md.lin_output(cpt).sigmoid().reshape((b,t,nh,ne))
            # xpred = md.lin_output(cpt)
            xpred = cpt.matmul(md.lin_output.weight)
            return xpred

        def get_cpred(x,cpt):
            ### (b,tp,nei,tc)
            ### no skip connection for the momenta
            cpred = md.lin_transit(cpt).unsqueeze(-1).expand((b,t,nei,t)) 
            cpred = cpred + md.lin_input(x).transpose(1,2)[:,None,:,:].expand((b,t,nei,t)) 
            cpred = 0.5 * cpred
            return cpred

        # for i in range(4):
        for i in range(8):
            xpred = get_xpred(cpt)
            cpred = get_cpred(x,cpt)

            ### (b,tp,nh,tc)
            lps = 0.
            lp = 0.
            #### gradient to predict the token
            lp += torch.einsum('bpe,bce ->bcp', xpred,   x)
            lp += torch.einsum('bpec,bce->bcp', cpred, cpt.matmul(md.lin_internal.weight))

            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            lps += lp.logsumexp(2).mean()
            
            ### (b,tc,tp)
            att  = lp.softmax(-1)
            dcpt = torch.einsum('bcp,bpec->bce', att, cpred).matmul(md.lin_internal.weight.T)
            cpt  = 0.9*cpt + 0.1*dcpt


            ### (b,tp,nh,nei)
            cpt_pred = md.lin_output_2(cpt2)

            ### (b,tp,nei,tc)
            cpt2_pred = md.lin_transit_2(cpt2).unsqueeze(-1) + md.lin_input_2(cpt).transpose(1,2)[:,None,:,:].expand((b,t,ne,t)) 
            # xpred = get_xpred(cpt)
            ### (b,tp,nh,tc)
            lp = 0.
            #### gradient to predict the token
            lp += torch.einsum('bpe,bce ->bcp',  cpt_pred,   cpt)
            lp += torch.einsum('bpec,bce->bcp', cpt2_pred,  cpt2)
            
            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            # lps += lp.mean().item()
            lps += lp.logsumexp(2).mean()
            
            ### (b,tc,tp)
            att  = lp.softmax(-1)
            dcpt2 = torch.einsum('bcp,bpec->bce', att, cpt2_pred)
            cpt2  = 0.9*cpt2 + 0.1*dcpt2
            # print(f'[inner_iter_{i}] {lps}')

        
        x     = cpt2.matmul(md.lin_output_3.weight)
        x     = self.transformer.ln_f(x)            


        logits = self.lm_head(x).log_softmax(-1)
        # logits = logits.logsumexp(2)


        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            # if self.training:
            #     loss = loss - lps
        else:
            loss  = None
        return logits, loss     


class CCM07(GPT):
    '''

adding internal representation proba for gradient
step 1250: train loss 4.1276, val loss 4.9084
step 1750: train loss 4.0458, val loss 4.9245

step 1500: train loss 4.0254, val loss 4.9028
step 1750: train loss 3.9641, val loss 4.8874


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
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head = 1

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        # super(GPT,self).__init__(config)
        super(GPT,self).__init__()

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
            wb1 = nn.Embedding(config.vocab_size, config.n_embd),
            wb2 = nn.Embedding(config.vocab_size, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne,bias=True),
            lin_output_2   = nn.Linear(ne, ne,bias=True),
            lin_output_3   = nn.Linear(ne, ne,bias=True),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=True),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=True),
            lin_input_2   = nn.Linear(ne,ne,bias=True),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        cpt       = (torch.zeros((b,t,e),device=device)+ (x))
        cpt2      = (torch.zeros((b,t,e),device=device)+ (x))

        # cpt       = (torch.zeros((b,t,e),device=device)+ pos_emb)
        # cpt2      = (torch.zeros((b,t,e),device=device)+ pos_emb)
                
        # cpt       = (torch.zeros((b,t,e),device=device)+ x.matmul(md.lin_output.weight.T)).detach()
        # cpt2      = (torch.zeros((b,t,e),device=device)+ md.lin_input_2(cpt)).detach()

        # x_init   = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### need to concat a single prefix to avoid
        # att_mask = torch.tril(att_mask, diagonal=1)

        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        for i in range(4):
        # for i in range(8):
            xpred = md.lin_output(cpt)
            # xpred = cpt.matmul(md.lin_output.weight)

            cpred = md.lin_transit(cpt).unsqueeze(-1)
            cpred = cpred + md.lin_input(x).transpose(1,2)[:,None,:,:]
            cpred = cpred


            ### (b,tp,nh,tc)
            lps = 0.
            lp  = 0.
            #### gradient to predict the token

            ### calculating the gaussian -(x-y)^2 as xy-x^2-y^2
            lp += torch.einsum('bpe,bce ->bcp', xpred,  x )
            lp += -xpred.square().sum(-1,keepdims=True).transpose(1,2)
            lp += -x.square().sum(-1,keepdims=True)

            lp += torch.einsum('bpec,bce->bcp', cpred, cpt)
            lp += -cpred.square().sum(2,keepdims=False).transpose(1,2)
            lp += -cpt.square().sum(-1,keepdims=True)

            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            lps += lp.logsumexp(2).mean()
            
            ### (b,tc,tp)
            att  = lp.softmax(-1)
            dcpt = torch.einsum('bcp,bpec->bce', att, cpred)
            cpt  = cpt + 0.3 * (dcpt - cpt)
            

            ### (b,tp,nh,nei)
            cpt_pred = md.lin_output_2(cpt2)

            ### (b,tp,nei,tc)
            cpt2_pred = md.lin_transit_2(cpt2).unsqueeze(-1) + md.lin_input_2(cpt).transpose(1,2)[:,None,:,:].expand((b,t,ne,t)) 
            # xpred = get_xpred(cpt)
            ### (b,tp,nh,tc)
            lp = 0.
            #### gradient to predict the token
            lp += torch.einsum('bpe,bce ->bcp',  cpt_pred,   cpt)
            lp += torch.einsum('bpec,bce->bcp', cpt2_pred,  cpt2)

            lp += -cpt.square().sum(-1,keepdims=True)
            lp += -cpt2.square().sum(-1,keepdims=True)
            lp += -cpt_pred.square().sum(-1,keepdims=True).transpose(1,2)
            lp += -cpt2_pred.square().sum(2,keepdims=False).transpose(1,2)
            
            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            # lps += lp.mean().item()
            lps += lp.logsumexp(2).mean()
            
            ### (b,tc,tp)
            att   = lp.softmax(-1)
            dcpt2 = torch.einsum('bcp,bpec->bce', att, cpt2_pred)
            # cpt2  = 0.8*cpt2 + 0.2*dcpt2
            cpt2  = cpt2 + 0.3*(dcpt2-cpt2)
            # print(f'[inner_iter_{i}] {lps.item()}')

        
        # y     = cpt2.matmul(md.lin_output_3.weight)
        y     = md.lin_output_3(cpt2)
        # .matmul(md.lin_output_3.weight)

        # y     = self.transformer.ln_f(y)            
        # logits = self.lm_head(y).log_softmax(-1)

        # pred = 10*self.lm_head(y)
        pred = self.lm_head(y)
        logits = (pred - y.square().sum(-1,keepdims=True) - self.lm_head.weight.T[None].square().sum(1,keepdims=True)).log_softmax(-1)


        # pred = self.lm_head(y)
        # # pred = 10*self.lm_head(y)
        # logits = (pred - y.square().sum(-1,keepdims=True) - self.lm_head.weight.T[None].square().sum(1,keepdims=True)).log_softmax(-1)



        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                loss = loss - lps
        else:
            loss  = None
        return logits, loss     


class CCM08(GPT):
    '''

adding internal representation proba for gradient
step 1250: train loss 4.1276, val loss 4.9084
step 1750: train loss 4.0458, val loss 4.9245

step 1500: train loss 4.0254, val loss 4.9028
step 1750: train loss 3.9641, val loss 4.8874


step 1000: train loss 4.0233, val loss 4.8508
step 1250: train loss 3.8650, val loss 4.8137
step 1500: train loss 3.7054, val loss 4.8038

single layer with performance 
step 2000: train loss 3.9888, val loss 4.8662
step 2250: train loss 3.8395, val loss 4.8260



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
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head = 1

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        # super(GPT,self).__init__(config)
        super(GPT,self).__init__()

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
            wb1 = nn.Embedding(config.vocab_size, config.n_embd),
            wb2 = nn.Embedding(config.vocab_size, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne,bias=True),
            lin_output_2   = nn.Linear(ne, ne,bias=True),
            lin_output_3   = nn.Linear(ne, ne,bias=True),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=True),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=True),
            lin_input_2   = nn.Linear(ne,ne,bias=True),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        # cpt       = (torch.zeros((b,t,e),device=device)+ (x))
        # cpt2      = (torch.zeros((b,t,e),device=device)+ (x))

        # cpt       = (torch.zeros((b,t,e),device=device)+ pos_emb + md.wb1(idx) )
        # cpt2      = (torch.zeros((b,t,e),device=device)+ pos_emb + md.wb2(idx))

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt       = (torch.zeros((b,t,e),device=device)+ pos_emb )
        cpt2      = (torch.zeros((b,t,e),device=device)+ pos_emb)

        # cpt       = (torch.zeros((b,t,e),device=device)+ x.matmul(md.lin_output.weight.T)).detach()
        # cpt2      = (torch.zeros((b,t,e),device=device)+ md.lin_input_2(cpt)).detach()

        # x_init   = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### need to concat a single prefix to avoid
        # att_mask = torch.tril(att_mask, diagonal=1)

        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        for i in range(4):
        # for i in range(8):
            xpred = md.lin_output(cpt)
            xpred = self.transformer.ln_f(F.softplus(xpred))
            # xpred = cpt.matmul(md.lin_output.weight)

            cpred = md.lin_transit(cpt).unsqueeze(-1)
            cpred = cpred + md.lin_input(x).transpose(1,2)[:,None,:,:]
            cpred = self.transformer.ln_f(F.softplus(cpred.transpose(2,3))).transpose(2,3)
            cpred = cpred
            xnorm = self.transformer.ln_f(F.softplus(x))


            ### (b,tp,nh,tc)
            lps = 0.
            lp  = 0.
            #### gradient to predict the token

            ### calculating the gaussian -(x-y)^2 as xy-x^2-y^2
            lp += torch.einsum('bpe,bce ->bcp', xpred,  xnorm )

            lp += torch.einsum('bpec,bce->bcp', cpred, cpt)

            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            lps += lp.logsumexp(2).mean()
            
            ### (b,tc,tp)
            att  = lp.softmax(-1)
            dcpt = torch.einsum('bcp,bpec->bce', att, cpred)
            cpt  = cpt + 0.3 * (dcpt - cpt)
            

            ### (b,tp,nh,nei)
            cpt_pred = md.lin_output_2(cpt2)
            cpt_pred = self.transformer.ln_f(F.softplus(cpt_pred))

            ### (b,tp,nei,tc)
            cpt2_pred = md.lin_transit_2(cpt2).unsqueeze(-1) + md.lin_input_2(cpt).transpose(1,2)[:,None,:,:].expand((b,t,ne,t)) 
            cpt2_pred = self.transformer.ln_f(F.softplus(cpt2_pred.transpose(2,3))).transpose(2,3)

            cpt2_norm = self.transformer.ln_f(F.softplus(cpt2))

            cpt_norm = self.transformer.ln_f(F.softplus(cpt))

            # xpred = get_xpred(cpt)
            ### (b,tp,nh,tc)
            lp = 0.
            #### gradient to predict the token
            lp += torch.einsum('bpe,bce ->bcp',  cpt_pred,   cpt_norm)
            lp += torch.einsum('bpec,bce->bcp', cpt2_pred,  cpt2_norm)

            # lp += -cpt.square().sum(-1,keepdims=True)
            # lp += -cpt2.square().sum(-1,keepdims=True)
            # lp += -cpt_pred.square().sum(-1,keepdims=True).transpose(1,2)
            # lp += -cpt2_pred.square().sum(2,keepdims=False).transpose(1,2)
            
            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            # lps += lp.mean().item()
            lps += lp.logsumexp(2).mean()
            
            ### (b,tc,tp)
            att   = lp.softmax(-1)
            dcpt2 = torch.einsum('bcp,bpec->bce', att, cpt2_pred)
            # cpt2  = 0.8*cpt2 + 0.2*dcpt2
            cpt2  = cpt2 + 0.3*(dcpt2-cpt2)
            # print(f'[inner_iter_{i}] {lps.item()}')

        
        
        y     = md.lin_output_3(cpt2)

        # y     = cpt2.matmul(md.lin_output_3.weight)
        #### treat xi wij yj as probability, and use log
        #### logxi + logwij + logyj then do lse on i dimension
        # torch.einsum


        ### 4.80

        # y     = self.transformer.ln_f(y) 
        # wt2 = self.lm_head.weight.abs()

        # logits = y.relu().matmul(wt2.T)
        # logits = logits.log_softmax(-1)

        ### step 1750: train loss 3.9248, val loss 4.8322
        y     = self.transformer.ln_f( F.softplus(y))
        wt2   = F.softplus(self.lm_head.weight)

        logits = y.matmul(wt2.T)
        logits = logits.log_softmax(-1)


        ### step 1750: train loss 3.9257, val loss 4.8635
        # y     = self.transformer.ln_f( F.relu(y))
        # wt2   = F.relu(self.lm_head.weight)

        # logits = y.matmul(wt2.T)
        # logits = logits.log_softmax(-1)


        ### adding layernorm to intermediate
        ### step 1750: train loss 3.9853, val loss 4.7923

        ### using other vector to initialise the context
        ### step 1500: train loss 3.7383, val loss 4.8145

        ### init with only positional embedding
        ### step 2500: train loss 3.7122, val loss 4.7536

        ### GPT01,2-layer
        ### step 1000: train loss 3.9342, val loss 4.7494
        ### step 1250: train loss 3.7479, val loss 4.7120


        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss             



class CCM09(GPT):
    '''
single layer chain machine with special emission

### GPT01,2-layer
### step 1000: train loss 3.9342, val loss 4.7494
### step 1250: train loss 3.7479, val loss 4.7120


### no positional induction
### step 2500: train loss 3.7307, val loss 4.8103


### not much difference, marginally better perf.
### might be due to dominated by the nearest neighbor anyway.

### with positional induction
### step 2000: train loss 4.2439, val loss 4.9472
### step 2250: train loss 4.0596, val loss 4.8990
### step 2500: train loss 3.9507, val loss 4.8722
### step 2750: train loss 3.8764, val loss 4.8184
### step 3250: train loss 3.7130, val loss 4.7826

### 
### matching the output function
### step 3500: train loss 4.0816, val loss 4.9196
### step 3750: train loss 3.7779, val loss 4.8369
### step 4000: train loss 3.6714, val loss 4.8052



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
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        # super(GPT,self).__init__(config)
        super(GPT,self).__init__()

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
            wpe = nn.Embedding(config.block_size+2, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne,bias=True),
            lin_output_2   = nn.Linear(ne, ne,bias=True),
            lin_output_3   = nn.Linear(ne, ne,bias=True),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=True),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=True),
            lin_input_2   = nn.Linear(ne,ne,bias=True),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict


        # forward the GPT model itself
        pos = torch.cat([pos[:1],pos],0)
        idx = torch.cat([idx[:,:1]*0,idx],1)

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # pos_emb

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
        # cpt       = (torch.zeros((b,t,e),device=device)+ (x))
        # cpt2      = (torch.zeros((b,t,e),device=device)+ (x))

        # pos_emb = torch.cat([pos_emb[:1],pos_emb],0)
        t = t+1

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt       = (torch.zeros((b,t,e),device=device)+ pos_emb )
        cptn      = (torch.zeros((b,t,e),device=device)+ self.transformer.wpe(pos+1))
        # cpt       = (torch.zeros((b,t,e),device=device)+ x.matmul(md.lin_output.weight.T)).detach()
 
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        for i in range(6):
        # for i in range(8):
            xpred = md.lin_output(cpt)
            xpred = self.transformer.ln_f(F.softplus(xpred))
            # xpred = cpt.matmul(md.lin_output.weight)

            cpred      = md.lin_transit(cpt)
            cpred_past = self.transformer.ln_f(F.softplus(cpred))

            cpred = cpred.unsqueeze(-1) + md.lin_input(x).transpose(1,2)[:,None,:,:]
            cpred = self.transformer.ln_f(F.softplus(cpred.transpose(2,3))).transpose(2,3)
            cpred = cpred

            xnorm = self.transformer.ln_f(F.softplus(x))

            ### (b,tp,nh,tc)
            lps = 0.
            lp  = 0.
            #### gradient to predict the token

            ### calculating the gaussian -(x-y)^2 as xy-x^2-y^2
            lp += torch.einsum('bpe,bce ->bcp', xpred,  xnorm )
            lp += torch.einsum('bpec,bce->bcp', cpred,  cpt)

            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            att  = lp.softmax(-1)
            lps += lp.logsumexp(2).mean()            

            dcpt = torch.einsum('bcp,bpec->bce', att, cpred)
            cpt  = cpt  + 0.3 * (dcpt - cpt)

            lp   = 0.
            # print(cpred_past.shape)
            lp  += torch.einsum('bpe,bce->bcp', cpred_past,  cptn)            
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            att  = lp.softmax(-1)
            lps += lp.logsumexp(2).mean()            

            dcptn = torch.einsum('bcp,bpe->bce', att, cpred_past)
            cptn  = cptn  + 0.3 * (dcptn - cptn)


        ### inducing the next token using the pos_embedding
        # cpt_ind = cptn        
        cpt_ind = cpt
        # y = md.lin_output(cpt_ind)
        y       = md.lin_output_3(cpt_ind)

        ### step 1750: train loss 3.9248, val loss 4.8322
        y     = self.transformer.ln_f( F.softplus(y))
        wt2   = F.softplus(self.lm_head.weight)

        logits = y.matmul(wt2.T)
        logits = logits[:,1:]
        logits = logits.log_softmax(-1)



        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss             


class CCM10(GPT):
    '''
    single layer chain machine with simpler update
step 1500: train loss 4.1113, val loss 4.9812

step 1250: train loss 4.0702, val loss 4.8442
step 1500: train loss 3.9191, val loss 4.8337
step 1750: train loss 3.8240, val loss 4.8263
step 2000: train loss 3.7416, val loss 4.8337


step 1250: train loss 4.0948, val loss 4.8566
step 1500: train loss 3.9369, val loss 4.8269
step 1750: train loss 3.8104, val loss 4.8232

step 1500: train loss 3.9342, val loss 4.8278
step 1750: train loss 3.8212, val loss 4.8054


gpt01:4.70

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
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        # super(GPT,self).__init__(config)
        super(GPT,self).__init__()

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
        config.nc = 2
        nc= config.nc

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size+2, config.n_embd),
            whe = nn.Embedding(config.nc, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            # lin_output_4   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=False),
            lin_input_2   = nn.Linear(ne,ne,bias=False),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        nc = self.config.nc
        head = pos[:nc]
        # forward the GPT model itself
        pos = torch.cat([pos[:1],pos],0)
        idx = torch.cat([idx[:,:1]*0,idx],1)

        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        head_emb = md.whe(head) ## (nc,emb)

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
        t = t+1

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt        = (torch.zeros((b,t,e),device=device) + pos_emb + head_emb[0:1])
        cpt2       = (torch.zeros((b,t,e),device=device) + pos_emb + head_emb[1:2])
 
        att_mask = torch.ones((t+1,t+1),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)

        ### non-self attention
        att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask[0,0]=1
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T
        cpt = self.transformer.ln_f(F.softplus(cpt))
        cpt2 = self.transformer.ln_f(F.softplus(cpt2))

        xnorm = self.transformer.ln_f(F.softplus(x))

        for i in range(4):
            cpred = md.lin_transit(cpt)
            cpred = self.transformer.ln_f(F.softplus(cpred))

            ### (b,tp,nh,tc)
            lp  = 0.
            
            ### equalising the final layer
            lp  += torch.einsum('bpe,bce->bcp', cpred,  cpt)

            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:t,:t]==0,float('-inf'))
            att  = lp.softmax(-1)

            cpred2 = md.lin_transit_2(cpt2)
            cpred2 = self.transformer.ln_f(F.softplus(cpred2))

            ### bce
            cpred2_cpt = cpt.matmul(md.lin_output_2.weight)
            cpred2_cpt = self.transformer.ln_f(F.softplus(cpred2_cpt))

            lp    = 0.            
            lp   += torch.einsum('bpe,bce->bcp', cpred2,     cpt2)
            lpp   = torch.einsum('bce,bce->bc',  cpred2_cpt, cpt2).unsqueeze(-1)
            lp    = torch.cat([lpp,lp],dim=2)
            lp    = lp.masked_fill(att_mask[None,:t,:t+1]==0,float('-inf'))
            att2  = lp.softmax(-1)


            dcpt2  = 0.
            dcpt2 += torch.einsum('bc,bce->bce',  att2[:,:,0],  cpred2_cpt)
            dcpt2 += torch.einsum('bcp,bpe->bce', att2[:,:,1:], cpred2)
            dcpt2 += xnorm.matmul(md.lin_output.weight.T)
            # dcpt2[:,0]=0.
            cpt2  += 0.3*(dcpt2 - cpt2)
            cpt2  = self.transformer.ln_f(F.softplus(cpt2 ))


            dcpt = 0.
            dcpt += torch.einsum('bcp,bpe->bce', att,       cpred) 
            dcpt += torch.einsum('bc,bce->bce',  att2[:,:,0], cpt2.matmul(md.lin_output_2.weight.T))
            # dcpt += cpt2.matmul(md.lin_output_2.weight.T)
            # print(att2[:,5:6,0].mean((0,1)).item())
            # dcpt = self.transformer.ln_f(F.softplus(dcpt))
            # dcpt[:,0]=0.
            cpt +=  0.3 * (dcpt - cpt)
            cpt  = self.transformer.ln_f(F.softplus(cpt))


        ### inducing the next token using the pos_embedding
        # cpt_ind = cpt2
        y       = md.lin_output_3(cpt2)
        y       += md.lin_input(cpt)
        # y       += md.lin_output_4(cpt)

        ### step 1750: train loss 3.9248, val loss 4.8322
        y = y[:,1:]
        y     = self.transformer.ln_f( F.softplus(y))
        wt2   = F.softplus(self.lm_head.weight)

        logits = y.matmul(wt2.T)
        logits = logits.log_softmax(-1)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss         


class CCM11(GPT):
    '''
    single layer chain machine with simpler update
step 1500: train loss 4.1113, val loss 4.9812


gpt01:4.70


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
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        # super(GPT,self).__init__(config)
        super(GPT,self).__init__()

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
        config.nc = 2
        nc= config.nc

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size+2, config.n_embd),
            wpe2 = nn.Embedding(config.block_size+2, config.n_embd),
            whe = nn.Embedding(config.nc, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=False),
            lin_input_2   = nn.Linear(ne,ne,bias=False),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        nc = self.config.nc
        head = pos[:nc]
        # forward the GPT model itself
        pos = torch.cat([pos[:1],pos],0)
        idx = torch.cat([idx[:,:1]*0,idx],1)

        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2  = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        # pos_emb

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
        t = t+1

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt      = (torch.zeros((b,t,e),device=device) + pos_emb_2 )
 
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T
        cpt   = self.transformer.ln_f(F.softplus(cpt))

        xnorm = self.transformer.ln_f(F.softplus(x))

        for i in range(4):
            cpred = md.lin_transit_2(cpt)
            cpred = self.transformer.ln_f(F.softplus(cpred))

            ### (b,tp,nh,tc)
            lps  = 0.
            lp   = 0.
            #### gradient to predict the token

            # lp += torch.einsum('bpe,bce ->bcp', xpred,  xnorm )
            lp  += torch.einsum('bpe,bce->bcp', cpred,  cpt)

            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            att  = lp.softmax(-1)
            # lps += lp.logsumexp(2).mean()            

            dcpt = torch.einsum('bcp,bpe->bce', att, cpred) + xnorm.matmul(md.lin_output_2.weight)
            # dcpt = self.transformer.ln_f(F.softplus(dcpt))
            cpt  = cpt  +  0.3 * (dcpt - cpt)
            cpt  = self.transformer.ln_f(F.softplus(cpt))






        ### inducing the next token using the pos_embedding
        cpt_ind = cpt
        y       = md.lin_output_3(cpt_ind)

        ### step 1750: train loss 3.9248, val loss 4.8322
        y     = y[:,1:]
        y     = self.transformer.ln_f( F.softplus(y))
        wt2   = F.softplus(self.lm_head.weight)

        logits = y.matmul(wt2.T)
        logits = logits.log_softmax(-1)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss       
        


class CCM12(GPT):
    '''
    single layer chain machine with simpler update
step 1500: train loss 4.1113, val loss 4.9812

gpt01:4.70

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
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        # super(GPT,self).__init__(config)
        super(GPT,self).__init__()

        # config.block_size = config.block_size*

        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # print(config)
        # breakpoint()
        ne = config.n_embd
        nh = config.n_head
        nei = config.n_internal
        config.nc = 4
        nc= config.nc

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size+2, config.n_embd),
            whe = nn.Embedding(config.nc, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            # lin_output_4   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=False),
            lin_input_2   = nn.Linear(ne,ne,bias=False),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        nc = self.config.nc
        head = pos[:nc]
        # forward the GPT model itself
        pos = torch.cat([pos[:1],pos],0)
        idx = torch.cat([idx[:,:1]*0,idx],1)

        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        head_emb = md.whe(head) ## (nc,emb)

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
        t = t+1

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt        = (torch.zeros((b,t,e),device=device) + pos_emb + head_emb[0:1])
        cpt2       = (torch.zeros((b,t,e),device=device) + pos_emb + head_emb[1:2])
 
        att_mask = torch.ones((t+1,t+1),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T
        cpt = self.transformer.ln_f(F.softplus(cpt))
        cpt2= self.transformer.ln_f(F.softplus(cpt2))

        xnorm = self.transformer.ln_f(F.softplus(x))

        for i in range(4):
            cpred = md.lin_transit(cpt)
            cpred = self.transformer.ln_f(F.softplus(cpred))
            ### (b,tp,nh,tc)
            lp   = 0.            
            lp  += torch.einsum('bpe,bce->bcp', cpred,  cpt)
            ### (b,tc,tp)
            lp     = lp.masked_fill(att_mask[None,:t,:t]==0,float('-inf'))
            att    = lp.softmax(-1)
            att1 = att

            cpred2 = md.lin_transit_2(cpt2)
            cpred2 = self.transformer.ln_f(F.softplus(cpred2))
            lp   = 0.            
            ### equalising the final layer
            lp  += torch.einsum('bpe,bce->bcp', cpred2,  cpt2)
            ### (b,tc,tp)
            lp     = lp.masked_fill(att_mask[None,:t,:t]==0,float('-inf'))
            att    = lp.softmax(-1)
            att2   = att


            xpred = torch.stack([
                cpt.matmul(md.lin_output.weight), 
                cpt2.matmul(md.lin_output_2.weight), 
                ],dim=2)
            xpred = self.transformer.ln_f(F.softplus(xpred))


            lp    = 0.            
            lp   += torch.einsum('bcke,bce->bck',  xpred,    xnorm)
            attx  = lp.softmax(-1)

            dcpt = 0.
            dcpt += torch.einsum('bcp,bpe->bce', att1,       cpred)
            dcpt += torch.einsum('bce,bc->bce',  xnorm.matmul(md.lin_output.weight.T), attx[:,:,0])
            cpt  += 0.3 * (dcpt - cpt)
            cpt  = self.transformer.ln_f(F.softplus(cpt))

            dcpt2 = 0.
            dcpt2 += torch.einsum('bcp,bpe->bce', att2,       cpred2)
            dcpt2 += torch.einsum('bce,bc->bce',  xnorm.matmul(md.lin_output_2.weight.T), attx[:,:,1])
            cpt2 += 0.3 * (dcpt2 - cpt2)
            cpt2  = self.transformer.ln_f(F.softplus(cpt2))

            # print(f'attx:{attx.mean(dim=(0,1)).cpu().numpy()}')



        cptk  = torch.stack([cpt,cpt2],dim=2)        
        lp     = 0.            
        lp    += torch.einsum('bpke,ce->bckp', cptk, md.wpe(pos+1)).softmax(-1)
        ### (b,tc,tp)
        lp     = lp.masked_fill(att_mask[None,:t,None,:t]==0,float('-inf'))
        att    = lp.softmax(-1)

        cptke   = torch.einsum('bckp,bpke->bcke',att, cptk)
        # print(f'attx:{att.mean(dim=(0,1)).cpu().numpy()}')
        # y       = md.lin_input(cptx)
        # .matmul()  

        ### inducing the next token using the pos_embedding
        # cpt_ind = cpt2
        y       = md.lin_input(cptke[:,:,0])
        y       += md.lin_input_2(cptke[:,:,1])
        # y       += md.lin_output_4(cpt)

        ### step 1750: train loss 3.9248, val loss 4.8322
        y = y[:,1:]
        y     = self.transformer.ln_f( F.softplus(y))
        wt2   = F.softplus(self.lm_head.weight)

        logits = y.matmul(wt2.T)
        logits = logits.log_softmax(-1)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss         


class CCM13(GPT):
    '''
    single layer chain machine with simpler update
step 1500: train loss 4.1113, val loss 4.9812

gpt01:4.70


### disconnected layer2 from surface
step 2500: train loss 3.6747, val loss 4.8432

### connected layer2 to surface
step 2750: train loss 3.5988, val loss 4.8508
step 2500: train loss 3.7302, val loss 4.8392
step 3000: train loss 3.5849, val loss 4.8459
step 2500: train loss 3.6963, val loss 4.7974
step 2500: train loss 3.7159, val loss 4.8175

### disconnected layer2 modules
step 2500: train loss 3.7206, val loss 4.8239
###
step 2500: train loss 3.7040, val loss 4.8559

### 3 module
step 2000: train loss 3.8644, val loss 4.8257
step 2500: train loss 3.6597, val loss 4.8222



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
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__()

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
        config.nc = 2
        nc= config.nc

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size+2, config.n_embd),
            wpe_2 = nn.Embedding(config.block_size+2, config.n_embd),
            whe = nn.Embedding(config.nc, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(ne, ne,bias=config.bias),
            lin_internal_2 = nn.Linear(ne, ne,bias=config.bias),
            # lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            lin_transit_3 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=False),
            lin_input_2   = nn.Linear(ne,ne,bias=False),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        # nc = self.config.nc
        # head = pos[:nc]
        # forward the GPT model itself
        pos = torch.cat([pos[:1],pos],0)
        idx = torch.cat([idx[:,:1]*0,idx],1)

        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2  = self.transformer.wpe_2(pos) # position embeddings of shape (t, n_embd)
        # pos_emb

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
        t = t+1

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt        = (torch.zeros((b,t,e),device=device) + pos_emb )
        cpt2       = (torch.zeros((b,t,e),device=device) + pos_emb_2 )
        cpt3       = (torch.zeros((b,t,e),device=device) + pos_emb )
 
        # cpt    = self.transformer.ln_f(F.softplus(cpt)).square()
        # cpt2   = self.transformer.ln_f(F.softplus(cpt2)).square()
        # cpt3   = self.transformer.ln_f(F.softplus(cpt3)).square()

        cpt   = F.softplus(cpt)
        cpt   = cpt / cpt.sum(dim=2,keepdims=True) 
        cpt2   = F.softplus(cpt2)
        cpt2   = cpt2 / cpt2.sum(dim=2,keepdims=True) 

        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        xnorm = self.transformer.ln_f(F.softplus(x))

        for i in range(4):

            #### gradient to predict the token
            ### (b,tp,nh,tc)
            cpred = md.lin_transit(cpt)
            # cpred = self.transformer.ln_f(F.softplus(cpred)).square()

            cpred   = F.softplus(cpred)
            cpred   = cpred / cpred.sum(dim=2,keepdims=True) 

            lp   = 0.
            # lp += torch.einsum('bpe,bce ->bcp', xpred,  xnorm )
            lp  += torch.einsum('bpe,bce->bcp', cpred,  cpt)
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            att  = lp.softmax(-1)
            att1 = att


            cpred2 = md.lin_transit_2(cpt2)
            # cpred2 = self.transformer.ln_f(F.softplus(cpred2)).square()

            cpred2   = F.softplus(cpred2)
            cpred2   = cpred2 / cpred2.sum(dim=2,keepdims=True) 

            ### (b,tp,nh,tc)
            lp   = 0.
            lp  += torch.einsum('bpe,bce->bcp', cpred2,  cpt2)
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            att  = lp.softmax(-1)
            att2 = att

            # cpred3 = md.lin_transit_3(cpt3)
            # cpred3 = self.transformer.ln_f(F.softplus(cpred3))
            # ### (b,tp,nh,tc)
            # lp   = 0.
            # lp  += torch.einsum('bpe,bce->bcp', cpred3,  cpt3)
            # ### (b,tc,tp)
            # lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            # att  = lp.softmax(-1)
            # att3 = att


            dcpt =0.
            dcpt += torch.einsum('bcp,bpe->bce', att1, cpred) 
            dcpt += xnorm.matmul(md.lin_output.weight)    
            # dcpt += cpt2.matmul(md.lin_internal.weight)        
            cpt  += 0.3 * (dcpt - cpt)
            # cpt  = self.transformer.ln_f(F.softplus(cpt))
            # cpt   = F.softplus(cpt)
            # cpt   = cpt / cpt.sum(dim=2,keepdims=True) 


            dcpt2 =0.
            dcpt2 += torch.einsum('bcp,bpe->bce', att2, cpred2) 
            # dcpt2 += xnorm.matmul(md.lin_output_2.weight)            
            dcpt2 += cpt.matmul(md.lin_internal.weight.T)             
            # dcpt2 += cpt        
            # dcpt2 += cpt3.matmul(md.lin_internal_2.weight.T)        
            cpt2  += 0.3 * (dcpt2 - cpt2)
            # cpt2   = F.softplus(cpt2)
            # cpt2   = cpt2 / cpt2.sum(dim=2,keepdims=True) 
            # cpt2  = self.transformer.ln_f(F.softplus(cpt2))            


            # dcpt3 =0.
            # dcpt3 += torch.einsum('bcp,bpe->bce', att3, cpred3) 
            # dcpt3 += xnorm.matmul(md.lin_output_3.weight)            
            # dcpt3 += cpt2.matmul(md.lin_internal_2.weight)        
            # cpt3  += 0.3 * (dcpt3 - cpt3)
            # cpt3  = self.transformer.ln_f(F.softplus(cpt3))            



        ### inducing the next token using the pos_embedding
        # cpt_ind = cpt
        y = 0.
        # y        = md.lin_input(cpt)
        y       += md.lin_input_2(cpt2)

        ### step 1750: train loss 3.9248, val loss 4.8322
        y     = y[:,1:]
        y     = self.transformer.ln_f( F.softplus(y))
        wt2   = F.softplus(self.lm_head.weight ) 

        logits = y.matmul(wt2.T)
        logits = logits.log_softmax(-1)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss



class CCM14(GPT):
    '''
    single layer chain machine with simpler update
step 1500: train loss 4.1113, val loss 4.9812
step 3500: train loss 3.5270, val loss 4.9294

gpt01:4.70



    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        config.g = 300
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__()


        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        print(config)
        # breakpoint()
        ne = config.n_embd
        nh = config.n_head
        nei = config.n_internal
        config.nc = 2
        nc= config.nc

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size+2, config.n_embd),
            wpe_2 = nn.Embedding(config.block_size+2, config.n_embd),
            whe = nn.Embedding(config.nc, config.n_embd),
            wke = nn.Embedding(config.g, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(ne, ne,bias=config.bias),
            lin_internal_2 = nn.Linear(ne, ne,bias=config.bias),
            # lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            lin_transit_3 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=False),
            lin_input_2   = nn.Linear(ne,ne,bias=False),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        # nc = self.config.nc
        # head = pos[:nc]
        # forward the GPT model itself
        pos = torch.cat([pos[:1],pos],0)
        idx = torch.cat([idx[:,:1]*0,idx],1)

        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2  = self.transformer.wpe_2(pos) # position embeddings of shape (t, n_embd)
        # pos_emb

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
        t = t+1

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt       = (torch.zeros((b,t,e),device=device) + pos_emb )
        cpt2       = (torch.zeros((b,t,e),device=device) + pos_emb_2 )
        cpt3       = (torch.zeros((b,t,e),device=device) + pos_emb )
 
        cpt   = self.transformer.ln_f(F.softplus(cpt))
        cpt2   = self.transformer.ln_f(F.softplus(cpt2))
        cpt3   = self.transformer.ln_f(F.softplus(cpt3))

        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        xnorm = self.transformer.ln_f(F.softplus(x))
        kv = md.wke.weight
        knorm = self.transformer.ln_f(F.softplus(kv))

        for i in range(4):

            #### gradient to predict the token
            ### (b,tp,nh,tc)
            lp   = 0.
            cpred = md.lin_transit(cpt)
            cpred = self.transformer.ln_f(F.softplus(cpred))
            lp  += torch.einsum('bpe,bce->bcp', cpred,  cpt)
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            att  = lp.softmax(-1)
            att1 = att

            ### pull gradient from a reservoir
            lp   = 0.
            cpred2 = md.lin_transit_2(knorm)
            cpred2 = self.transformer.ln_f(F.softplus(cpred2))
            lp  += torch.einsum('ke,bce->bck', cpred2,  cpt)
            ### (b,tc,tp)
            att  = lp.softmax(-1)
            attk = att


            dcpt =0.
            dcpt += torch.einsum('bcp,bpe->bce', att1, cpred) 
            dcpt += torch.einsum('bck,ke->bce',  attk, cpred2) 
            dcpt += xnorm.matmul(md.lin_output.weight)    
            # dcpt += cpt2.matmul(md.lin_internal.weight)        
            cpt  += 0.3 * (dcpt - cpt)
            cpt  = self.transformer.ln_f(F.softplus(cpt))



        ### inducing the next token using the pos_embedding
        # cpt_ind = cpt
        y = 0.
        # y        = md.lin_input(cpt)
        y       += md.lin_input_2(cpt)

        ### step 1750: train loss 3.9248, val loss 4.8322
        y     = y[:,1:]
        y     = self.transformer.ln_f( F.softplus(y))
        wt2   = F.softplus(self.lm_head.weight)

        logits = y.matmul(wt2.T)
        logits = logits.log_softmax(-1)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss



class CCM15(GPT):
    '''
single layer with a reservoir vector space

This is the most promising direction, because this
is essentially a transformer with internal prompt,
and conceptually solves the problem of having internal 
state.



step 1500: train loss 4.1113, val loss 4.9812
step 3500: train loss 3.5270, val loss 4.9294
step 2750: train loss 3.7282, val loss 4.8884
step 4500: train loss 3.3463, val loss 4.9360

ccm11
step 1500: train loss 4.1113, val loss 4.9812
step 2750: train loss 3.7984, val loss 5.0579


###
step 1750: train loss 4.0215, val loss 4.9078
step 2250: train loss 3.8734, val loss 4.8862
step 3000: train loss 3.6586, val loss 4.8644


ccm15

gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        config.g = 30
        # config.g = 50
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__()


        config = self.add_config(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        print(config)
        # breakpoint()
        ne = config.n_embd
        nh = config.n_head
        nei = config.n_internal
        config.nc = 2
        nc= config.nc

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size+2, config.n_embd),
            wpe_2 = nn.Embedding(config.block_size+2, config.n_embd),
            whe = nn.Embedding(config.nc, config.n_embd),
            wke = nn.Embedding(config.g, config.n_embd),
            wve = nn.Embedding(config.g, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output     = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            lin_internal   = nn.Linear(ne, ne,bias=config.bias),
            lin_internal_2 = nn.Linear(ne, ne,bias=config.bias),
            # lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit   = nn.Linear(ne,ne,bias=False),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            lin_transit_3 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input     = nn.Linear(ne,ne,bias=False),
            lin_input_2   = nn.Linear(ne,ne,bias=False),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        # nc = self.config.nc
        # head = pos[:nc]
        # forward the GPT model itself
        pos = torch.cat([pos[:1],pos],0)
        idx = torch.cat([idx[:,:1]*0,idx],1)

        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2  = self.transformer.wpe_2(pos) # position embeddings of shape (t, n_embd)
        # pos_emb

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
        t = t+1
        k = self.config.g
        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt       = (torch.zeros((b,t,e),device=device) + pos_emb )
        # kv         = md.wke.weight
        # cptk       = (torch.zeros((b,t,k,e),device=device) + pos_emb_2[None,:,None] + md.wke.weight[None,None] )
        cptk       = (torch.zeros((b,t,k,e),device=device) + md.wke.weight[None,None] )

        cpt3       = (torch.zeros((b,t,e),device=device) + pos_emb )
 
        cpt   = self.transformer.ln_f(F.softplus(cpt))
        cptk   = self.transformer.ln_f(F.softplus(cptk))
        cpt3   = self.transformer.ln_f(F.softplus(cpt3))

        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        xnorm = self.transformer.ln_f(F.softplus(x))
        # knorm = self.transformer.ln_f(F.softplus(kv))

        for i in range(4):

            #### gradient to predict the token
            ### (b,tp,nh,tc)
            lp   = 0.
            cpred = md.lin_transit(cpt)
            cpred = self.transformer.ln_f(F.softplus(cpred))
            lp  += torch.einsum('bpe,bce->bcp', cpred,  cpt)
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            att  = lp.softmax(-1)
            att1 = att

            ### pull gradient from a reservoir
            lp   = 0.
            cpred2 = cptk.matmul(md.lin_transit_2.weight)
            cpred2 = self.transformer.ln_f(F.softplus(cpred2))
            lp  += torch.einsum('bcke,bce->bck', cpred2,  cpt)
            ### (b,tc,tp)
            att  = lp.softmax(-1)
            attk = att


            dcpt =0.
            dcpt += torch.einsum('bcp,bpe->bce', att1, cpred) 
            # dcpt += torch.einsum('bck,bcke->bce',  attk, cpred2) 
            dcpt += torch.einsum('bck,ke->bce',  attk, self.transformer.ln_f(F.softplus(md.wve.weight) ))
            # dcpt += torch.einsum('bck,ke->bce',  attk, (md.wve.weight) )
            # dcpt += torch.einsum('bck,bcke->bce',  attk, cptk.matmul(md.lin_transit_3.weight)) 
            # dcpt += torch.einsum('bck,ke->bce',  attk, md.wke.weight.matmul(md.lin_transit_3.weight)) 

            dcpt += xnorm.matmul(md.lin_output.weight)    
            # dcpt += cpt2.matmul(md.lin_internal.weight)        
            cpt  += 0.3 * (dcpt - cpt)
            cpt  = self.transformer.ln_f(F.softplus(cpt))

            # dcptk = 0.
            # dcptk += torch.einsum('bck,bce->bcke', attk, 
            #     self.transformer.ln_f(F.softplus(cpt.matmul(md.lin_transit_2.weight.T) )) )
            # dcptk += torch.einsum('bck,ke->bcke', attk,  md.wve.weight) 
            # dcptk += torch.einsum('bcp,bpke->bcke', att1,  cptk)
            # cptk  += 0.3 * (dcptk - cptk)
            # cptk  = self.transformer.ln_f(F.softplus(cptk))


        ### inducing the next token using the pos_embedding
        # cpt_ind = cpt
        y = 0.
        # y        = md.lin_input(cpt)
        y       += md.lin_input_2(cpt)

        ### step 1750: train loss 3.9248, val loss 4.8322
        y     = y[:,1:]
        y     = self.transformer.ln_f( F.softplus(y))
        wt2   = F.softplus(self.lm_head.weight)

        logits = y.matmul(wt2.T)
        logits = logits.log_softmax(-1)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss


class CCM16(GPT):
    '''
    using a GRU for the recurrent

    step 1500: train loss 3.9082, val loss 4.7978

    step 1500: train loss 3.7906, val loss 4.7973

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
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__()

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
        config.nc = 2
        nc= config.nc

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size+2, config.n_embd),
            wpe_2 = nn.Embedding(config.block_size+2, config.n_embd),
            whe = nn.Embedding(config.nc, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(ne, ne,bias=config.bias),
            lin_internal_2 = nn.Linear(ne, ne,bias=config.bias),
            # lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            lin_transit_3 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=True),
            lin_input_2   = nn.Linear(ne,ne,bias=True),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        # nc = self.config.nc
        # head = pos[:nc]
        # forward the GPT model itself
        pos = torch.cat([pos[:1],pos],0)
        idx = torch.cat([idx[:,:1]*0,idx],1)

        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2  = self.transformer.wpe_2(pos) # position embeddings of shape (t, n_embd)
        # pos_emb

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
        t = t+1

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt        = (torch.zeros((b,t,e),device=device) + pos_emb )
        cpt2       = (torch.zeros((b,t,e),device=device) + pos_emb_2 )
        cpt3       = (torch.zeros((b,t,e),device=device) + pos_emb )
 
        # cpt    = self.transformer.ln_f(F.softplus(cpt))
        # cpt   = cpt.sigmoid()
        #.square()
        # cpt2   = self.transformer.ln_f(F.softplus(cpt2)).square()
        # cpt3   = self.transformer.ln_f(F.softplus(cpt3)).square()

        cpt   = F.softplus(cpt)
        cpt   = cpt / cpt.sum(dim=2,keepdims=True) 
        # cpt2   = F.softplus(cpt2)
        # cpt2   = cpt2 / cpt2.sum(dim=2,keepdims=True) 

        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        xnorm = self.transformer.ln_f(F.softplus(x))
        
        # torch.autograd.set_detect_anomaly(True)
        for i in range(4):

            #### gradient to predict the token
            ### (b,tp,nh,tc)
            # cpred = md.lin_transit(cpt)
            ft = (md.lin_input(xnorm)   + md.lin_transit(cpt)).sigmoid()
            ht = (md.lin_input_2(xnorm) + md.lin_transit_2(cpt) ).sigmoid()
            # ht = (md.lin_input_2(xnorm) + md.lin_transit_2(cpt) ).tanh()
            cpred = (1-ft).detach()*cpt + ft*ht
            # cpred = (1-ft.sigmoid()).detach() *cpt  + ft.sigmoid()*ht
            # cpred = (1-ft.sigmoid()) *cpt  + ft.sigmoid()*ht
            # cpred = cpt - ft*cpt  + ft*ht
            # cpred = (1-ft)
 



            lp   = 0.
            # lp += torch.einsum('bpe,bce ->bcp', xpred,  xnorm )
            lp  += torch.einsum('bpe,bce->bcp', cpred,  cpt)
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            att  = lp.softmax(-1)
            att1 = att




            dcpt =0.
            dcpt += torch.einsum('bcp,bpe->bce', att1, cpred) 
            dcpt += xnorm.matmul(md.lin_output.weight).sigmoid()    
            # dcpt += cpt2.matmul(md.lin_internal.weight)        
            cpt  += 0.3 * (dcpt - cpt)
            # cpt  = self.transformer.ln_f(F.softplus(cpt))
            # cpt   = F.softplus(cpt)
            # cpt   = cpt / cpt.sum(dim=2,keepdims=True) 


        ### inducing the next token using the pos_embedding
        # cpt_ind = cpt
        y = 0.
        # y        = md.lin_input(cpt)
        y       += md.lin_output_3(cpt)

        ### step 1750: train loss 3.9248, val loss 4.8322
        y     = y[:,1:]
        y     = self.transformer.ln_f( F.softplus(y))
        wt2   = F.softplus(self.lm_head.weight)

        logits = y.matmul(wt2.T)
        logits = logits.log_softmax(-1)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss



class CCM17(GPT):
    '''
    using a GRU for the recurrent

    step 1500: train loss 3.9082, val loss 4.7978

    step 1500: train loss 3.7906, val loss 4.7973

    step 2000: train loss 3.6797, val loss 4.7919

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
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__()

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
        config.nc = 2
        nc= config.nc

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size+2, config.n_embd),
            wpe_2 = nn.Embedding(config.block_size+2, config.n_embd),
            whe = nn.Embedding(config.nc, config.n_embd),
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(ne, ne,bias=config.bias),
            lin_internal_2 = nn.Linear(ne, ne,bias=config.bias),
            # lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            lin_transit_3 = nn.Linear(ne,ne,bias=config.bias),
            lin_transit_4 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=True),
            lin_input_2   = nn.Linear(ne,ne,bias=True),
            lin_input_3   = nn.Linear(ne,ne,bias=True),
            lin_input_4   = nn.Linear(ne,ne,bias=True),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        # nc = self.config.nc
        # head = pos[:nc]
        # forward the GPT model itself
        pos = torch.cat([pos[:1],pos],0)
        idx = torch.cat([idx[:,:1]*0,idx],1)

        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2  = self.transformer.wpe_2(pos) # position embeddings of shape (t, n_embd)

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
        t = t+1

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt        = (torch.zeros((b,t,e),device=device) + pos_emb )
        cpt2       = (torch.zeros((b,t,e),device=device) + pos_emb_2 )
        cpt3       = (torch.zeros((b,t,e),device=device) + pos_emb )
 
        # cpt    = self.transformer.ln_f(F.softplus(cpt)).square()
        # cpt2   = self.transformer.ln_f(F.softplus(cpt2)).square()
        # cpt3   = self.transformer.ln_f(F.softplus(cpt3)).square()

        cpt   = F.softplus(cpt)
        cpt   = cpt / cpt.sum(dim=2,keepdims=True) 
        # cpt2   = F.softplus(cpt2)
        # cpt2   = cpt2 / cpt2.sum(dim=2,keepdims=True) 

        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T

        xnorm = self.transformer.ln_f(F.softplus(x))
        
        # torch.autograd.set_detect_anomaly(True)
        for i in range(4):

            # #### gradient to predict the token
            # ### (b,tp,nh,tc)
            # cpred = md.lin_transit(cpt)
            ft = (md.lin_input(xnorm)   + md.lin_transit(cpt)).sigmoid()
            ht = (md.lin_input_2(xnorm) + md.lin_transit_2(cpt) ).sigmoid()
            # ht = (md.lin_input_2(xnorm) + md.lin_transit_2(cpt) ).tanh()
            cpred = (1-ft).detach()*cpt + ft*ht            
            # cpred = md.lin_transit(cpt)
            
            lp   = 0.
            # lp += torch.einsum('bpe,bce ->bcp', xpred,  xnorm )
            lp  += torch.einsum('bpe,bce->bcp', cpred,  cpt)
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            att  = lp.softmax(-1)
            att1 = att


            # ft = (md.lin_input_3(xnorm)   + md.lin_transit_3(cpt)).sigmoid()
            # ht = (md.lin_input_4(xnorm) + md.lin_transit_4(cpt) ).sigmoid()
            # # ht = (md.lin_input_2(xnorm) + md.lin_transit_2(cpt) ).tanh()
            # cpred2 = (1-ft).detach()*cpt + ft*ht

            # # cpred2 = md.lin_transit_2(cpt)

            # lp   = 0.
            # # lp += torch.einsum('bpe,bce ->bcp', xpred,  xnorm )
            # lp  += torch.einsum('bpe,bce->bcp', cpred2,  cpt)
            # ### (b,tc,tp)
            # lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            # att  = lp.softmax(-1)
            # att2 = att

            dcpt =0.
            dcpt += torch.einsum('bcp,bpe->bce', att1, cpred) 
            # dcpt += torch.einsum('bcp,bpe->bce', att2, cpred2) 
            dcpt += xnorm.matmul(md.lin_output.weight).sigmoid()    
            # dcpt += cpt2.matmul(md.lin_internal.weight)        
            cpt  += 0.3 * (dcpt - cpt)
            # cpt  = self.transformer.ln_f(F.softplus(cpt))
            # cpt   = F.softplus(cpt)
            # cpt   = cpt / cpt.sum(dim=2,keepdims=True) 


        ### inducing the next token using the pos_embedding
        # cpt_ind = cpt
        y = 0.
        # y        = md.lin_input(cpt)
        y       += md.lin_output_3(cpt)

        ### step 1750: train loss 3.9248, val loss 4.8322
        y     = y[:,1:]
        y     = self.transformer.ln_f( F.softplus(y))
        wt2   = F.softplus(self.lm_head.weight)

        logits = y.matmul(wt2.T)
        logits = logits.log_softmax(-1)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss



class CCM18(GPT):
    '''

using gaussian displacement for updates

step 1500: train loss 4.1113, val loss 4.9812


gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in 


class CCM18(GPT):
    '''

using gaussian displacement for updates

step 1500: train loss 4.1113, val loss 4.9812


gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        # super(GPT,self).__init__(config)
        super(GPT,self).__init__()

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
        config.nc = 2
        nc= config.nc
        # config.g = 5
        config.g = 1

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size+2, config.n_embd),
            wpe2 = nn.Embedding(config.block_size+2, config.n_embd),
            wpe3 = nn.Embedding(config.block_size+2, config.n_embd),
            whe  = nn.Embedding(config.nc, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            # lin_transit_k= nn.Linear(ne,config.g*ne,bias=True),
            lin_transit_k= nn.Linear(ne,config.g*ne,bias=False),
            # lin_transit_k  = nn.Embedding(config.n_embd, config.g  *config.n_embd,bias=True),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=False),
            lin_input_2   = nn.Linear(ne,ne,bias=False),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        nc = self.config.nc
        head = pos[:nc]
        # forward the GPT model itself
        pad = 1
        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)

        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2  = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb_3  = self.transformer.wpe3(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2[:pad] = pos_emb_3[:pad]

        x = tok_emb + pos_emb
        # x = tok_emb 
        if self.config.use_dropout:
            x = self.transformer.drop(x)


        lp_internal = torch.zeros((b,),device=device)
        #### the problem here is t[5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        # super(GPT,self).__init__(config)
        super(GPT,self).__init__()

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
        config.nc = 2
        nc= config.nc
        # config.g = 5
        config.g = 1

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size+2, config.n_embd),
            wpe2 = nn.Embedding(config.block_size+2, config.n_embd),
            wpe3 = nn.Embedding(config.block_size+2, config.n_embd),
            whe  = nn.Embedding(config.nc, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            lin_transit_k= nn.Linear(ne,config.g*ne,bias=True),
            # lin_transit_k  = nn.Embedding(config.n_embd, config.g  *config.n_embd,bias=True),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=False),
            lin_input_2   = nn.Linear(ne,ne,bias=False),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        nc = self.config.nc
        head = pos[:nc]
        # forward the GPT model itself
        pad = 1
        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)

        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2  = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb_3  = self.transformer.wpe3(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2[:pad] = pos_emb_3[:pad]

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
        t = t+pad

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt      = (torch.zeros((b,t,e),device=device) + pos_emb_2 )
 
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T
        # cpt   = self.transformer.ln_f(F.softplus(cpt))

        cpt = F.softplus(cpt)
        cpt = cpt / cpt.sum(dim=2,keepdims=True)

        xnorm = self.transformer.ln_f(F.softplus(x))
        knorm = md.wke.weight.T
        knorm = knorm - knorm.sum(dim=0,keepdim=True)

        g = self.config.g

        for i in range(4):
            ### (b,c,e),(e,k)->bcek
            ### (bcek)
            # cpred = cpt + xnorm.matmul(md.lin_output_2.weight)
            cpred = cpt# + xnorm.matmul(md.lin_output_2.weight)
            # cpred = cpred.unsqueeze(-1) + knorm[None,None] 
            cpred = md.lin_transit_k(cpt).reshape((b,t,e,g))
            cpred = (F.softplus(cpred))
            cpred = cpred / cpred.sum(dim=2,keepdims=True)

            # cpred = md.lin_move( cpred )
            # cpred = md.lin_transit_2(cpt)
            # cpred = self.transformer.ln_f(F.softplus(cpred))

            ### (b,tp,nh,tc)
            lps  = 0.
            lp   = 0.
            #### gradient to predict the token

            # lp += torch.einsum('bpe,bce ->bcp', xpred,  xnorm )
            lp  += torch.einsum('bpek,bce->bcpk', cpred,  cpt)

            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:,None]==0,float('-inf'))
            att  = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)
            # lps += lp.logsumexp(2).mean()            

            dcpt = 0.
            dcpt += torch.einsum('bcpk,bpek->bce', att, cpred)
            #  + xnorm.matmul(md.lin_output_2.weight)
            dcpt += xnorm.matmul(md.lin_output_2.weight)
            # dcpt = dcpt * 0.5

            # dcpt = self.transformer.ln_f(F.softplus(dcpt))
            # cpt  = cpt  +  0.3 * (dcpt - cpt)
            cpt  +=  0.3 * (dcpt - cpt)
            cpt = F.softplus(cpt)
            cpt = cpt / cpt.sum(dim=2,keepdims=True)


        ### inducing the next token using the pos_embedding
        cpt_ind = cpt
        y       = md.lin_output_3(cpt_ind)

        ### step 1750: train loss 3.9248, val loss 4.8322
        y     = y[:,pad:]
        y     = self.transformer.ln_f( F.softplus(y))
        wt2   = F.softplus(self.lm_head.weight)

        logits = y.matmul(wt2.T)
        logits = logits.log_softmax(-1)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss   



class CCM19(GPT):
    '''

using gaussian displacement for updates

step 1500: train loss 4.1113, val loss 4.9812


gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        # super(GPT,self).__init__(config)
        super(GPT,self).__init__()

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
        config.nc = 2
        nc= config.nc
        config.g = 25
        # config.g = 1
        # config.g = 1

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size+2, config.n_embd),
            wpe2 = nn.Embedding(config.block_size+2, config.n_embd),
            wpe3 = nn.Embedding(config.block_size+2, config.n_embd),
            whe  = nn.Embedding(config.nc, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            # lin_transit_k= nn.Linear(ne,config.g*ne,bias=True),
            lin_transit_k= nn.Linear(ne,config.g*ne,bias=False),
            # lin_transit_k  = nn.Embedding(config.n_embd, config.g  *config.n_embd,bias=True),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=False),
            lin_input_2   = nn.Linear(ne,ne,bias=False),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        nc = self.config.nc
        head = pos[:nc]
        # forward the GPT model itself
        pad = 6
        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)


        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2  = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb_3  = self.transformer.wpe3(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2[:pad] = pos_emb_3[:pad]

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

        ### cpt is the post context at position 
        t = t+pad

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt      = (torch.zeros((b,t,e),device=device) + pos_emb_2 )
 
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T
        # cpt   = self.transformer.ln_f(F.softplus(cpt))
        cpt   = self.transformer.ln_f(cpt)

        # xnorm = self.transformer.ln_f(F.softplus(x))
        xnorm = self.transformer.ln_f(x)

        knorm = md.wke.weight
        knorm = knorm - knorm.sum(dim=1,keepdim=True)
        knorm = self.transformer.ln_f(knorm).T

        vnorm = md.wve.weight
        vnorm = self.transformer.ln_f(vnorm).T

        g = self.config.g

        # for i in range(4):
        for i in range(4):
            ### (b,c,e),(e,k)->bcek
            ### (bcek)

            # cpred = cpt.unsqueeze(-1) + 0.5 * vnorm[None,None] 

            # cpred = (cpt + 0.5 * md.lin_transit_2(cpt)).unsqueeze(-1)
            # cpred = md.lin_transit_k(cpt).reshape((b,t,e,g))
            # cpred = 0.5*cpred + cpt.unsqueeze(-1)
            cpred =  cpt.unsqueeze(-1) + 0.5 * vnorm[None,None] 
            ### (b,tp,nh,tc)
            lp   = 0.
            #### gradient to predict the token

            lp += torch.einsum('bpek,bce ->bcpk', cpred,  cpt )
            # lp  += torch.einsum('bpe,bce->bcp',  cpt,      cpt).unsqueeze(-1)
            lpv  = torch.einsum('bpe,ek->bpk',   xnorm,  knorm)[:,None]
            lp   = lp+ lpv

            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:,None]==0,float('-inf'))
            att  = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)
            # print(att.mean((0,1)).sum(0))

            dcpt = 0.
            dcpt += torch.einsum('bcpk,bpek->bce', att, cpred)
            #  + xnorm.matmul(md.lin_output_2.weight)
            dcpt += x.matmul(md.lin_output_2.weight)
            # dcpt = dcpt * 0.5

            # dcpt = self.transformer.ln_f(F.softplus(dcpt))
            # cpt  = cpt  +  0.3 * (dcpt - cpt)
            cpt  +=  0.3 * (dcpt - cpt)
            # cpt   = self.transformer.ln_f((cpt))

            # cpt = F.softplus(cpt)
            # cpt = cpt / cpt.sum(dim=2,keepdims=True)


        ### inducing the next token using the pos_embedding
        cpt_ind = cpt
        y       = md.lin_output_3(cpt_ind) + cpt

        # ### step 1750: train loss 3.9248, val loss 4.8322
        y     = y[:,pad:]
        y     = self.transformer.ln_f((y))
        # wt2   = F.softplus(self.lm_head.weight)

        # logits = y.matmul(wt2.T)
        # logits = logits.log_softmax(-1)

        logits = self.lm_head(y)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss   


class CCM20(GPT):
    '''

simple linear predictor

step 1500: train loss 4.1113, val loss 4.9812


gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        # super(GPT,self).__init__(config)
        super(GPT,self).__init__()

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
        config.nc = 2
        nc= config.nc
        # config.g = 5
        config.g = 1

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size+2, config.n_embd),
            wpe2 = nn.Embedding(config.block_size+2, config.n_embd),
            wpe3 = nn.Embedding(config.block_size+2, config.n_embd),
            whe  = nn.Embedding(config.nc, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            lin_transit_k= nn.Linear(ne,config.g*ne,bias=True),
            # lin_transit_k  = nn.Embedding(config.n_embd, config.g  *config.n_embd,bias=True),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=False),
            lin_input_2   = nn.Linear(ne,ne,bias=False),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        nc = self.config.nc
        head = pos[:nc]
        # forward the GPT model itself
        pad = 0
        # pos = torch.cat([pos[:pad],pos],0)
        # idx = torch.cat([idx[:,:pad]*0,idx],1)


        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2  = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb_3  = self.transformer.wpe3(pos) # position embeddings of shape (t, n_embd)
        # pos_emb_2[:pad] = pos_emb_3[:pad]

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
        t = t+pad

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt      = (torch.zeros((b,t,e),device=device) + pos_emb_2 )
 
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T
        # cpt   = self.transformer.ln_f(F.softplus(cpt))
        cpt   = self.transformer.ln_f(cpt)

        # xnorm = self.transformer.ln_f(F.softplus(x))
        xnorm = self.transformer.ln_f(x)

        g = self.config.g

        # for i in range(4):
        for i in range(1):
        # for i in range(6):
            ### (b,c,e),(e,k)->bcek
            ## (bcek)
            dcpt  = 0.
            dcpt += x.matmul(md.lin_output_2.weight)
            # dcpt = dcpt * 0.5

            # dcpt = self.transformer.ln_f(F.softplus(dcpt))
            # cpt  = cpt  +  0.3 * (dcpt - cpt)
            # cpt  +=  0.3 * (dcpt - cpt)
            # cpt   = self.transformer.ln_f((cpt))
            cpt = 0.5 * (cpt + dcpt)


        ### inducing the next token using the pos_embedding
        cpt_ind = cpt
        # y       = md.lin_output_3(cpt_ind)
        y       = cpt + md.lin_output_3(cpt_ind)

        ### step 1750: train loss 3.9248, val loss 4.8322
        # y     = y[:,pad:]
        # y     = self.transformer.ln_f( F.softplus(y))
        y     = self.transformer.ln_f( (y))
        wt2   = (self.lm_head.weight)

        logits = y.matmul(wt2.T)
        logits = logits.log_softmax(-1)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss   


class CausalSelfAttention13(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
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
        self.flash = False
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # v = x[:,:,None].transpose(1,2) + v 
        # v = 

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block23(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_1 = LayerNorm(config.n_embd, bias=True)
        self.attn = CausalSelfAttention13(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=True)
        self.mlp  = MLP(config)
        # self.wn   = nn.Parameter(nn.Linear(config.n_embed,1).weight[0])
        
    def forward(self, x):
        return self.attn(self.ln_1(x)) + self.mlp(self.ln_2(x))

        # x = x + self.attn(self.ln_1(x)) + self.mlp(self.ln_2(x))
        # # x = x + self.mlp(self.ln_2(x))
        # return x


class CCM23(GPT):
    '''

using gaussian displacement for updates

step 1500: train loss 4.1113, val loss 4.9812


gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # config.dropout = 0.0
        # print (config)
        config.n_head =  1
        # config.block_size = config.block_size + 3

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            wpe2 = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself

        # assert 0
        # pad = 17
        pad = 0
        # pos = torch.cat([pos[:pad],pos],0)
        # idx = torch.cat([idx[:,:pad]*0,idx],1)

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        # pos_emb[:pad] = pos_emb_2[:pad]
        # tok_emb[:,:pad] = pos_emb_2[:pad]

        x = self.transformer.drop(tok_emb + pos_emb)
        
        # x = self.transformer.ln_f(x)
        # x = self.transformer.h[0].ln_1(x)
        
        dx = 0.
        nstep = 2
        for ii in range(nstep):
            # xn = block.ln_1(x)
            xn = x - x.mean(-1,keepdims=True)
            xn = xn/(0.001 + xn.std(-1,keepdims=True))

            # for i,block in enumerate(self.transformer.h[:3]):
            for i,block in enumerate(self.transformer.h):
                # xn = x * block.ln_1.weight[None,None] 
                dx = dx+ block.attn(xn)
            dx = dx+ block.mlp(xn)
            x  = x+ dx


        x = self.transformer.ln_f(x)
        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


class CCM26(GPT):
    '''
    using gaussian displacement for updates

    step 1500: train loss 4.1113, val loss 4.9812

    gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # config.dropout = 0.0
        # print (config)
        config.n_head =  1
        # config.block_size = config.block_size + 3
        # config.g = 100
        # config.g = 700
        # config.g = config.n_embd
        config.g = 100
        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            wpe2 = nn.Embedding(config.block_size, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve   = nn.Embedding(config.g, config.n_embd*ng),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd*ng, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0
 
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask = torch.tril(att_mask, diagonal=-1)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True


        md = self.transformer

        knorm = md.wke.weight
        # knorm = knorm - knorm.sum(dim=1,keepdim=True)
        # knorm = self.transformer.ln_f(knorm)

        # knorm = md.wke.weight.T
        vnorm = md.wve.weight
        vnorm2 = md.wve2.weight
        # vnorm = self.transformer.ln_f(vnorm).T

        dx = 0.
        nstep = 1
        '''
        step 1000: train loss 3.9572, val loss 4.7648
        step 1250: train loss 3.6906, val loss 4.7936

        step 1000: train loss 4.1372, val loss 4.8331
        step 1250: train loss 3.8674, val loss 4.8308

        ### the temperature of the classifier is very important (of course)

        step 1000: train loss 3.9551, val loss 4.7831

        ### no k vect
        step 1000: train loss 4.0397, val loss 4.8065


        '''

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        for ii in range(nstep):
            xn = x - x.mean(-1,keepdims=True)
            xn = xn/(0.001 + xn.std(-1,keepdims=True))

            for i,block in enumerate(self.transformer.h):
                attn = block.attn
                xq, k, v  = attn.c_attn(xn).split(attn.n_embd, dim=2)

                # xq = xn + md.wke.weight[i,None,None,:]
                # xq    
                # xq = xq - xq.mean(-1,keepdims=True)
                xq = xq / (0.001 + xq.std(-1,keepdims=True))

                att = xq@xn.transpose(2,1)/attn.n_embd**0.5
                att = att.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                dx = dx + att.softmax(-1) @ v                
                # dx = dx + block.attn(xn)

            # dx = dx + F.clip(block.mlp(xn),-1,1)  
            dx = dx + (block.mlp(xn))
            # dx = dx - 0.5* x

            # knormm = self.transformer.ln_f(knorm)
            # att = torch.einsum('bce,ke->bck', xn, knormm).softmax(-1)
            # # dxx = torch.einsum( 'bck,ke->bce',att, vnorm.reshape((g,ng,e))[:,0], )
            # dxx = torch.einsum('bck,kve,bce->bcvk', att, vnorm.reshape((g,ng,e)), xn)
            # # dxx = torch.einsum('bck,kve,bce->bcvk', att, vnorm.reshape((g,ng,e)), xn - torch.einsum('bck,ke->bce',att, knormm ))
            # dxx = torch.einsum('bcvk,kve->bce', dxx, vnorm2.reshape((g,ng,e)))
            # dx += dxx
        
            x  = x + dx



        ### being constrained on a hyper-sphere is crucial for the effectiveness
        ### probably because of both the way relation is manifested 
        ### and because of the way 
        ### meanNorm only        step 1500: train loss 3.6714, val loss 5.0438
        ### stdNorm only         step 1500: val loss 4.88
        ### meanNorm and stdNorm step 1000: train loss 4.0157, val loss 4.7911

        ### this means that the points are overparametrised expression of lower dimensional manifold



        # assert 0  
        xn = x
        xn = xn - xn.mean(-1,keepdims=True)   
        xn = xn/(0.001 + xn.std(-1,keepdims=True))  
        x = xn
        # x = self.transformer.ln_f(x)        
        x = x[:,pad:]



        if targets is not None:
            # if we are given some desired targets also calculate the loss

            ### using  gaussian emission here
            logits = self.lm_head(x) - self.lm_head.weight.square().sum(-1)[None,None,] - x.square().sum(-1,keepdims=True)
            #   logits = self.lm_head(x)
                    
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss




class CCM27(GPT):
    '''
    using gaussian displacement for updates

    step 1500: train loss 4.1113, val loss 4.9812

    gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # config.dropout = 0.0
        # print (config)
        config.n_head =  1
        # config.block_size = config.block_size + 3
        # config.g = 100
        # config.g = 700
        # config.g = config.n_embd
        config.g = 100
        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            wpe2 = nn.Embedding(config.block_size, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve   = nn.Embedding(config.g, config.n_embd*ng),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0
 
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True


        md = self.transformer

        knorm   = md.wke.weight
        # knorm = knorm - knorm.sum(dim=1,keepdim=True)
        # knorm = self.transformer.ln_f(knorm)
        # knorm = md.wke.weight.T
        vnorm   = md.wve.weight
        vnorm2  = md.wve2.weight
        # vnorm = self.transformer.ln_f(vnorm).T

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd

        xn = x - x.mean(-1,keepdims=True)
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        x = xn

        for ii in range(nstep):
            dx = 0.

            for i,block in enumerate(self.transformer.h):
                attn      = block.attn
                xq, k, v  = attn.c_attn(xn).split(attn.n_embd, dim=2)
                att       = xq@xn.transpose(2,1)/attn.n_embd**0.5
                att       = att.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                dx        = dx + att.softmax(-1) @ v                
                # dx = dx + block.attn(xn)

            dx = dx + block.mlp(xn)

            # knormm = self.transformer.ln_f(knorm)
            # att = torch.einsum('bce,ke->bck', xn, knormm).softmax(-1)
            # # dxx = torch.einsum( 'bck,ke->bce',att, vnorm.reshape((g,ng,e))[:,0], )
            # dxx = torch.einsum('bck,kve,bce->bcvk', att, vnorm.reshape((g,ng,e)), xn)
            # # dxx = torch.einsum('bck,kve,bce->bcvk', att, vnorm.reshape((g,ng,e)), xn - torch.einsum('bck,ke->bce',att, knormm ))
            # dxx = torch.einsum('bcvk,kve->bce', dxx, vnorm2.reshape((g,ng,e)))
            # dx += dxx

            ### step 1000: train loss 3.9426, val loss 4.7900
            ### step 1000: train loss 3.9426, val loss 4.7900

            ### step 1000: train loss 3.9660, val loss 4.7700
            ### step 1500: train loss 3.8446, val loss 4.7721
            ### step 1000: train loss 3.9033, val loss 4.7393

            ### step 1000: train loss 4.0026, val loss 4.7452


            ### state switching is important for the effectiveness of transformer
            # dx = dx.clip(-0.3,0.3)
            # dx = dx.clip(-0.7,0.7)

            x0 = x

            # x  = x + dx
            x = md.proj(x) + dx
            '''
            This model is characterised by the usage 
            of hypersphere as the phase space, and using 
            linear transformation and compression to 
            parameterise the dynamics.

            The ultmate objective is to 
            parametrise a dynamical system that could 
            capture the dynamics in human language.
            This means to find a phase space, and 
            a paramterised transfer function on this 
            phase space.

            Self critical means to enable the 
            model to decide whether the token should be 
            outputed at a particular stage, this could be 
            a linear function  on the phase space.

            What is the primary objective here?
            The primary objective is to understand 
            that how we can better parametrise 
            the phase space? how can we squash the 
            output space back into the input space?

            One possibility would be to partition 
            the output space explicitly, for example
            to be the points on simplectical surface.


            Using simplectical surface has the advantage 
            that we can represent new points to be 
            mixture of prototypic points. And we can 
            locate any given point according to their
            proximity to prototypes.

            There are two types of vectors in sympletitcal surface,
            one is which sum to one, one is which sum to zero. 
            Thus, we can parametrise the whole space to be 
            the position vector, and the displacement vector.
            when summing the positional vector, we have to keep 
            the factors sum to 1. when summing the displacement vector,
            we have to keep that all values are above zero.

            There is another alternative, is to use a clipped 
            hyperspace. Whenever the point goes out of the hypercube,
            we put it back into the box by dividing by the maximal
            norm.

            This means to perform some 

            ### hypothesis
            ### the norm of the state vector is related to the certainty 
            ### of that particular state.    
            ### a vector with a small norm transduce more movement to post-norm vector
            ### whereas a vector with a big norm transduce less movement to post-norm vector            

            '''


            xn  = x - x.mean(-1,keepdims=True)
            xn  = xn/(0.001 + xn.std(-1,keepdims=True))
            ddx = (xn - x0)
            # assert 0
            # print([x0.square().mean(-1).mean().item(), 
            #     dx.square().mean(-1).mean().item(),
            #     ddx.square().mean(-1).mean().item(),
            #     ])
            x = xn

        x = self.transformer.ln_f(x)        
        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


class CCM30(GPT):
    '''
    using gaussian displacement for updates

    step 1500: train loss 4.1113, val loss 4.9812

    gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # config.dropout = 0.0
        # print (config)
        config.n_head =  1
        # config.block_size = config.block_size + 3
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.nh = 3
        # ng = 2
        config.ng = 4
        ng = config.ng
        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            wpe2 = nn.Embedding(config.block_size, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd*ng),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nl+1)*ne, ne*ng,bias=False),
            alayer1  = nn.Linear( ne, ne ,bias=False),
            alayer2  = nn.Linear( ne, ne ,bias=False),
            alayer3  = nn.Linear( ne, ne ,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0
 
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True


        md = self.transformer

        knorm   = md.wke.weight
        # knorm = knorm - knorm.sum(dim=1,keepdim=True)
        # knorm = md.wke.weight.T
        vnorm   = md.wve.weight
        vnorm2  = md.wve2.weight
        # vnorm = self.transformer.ln_f(vnorm).T

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = x
        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer

        y = torch.zeros((b,t,(nl+1)*ne),device=device)
        for ii in range(nstep):
            dx = 0.

            for i,block in enumerate(self.transformer.h):
                attn      = block.attn
                xq, k, v  = attn.c_attn(xn).split(attn.n_embd, dim=2)
                att       = xq@xn.transpose(2,1)/attn.n_embd**0.5
                att       = att.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                att_xn       = att.softmax(-1) @ xn                
                # ddx       = att.softmax(-1) @ xn                
                dx        = dx +  attn.c_attn(att_xn)[:,:,2*ne:3*ne]

                y[:,:,i*ne:(i+1)*ne] = att_xn
                # dx = dx + block.attn(xn)
            '''
            step 1250: train loss 3.7433, val loss 4.7538  
            step 1250: train loss 3.8416, val loss 4.7337

            step 1250: train loss 3.7150, val loss 4.7379
            step 1250: train loss 3.7230, val loss 4.7065

            step 1250: train loss 3.6699, val loss 4.7085
            ### factor3, 3linearhead
            step 1250: train loss 3.6814, val loss 4.6993

            ### factor3, 3linearhead
            step 1250: train loss 3.7068, val loss 4.6834


            ### no gating 
            ### step 1250: train loss 3.8193, val loss 4.8412

            ### dynamic gating?

            GPT01 layer1
            step 1250: train loss 3.8768, val loss 4.8289


            '''

            i += 1
            y[:,:,i*ne:(i+1)*ne] = xn



            dx = md.flayer(y).reshape((b,t,e,ng))[:,:,:,:]
            dx1 = dx[:,:,:,0]
            dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))
            

            dx2 = dx[:,:,:,1] + md.alayer1(dx1)
            dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))

            dx3 = dx[:,:,:,2] + md.alayer2(dx1) + md.alayer3(dx2)
            # dx3 = dx[:,:,:,2] + md.alayer1(dx2)
            dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))


            # dx4 = dx[:,:,:,3]
            # dx4 = dx4/(0.001 + dx4.std(-1,keepdims=True))
            # dx4 = dx4*0.0001
            # ### the larger the scale, the colder and sharper the gate

            # # gate = dx4.softmax(-1)*ne

            dx = 9*dx1+ 3 * dx2 + 1 * dx3

            # dx = dx1+ 4 * dx2 + 16 * dx3

            # # dx1 = dx
            # dx = md.flayer(y).reshape((b,t,e,ng))[:,:,:,0]
            
            # dg = (md.glayer(y)/(self.config.n_embd**0.5)).softmax(-1)
            # gk = md.wge.weight
            # gk = gk / (0.001 + gk.std(-1,keepdims=True))
            # # gk = gk * 0.01
            # gk = gk * 0.05
            # gate = torch.einsum('ge,btg->bte', gk, dg).softmax(-1)*ne


            x0 = x
            x  = dx

            xn = x
            # xn = x - x.mean(-1,keepdims=True)
            # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
            xn = xn/(0.001 + xn.std(-1,keepdims=True))
            ddx = (xn - x0)
            x = xn

        # x = x/(0.001 + x.std(-1,keepdims=True))
        # x = x*gate

        x = self.transformer.ln_f(x)        
        
        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


class CCM31(GPT):
    '''
    using gaussian displacement for updates

    step 1500: train loss 4.1113, val loss 4.9812

    gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        # config.block_size = config.block_size + 3
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 4
        ng = config.ng
        nh = config.n_head
        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            wpe2 = nn.Embedding(config.block_size, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            alayer1  = nn.Linear( ne, ne ,bias=False),
            alayer2  = nn.Linear( ne, ne ,bias=False),
            alayer3  = nn.Linear( ne, ne ,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                v_attn = nn.Linear(ne, ne,bias=False),
                k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                flayer = nn.Linear(ne*(nh+1)*2,    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0
 
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True


        md = self.transformer

        knorm   = md.wke.weight
        # knorm = knorm - knorm.sum(dim=1,keepdim=True)
        # knorm = md.wke.weight.T
        vnorm   = md.wve.weight
        vnorm2  = md.wve2.weight
        # vnorm = self.transformer.ln_f(vnorm).T

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = x
        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head


        '''
        step 1250: train loss 3.7433, val loss 4.7538  
        step 1250: train loss 3.8416, val loss 4.7337

        l5
        step 1250: train loss 3.7540, val loss 4.7746
        step 1250: train loss 3.7779, val loss 4.7469

        kernel gate
        step 1250: train loss 3.7205, val loss 4.7199

        ## nogate 
        step 1250: train loss 3.7494, val loss 4.6962

        step 1250: train loss 3.7550, val loss 4.7598


        step 1000: train loss 3.9051, val loss 4.7238
        step 1000: train loss 3.9517, val loss 4.7147
        step 1000: train loss 3.9760, val loss 4.7229




        '''

        def get_context(block, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            ### maybe adding channel to gate the context?
            ### getting nh neighbors, and maybe save to the memory?

            ### output is (b,t,nh*e)
            ### by computing (b,t,nh,t) attention from (b,t,nh,e) and (b,t,e)

            ### gating the attention with a linear head or a kernel

            ### hypothesis is that the model should consult internal knowledge 
            ### during its deduction of next word.
            ###  internal knowledge is reflected as the 

            Adding knowledge into the model meaning to bias the model dynamics
            

            '''
            xk = block.k_attn(xq).reshape((b,t,nh,e))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            # gate = torch.einsum('btke,bte->btk',block.k_gate(xq).reshape((b,t,nh,e)),xq/ne**0.5).sigmoid()
            # att  = torch.einsum('bktp,btk->bktp',att,gate)

            y    = torch.einsum('bktp,bpe->btke',att,xn)
            y    = y.reshape((b,t,-1))
            y    = torch.cat([xn,y],dim=-1).reshape((b,t,-1,e))

            # xv = block.v_attn(y)
            # wv = md.wve.weight
            # wv = wv / (0.001+wv.std(-1,keepdims=True))

            # wk = md.wke.weight
            # wk = wk / (0.001+wk.std(-1,keepdims=True))

            # att = torch.einsum('btke,ge->btkg',xv, wk /ne**0.5).softmax(-1)
            # xv  = torch.einsum('btkg,ge->btke', att, wv )#.reshape((b,t,-1))
            # y   = torch.cat([y,xv],dim=2).reshape((b,t,-1))

            y   = torch.cat([y,y],dim=2).reshape((b,t,-1))


            # print(y.shape)


            # ### (b,t,k)
            # y    = torch.einsum('bktp,bpe->btke',att,xn)
            # gate = torch.einsum('btke,btke->btk',block.k_gate(xq).reshape((b,t,nh,e)), y).sigmoid()
            # y    = torch.einsum('btk,btke->btke',gate, y)
            # y    = y.reshape((b,t,-1))
            # y    = torch.cat([xn,y],dim=-1)

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        dx1 = xn
        # dx2 = md.x2(xn)
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        for i, fblock in enumerate(( self.transformer.h)):
            y1 = get_context(block0, dx1, xn)
            dx1 = fblock.flayer(y1)
            dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            

        x = dx1 

        x0 = x

        xn = x
        # xn = x - x.mean(-1,keepdims=True)
        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn

        # x = x/(0.001 + x.std(-1,keepdims=True))
        # x = x*gate

        x = self.transformer.ln_f(x)        
        
        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        


class CCM32(GPT):
    '''
    using gaussian displacement for updates

    step 1500: train loss 4.1113, val loss 4.9812

    gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        # config.block_size = config.block_size + 3
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 4
        ng = config.ng
        nh = config.n_head
        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            wpe2 = nn.Embedding(config.block_size, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd*ng),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            alayer1  = nn.Linear( ne, ne ,bias=False),
            alayer2  = nn.Linear( ne, ne ,bias=False),
            alayer3  = nn.Linear( ne, ne ,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                flayer = nn.Linear(ne*(nh+1)*2,    2*ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0
 
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True


        md = self.transformer

        knorm   = md.wke.weight
        # knorm = knorm - knorm.sum(dim=1,keepdim=True)
        # knorm = md.wke.weight.T
        vnorm   = md.wve.weight
        vnorm2  = md.wve2.weight
        # vnorm = self.transformer.ln_f(vnorm).T

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = x
        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head


        '''
        step 1250: train loss 3.7433, val loss 4.7538  
        step 1250: train loss 3.8416, val loss 4.7337

        iter 1240: loss 3.8648, time 66.89ms, mfu 7.23%
        sigmoids
        step 1250: train loss 3.7345, val loss 4.7029
        saving checkpoint to out-shakespeare-word
        step 1250: train loss 3.6893, val loss 4.6879

        step 1000: train loss 3.8447, val loss 4.7052


        GPT01 layer1
        step 1250: train loss 3.8768, val loss 4.8289
        
        step 1250: train loss 3.7084, val loss 4.7064



        step 1250: train loss 3.7176, val loss 4.6754

        CCM31
        layer 5
        step 1250: train loss 3.7169, val loss 4.7016

        layer 6
        step 1250: train loss 3.7191, val loss 4.7037

        layer 7
        step 1250: train loss 3.7092, val loss 4.7106


        ### unable to resue the context calculated earlier?
        ### this could become possible if we allow context to be passed
        ### forward?

        step 1250: train loss 3.6943, val loss 4.7105

        CCM31 L3
        ### step 1250: train loss 3.7029, val loss 4.7227

        CCM31 L5
        step 1250: train loss 3.7169, val loss 4.7016        


        ### 
        CCM32 L5
        step 1000: train loss 3.8914, val loss 4.7396

        CCM32 L6
        step 1250: train loss 3.6742, val loss 4.7165

        GPT L7
        ### gpt is capable of resusing earlier deduction to make current deduction
        step 1250: train loss 3.5953, val loss 4.6015




        '''

        def get_context(block, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            ### maybe adding channel to gate the context?
            ### getting nh neighbors, and maybe save to the memory?

            ### output is (b,t,nh*e)
            ### by computing (b,t,nh,t) attention from (b,t,nh,e) and (b,t,e)
            
            '''
            xk = block.k_attn(xq).reshape((b,t,nh,e))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)
            y = torch.einsum('bktp,bpe->btke',att,xn).reshape((b,t,-1))
            y = torch.cat([xn,y],dim=-1)
            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        dx1 = xn
        dx2 = md.x2(xn)
        dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        for i, fblock in enumerate(( self.transformer.h)):
            y1 = get_context(block0, dx1, xn)
            y2 = get_context(block0, dx2, xn)
            # y = get_context(fblock, dx, xn)
            # dx = self.transformer.h[i].flayer(y)
            # dx = fblock.flayer(torch.cat([y1,y2],dim=-1))
            dx1,dx2 = torch.split( fblock.flayer(torch.cat([y1,y2],dim=-1)),ne,dim=-1)
            dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
            dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            

        x = dx1 

        x0 = x
        # x  = dx

        xn = x
        # xn = x - x.mean(-1,keepdims=True)
        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn

        # x = x/(0.001 + x.std(-1,keepdims=True))
        # x = x*gate

        x = self.transformer.ln_f(x)        
        
        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        

class CCM33(GPT):
    '''
    use a chain to equalise the information before inference
    using gaussian displacement for updates
    the pre-equalisation proves not useful. CCM31 is better


    gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        # config.block_size = config.block_size + 3
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 4
        ng = config.ng
        nh = config.n_head
        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            wpe2 = nn.Embedding(config.block_size, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            alayer1  = nn.Linear( ne, ne ,bias=False),
            alayer2  = nn.Linear( ne, ne ,bias=False),
            alayer3  = nn.Linear( ne, ne ,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            input_1  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            h  = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                v_attn = nn.Linear(ne, ne,bias=False),
                k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),

            h2 = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                v_attn = nn.Linear(ne, ne,bias=False),
                k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),


            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0
 
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True


        md = self.transformer

        knorm   = md.wke.weight
        # knorm = knorm - knorm.sum(dim=1,keepdim=True)
        # knorm = md.wke.weight.T
        vnorm   = md.wve.weight
        vnorm2  = md.wve2.weight
        # vnorm = self.transformer.ln_f(vnorm).T

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = x
        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head


        '''

        ## nogate 
        step 1250: train loss 3.7494, val loss 4.6962

        step 1250: train loss 3.7550, val loss 4.7598


        step 1000: train loss 3.9051, val loss 4.7238
        step 1000: train loss 3.9517, val loss 4.7147
        step 1000: train loss 3.9760, val loss 4.7229


        
        ### passed information between nodes.
        step 1000: train loss 3.9149, val loss 4.6693
        step 1250: train loss 3.7165, val loss 4.6608

        iter 990: loss 3.9544, time 130.84ms, mfu 6.35%
        step 1000: train loss 3.8987, val loss 4.6777

        step 750: train loss 3.8197, val loss 4.7255


        '''

        def get_context(block, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            ### maybe adding channel to gate the context?
            ### getting nh neighbors, and maybe save to the memory?

            ### output is (b,t,nh*e)
            ### by computing (b,t,nh,t) attention from (b,t,nh,e) and (b,t,e)

            ### gating the attention with a linear head or a kernel

            ### hypothesis is that the model should consult internal knowledge 
            ### during its deduction of next word.
            ###  internal knowledge is reflected as the 

            Adding knowledge into the model meaning to bias the model dynamics
            

            '''
            xk = block.k_attn(xq).reshape((b,t,nh,e))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            # gate = torch.einsum('btke,bte->btk',block.k_gate(xq).reshape((b,t,nh,e)),xq/ne**0.5).sigmoid()
            # att  = torch.einsum('bktp,btk->bktp',att,gate)

            y    = torch.einsum('bktp,bpe->btke',att,xn)
            y    = y.reshape((b,t,-1))
            y    = torch.cat([xn,y],dim=-1).reshape((b,t,-1,e))

            y   = y.reshape((b,t,-1))


            return y

        # for ii in range(nstep):
        dx = 0.

        x0 = torch.zeros((b,t,ne),device=device)+pos_emb
        block0= self.transformer.h2[0]
        # nt = 5
        nt = 2
        # nt = 10
        for i in range(nt):
            ### perform gradient-ish optimisation of the 

            y =  get_context(block0, x0 , x0)
            # dx1 = 0.5 * (x0 + block0.flayer(y)+ md.input_1(xn)) 
            dx1 = 0.5 * (x0 + y.reshape((b,t,-1,e)).mean(2)+ md.input_1(xn)) 
            dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
            x0 = dx1

        # dx1 = x0
        xn = x0

        block0 = self.transformer.h[0]
        dx1 = xn
        # dx2 = md.x2(xn)
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        for i, fblock in enumerate(( self.transformer.h)):
            y1 = get_context(block0, dx1, xn)
            dx1 = fblock.flayer(y1)
            dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            

        x = dx1 

        x0 = x

        xn = x
        # xn = x - x.mean(-1,keepdims=True)
        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn

        # x = x/(0.001 + x.std(-1,keepdims=True))
        # x = x*gate

        x = self.transformer.ln_f(x)        
        
        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        



class CCM34(GPT):
    '''
    using gaussian displacement for updates

    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 4
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292


        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            alayer1  = nn.Linear( ne, ne ,bias=False),
            alayer2  = nn.Linear( ne, ne ,bias=False),
            alayer3  = nn.Linear( ne, ne ,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                v_attn = nn.Linear(ne, ne,bias=False),
                k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        # lreg_factor = 0.01
        # lreg_factor = 0.001
        # lreg_factor = 0.00001
        def regnorm(weight):
            # return (weight.square().mean() - 1.).abs()*weight.numel()
            # return (weight.square().sum() - weight.numel()).abs()
            # return (weight.square().mean() - 1).abs()
            # return (weight.abs().sum())
            return 0
            # square().mean() - 1).abs()
            # weight.numel()).abs()

        lreg += regnorm(self.transformer.wte.weight)
        lreg += regnorm(self.transformer.wpe.weight)
        lreg += regnorm(self.transformer.wpe2.weight)
        # lreg += self.transformer.wte.weight.square().sum()
        # lreg += self.transformer.wpe.weight.square().sum()
        # lreg += self.transformer.wpe2.weight.square().sum()


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = x
        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head


        '''
        step 1250: train loss 3.7433, val loss 4.7538  
        step 1250: train loss 3.8416, val loss 4.7337

        l5
        step 1250: train loss 3.7540, val loss 4.7746
        step 1250: train loss 3.7779, val loss 4.7469

        kernel gate
        step 1250: train loss 3.7205, val loss 4.7199

        ## nogate 
        step 1250: train loss 3.7494, val loss 4.6962

        step 1250: train loss 3.7550, val loss 4.7598


        step 1000: train loss 3.9051, val loss 4.7238
        step 1000: train loss 3.9517, val loss 4.7147
        step 1000: train loss 3.9760, val loss 4.7229

        step 1250: train loss 3.7103, val loss 4.6997
        step 1250: train loss 3.7087, val loss 4.6985
        step 1500: train loss 3.5430, val loss 4.6979

        step 1000: train loss 3.9286, val loss 4.7164

        
        step 1000: train loss 3.9506, val loss 4.7399


        single head, L5
        step 1500: train loss 3.6509, val loss 4.7164
        step 1250: train loss 3.7835, val loss 4.7182


        ##residual is useful for deepnet
        step 1000: train loss 3.8928, val loss 4.6820

        step 1000: train loss 3.9271, val loss 4.7436

    

        step 1250: train loss 3.8331, val loss 4.7730

        step 1250: train loss 3.8228, val loss 4.7241

        step 1250: train loss 3.8218, val loss 4.7291


        ###
        step 1250: train loss 3.8247, val loss 4.7828

        ###
        step 1000: train loss 3.9573, val loss 4.7106
        ###residual
        step 1000: train loss 3.8489, val loss 4.6791

        ###
        step 1250: train loss 3.8341, val loss 4.7563



        '''

        def get_context(block, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            ### maybe adding channel to gate the context?
            ### getting nh neighbors, and maybe save to the memory?

            ### output is (b,t,nh*e)
            ### by computing (b,t,nh,t) attention from (b,t,nh,e) and (b,t,e)

            ### gating the attention with a linear head or a kernel

            ### hypothesis is that the model should consult internal knowledge 
            ### during its deduction of next word.
            ###  internal knowledge is reflected as the 

            Adding knowledge into the model meaning to bias the model dynamics
            

            '''
            xk = block.k_attn(xq).reshape((b,t,nh,e))
            # xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            # gate = torch.einsum('btke,bte->btk',block.k_gate(xq).reshape((b,t,nh,e)),xq/ne**0.5).sigmoid()
            # att  = torch.einsum('bktp,btk->bktp',att,gate)

            y    = torch.einsum('bktp,bpe->btke',att,xn)
            y    = y.reshape((b,t,-1))
            y    = torch.cat([xn,y],dim=-1).reshape((b,t,-1,e))

            y    = y.reshape((b,t,-1))

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        dx1 = xn
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            # lreg += block0.k_attn.weight.square().sum()
            lreg += regnorm( block0.k_attn.weight )
            for i, fblock in enumerate(( self.transformer.h)):
                y1 = get_context(block0, dx1, xn)
                # dx1 = fblock.flayer(y1)
                # dx1 = 0.5 *(dx1 + fblock.flayer(y1))
                dx1 = 0.5 *(dx1 + block0.flayer(y1))
                # lreg += fblock.flayer.weight.square().sum()
                lreg += regnorm( fblock.flayer.weight )

                # dx1 = dx1 + torch.normal(0.,0.02,dx1.shape,device=device)
                # dx1 = dx1 + torch.normal(0., 0.05,dx1.shape,device=device)
                # dx1 = 1.0 *(fblock.flayer(y1))
                # dx1 = 1.0 *(block0.flayer(y1)) 
                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            

        x = dx1 

        x0 = x

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn


        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                # loss = loss + 0.1*lreg
                loss = loss + lreg_factor*lreg

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        


class CCM35(GPT):
    '''
    using gaussian displacement for updates

    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        # config.optimizer ='adamw'
        config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 4
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292


        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                v_attn = nn.Linear(ne, ne,bias=False),
                k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        # lreg_factor = 0.01
        # lreg_factor = 0.001
        # lreg_factor = 0.00001
        def regnorm(weight):
            return 0.

        lreg += regnorm(self.transformer.wte.weight)
        lreg += regnorm(self.transformer.wpe.weight)
        lreg += regnorm(self.transformer.wpe2.weight)
        # lreg += self.transformer.wte.weight.square().sum()
        # lreg += self.transformer.wpe.weight.square().sum()
        # lreg += self.transformer.wpe2.weight.square().sum()


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = x
        # xn = x/(0.001 + x.abs().max(dim=-1,keepdims=True)[0])
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head
        


        '''
        step 1250: train loss 3.7433, val loss 4.7538  
        step 1250: train loss 3.8416, val loss 4.7337

        '''


        def wnorm(w,dim):
            xw = w / (0.001 + w.std(dim,keepdims=True))
            return xw
            # .square().sum(()))
        def get_context(block, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            ### maybe adding channel to gate the context?
            ### getting nh neighbors, and maybe save to the memory?

            ### output is (b,t,nh*e)
            ### by computing (b,t,nh,t) attention from (b,t,nh,e) and (b,t,e)

            ### gating the attention with a linear head or a kernel

            ### hypothesis is that the model should consult internal knowledge 
            ### during its deduction of next word.
            ###  internal knowledge is reflected as the 

            Adding knowledge into the model meaning to bias the model dynamics
            

            '''
            # xk = xq.matmul(block.k_attn.weight.T)
            # xk = xq.matmul(wnorm(block.k_attn.weight.T,(0,1)))
            xk = xq.matmul(wnorm(block.k_attn.weight.T,(1,))/(ne**0.5))
            # xk = xq.matmul(wnorm(block.k_attn.weight.T,(1,)))
            
            # xk = block.k_attn(xq)
            xk = xk.reshape((b,t,nh,e))
            # xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            # gate = torch.einsum('btke,bte->btk',block.k_gate(xq).reshape((b,t,nh,e)),xq/ne**0.5).sigmoid()
            # att  = torch.einsum('bktp,btk->bktp',att,gate)

            y    = torch.einsum('bktp,bpe->btke',att,xn)
            y    = y.reshape((b,t,-1))
            y    = torch.cat([xn,y],dim=-1).reshape((b,t,-1,e))

            y    = y.reshape((b,t,-1))

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        dx1 = xn
        for ii in range(1):
            for i, fblock in enumerate(( self.transformer.h)):
                y1 = get_context(block0, dx1, xn)
                # dx1 = 0.5 *(dx1 + y1@ wnorm(fblock.flayer.weight.T,(0,1)))
                # dx1 = 0.5 *(dx1 + y1@ (nh*ne* wnorm(fblock.flayer.weight.T,(0,1))))
                # dx1 = 0.5 *(dx1 + y1@ (nh* wnorm(fblock.flayer.weight.T,(0,1))))
                dx1 = 0.5 *(dx1 + y1@ (wnorm(fblock.flayer.weight.T,(1,)  )/(ne**0.5)))
                # dx1 = 0.5 *(dx1 + y1@ (wnorm(fblock.flayer.weight.T,(1,))))
                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            

        x = dx1 

        x0 = x

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn


        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # logits = self.lm_head(x)
            # logits = x.matmul(wnorm(self.lm_head.weight.T,(0,)))
            logits = x.matmul(wnorm(self.lm_head.weight.T,(1,)))/(ne**0.5)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                # loss = loss + lreg
                # loss = loss + 0.1*lreg
                # loss = loss + 0.9*lreg
                # loss = loss + 0.02*lreg
                # loss = loss + 0.001*lreg
                # loss = loss + 0.0001*lreg
                # loss = loss + 0.01*lreg
                # loss = loss + 0.05*lreg
                # loss = loss + 0.1*lreg
                loss = loss + lreg_factor*lreg

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        




class CCM36(GPT):
    '''
    using a matrix to select the context attention
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 4
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292


        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            alayer1  = nn.Linear( ne, ne ,bias=False),
            alayer2  = nn.Linear( ne, ne ,bias=False),
            alayer3  = nn.Linear( ne, ne ,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne, ne*(nh+1),bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                v_attn = nn.Linear(ne, ne,bias=False),
                k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        # lreg_factor = 0.01
        # lreg_factor = 0.001
        # lreg_factor = 0.00001
        def regnorm(weight):
            # return (weight.square().mean() - 1.).abs()*weight.numel()
            # return (weight.square().sum() - weight.numel()).abs()
            # return (weight.square().mean() - 1).abs()
            # return (weight.abs().sum())
            return 0
            # square().mean() - 1).abs()
            # weight.numel()).abs()

        lreg += regnorm(self.transformer.wte.weight)
        lreg += regnorm(self.transformer.wpe.weight)
        lreg += regnorm(self.transformer.wpe2.weight)
        # lreg += self.transformer.wte.weight.square().sum()
        # lreg += self.transformer.wpe.weight.square().sum()
        # lreg += self.transformer.wpe2.weight.square().sum()


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(block, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            ### maybe adding channel to gate the context?
            ### getting nh neighbors, and maybe save to the memory?

            ### output is (b,t,nh*e)
            ### by computing (b,t,nh,t) attention from (b,t,nh,e) and (b,t,e)

            ### gating the attention with a linear head or a kernel

            ### hypothesis is that the model should consult internal knowledge 
            ### during its deduction of next word.
            ###  internal knowledge is reflected as the 

            Adding knowledge into the model meaning to bias the model dynamics
            

            '''
            xk = block.k_attn(xq).reshape((b,t,nh,e))
            # xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            # gate = torch.einsum('btke,bte->btk',block.k_gate(xq).reshape((b,t,nh,e)),xq/ne**0.5).sigmoid()
            # att  = torch.einsum('bktp,btk->bktp',att,gate)

            y    = torch.einsum('bktp,bpe->btke',att,xn)
            y    = y.reshape((b,t,-1))
            y    = torch.cat([xn,y],dim=-1).reshape((b,t,-1,e))

            y    = y.reshape((b,t,-1))

            return y
        '''
        step 1250: train loss 3.9792, val loss 4.8399
        step 1250: train loss 3.9175, val loss 4.8342
        step 1500: train loss 3.7732, val loss 4.8229

        step 1250: train loss 3.8601, val loss 4.8149

        step 1250: train loss 3.8116, val loss 4.8056

        '''

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            for i, fblock in enumerate(( self.transformer.h)):
                y1 = get_context(block0, dx1, xn)
                # block1.k_attn(dx1)

                xk  = md.gate_attn(dx1).reshape((b,t,(nh+1),e))[:,:,:,0]
                att = xk
                att = att.softmax(-1)
                y1 = torch.einsum('btk,btke->btke',att,y1.reshape((b,t,-1,e))).reshape((b,t,-1))
                dx1 = 0.5 *(dx1 + block0.flayer(y1))
                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            

        x = dx1 

        x0 = x

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn


        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:

                loss = loss + lreg_factor*lreg

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        


class CCM37(GPT):
    '''
    using a matrix to select the context attention
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne, ng,bias=True),
            k_attn = nn.Linear(ne, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                v_attn = nn.Linear(ne, ne,bias=False),
                k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                # flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                flayer = nn.Linear(ne*(nh+2),    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            # xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)

            y    = torch.einsum('bktp,bpe->btke',att,xn)
            # y    = y.reshape((b,t,-1))

            # y    = torch.cat([xn,y],dim=-1).reshape((b,t,-1,e))
            # y    = y.reshape((b,t,-1))

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        dx3 = xn
        # nh  = self.config.n_head
        # dx2  = torch.ones((b,t,e),device=device)
        dx1k = torch.zeros((b,t,e,ng),device=device)
        dx2k = torch.zeros((b,t,e,ng),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            for i, fblock in enumerate(( self.transformer.h)):
                
                '''
                step 1250: train loss 3.8440, val loss 4.7829

                '''
                ### action1 is to squash c,ac,m together to get new c
                xk    = md.k_attn(dx1).reshape((b,t,nh))
                attk  = xk.softmax(-1)
                # wk    = block0.k_attn.weight.T.reshape((ne,ne,nh))
                # wk    = torch.einsum("btk,efk->btef", attk, wk)
                # y1    = get_context(wk, dx1, xn)
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, dx1, xn)
                ### btke


                xk  = md.gate_attn(dx1).reshape((b,t,ng))
                att = xk
                att = att.softmax(-1)

                ### the k possibilities of the 

                ### first possibility is to update dx1
                ### second possibility to update memory

                # ctx = torch.cat([attk.unsqueeze(-1)*wk, dx1,dx2,y1,],dim=-1)
                ctx = torch.cat([attk.unsqueeze(-1)*y1, 
                dx1[:,:,None], 
                dx2[:,:,None],
                # dx3[:,:,None],
                ],dim=2).reshape((b,t,-1))
                # dx1k[:,:,:,0] =  0.5 *(dx1 + block0.flayer(ctx)) 
                # dx1k[:,:,:,0] =  block0.flayer(ctx)
                # dx1k[:,:,:,1] =  dx1

                dx1k = torch.stack([
                    block0.flayer(ctx),
                    dx1, 
                    # dx1, 
                ],dim=-1)


                dx2k = torch.stack([
                    dx2,
                    torch.einsum('btk,btke->bte',attk,y1,),
                    # dx2,
                ],dim=-1)

                # dx3k = torch.stack([
                #     dx3,
                #     dx3,
                #     torch.einsum('btk,btke->bte',attk,y1,),
                # ],dim=-1)

                # 0.5 *(dx1 + block0.flayer(torch.cat([dx1,dx2,y1,],dim=-1))) 
                # dx2k

                # y1 = torch.einsum('btk,btke->btke',att,y1.reshape((b,t,-1,e))).reshape((b,t,-1))
                # dx1 = 0.5 *(dx1 + block0.flayer(y1)) 
                
                dx1 = torch.einsum('btk,btek->bte', att,dx1k)
                dx2 = torch.einsum('btk,btek->bte', att,dx2k)
                # dx3 = torch.einsum('btk,btek->bte', att,dx3k)

                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
                dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            
                # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            

        x = dx1 

        x0 = x

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn


        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        

class CCM38(GPT):
    '''
    using a matrix to select the context attention
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne, ng,bias=True),
            k_attn = nn.Linear(ne, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                # v_attn = nn.Linear(ne, ne,bias=False),
                # k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                # flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                flayer = nn.Linear(ne*(nh+2),    ne,bias=False),
                flayer2 = nn.Linear(ne*(nh+2),    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            # xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)

            y    = torch.einsum('bktp,bpe->btke',att,xn)

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        dx3 = xn
        # nh  = self.config.n_head
        # dx2  = torch.ones((b,t,e),device=device)
        dx1k = torch.zeros((b,t,e,ng),device=device)
        dx2k = torch.zeros((b,t,e,ng),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            for i, fblock in enumerate(( self.transformer.h)):
                
                '''
                step 1250: train loss 3.8440, val loss 4.7829

                '''
                ### action1 is to squash c,ac,m together to get new c
                xk    = md.k_attn(dx1).reshape((b,t,nh))
                attk  = xk.softmax(-1)
                # wk    = block0.k_attn.weight.T.reshape((ne,ne,nh))
                # wk    = torch.einsum("btk,efk->btef", attk, wk)
                # y1    = get_context(wk, dx1, xn)
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, dx1, xn)
                ### btke


                xk  = md.gate_attn(dx1).reshape((b,t,ng))
                att = xk
                att = att.softmax(-1)

                ### the k possibilities of the 

                ### first possibility is to update dx1
                ### second possibility to update memory

                # ctx = torch.cat([attk.unsqueeze(-1)*wk, dx1,dx2,y1,],dim=-1)
                ctx = torch.cat([attk.unsqueeze(-1)*y1, 
                dx1[:,:,None], 
                dx2[:,:,None],
                # dx[:,:,None],
                # dx3[:,:,None],
                ],dim=2).reshape((b,t,-1))

                dx1k = torch.stack([
                    block0.flayer(ctx),
                    dx1, 
                    # dx1, 
                ],dim=-1)


                dx2k = torch.stack([
                    dx2,
                    torch.einsum('btk,btke->bte',attk,y1,),
                    # dx2,
                ],dim=-1)

                dx3k  = torch.stack([
                    block0.flayer2(ctx),
                    dx3,
                    # dx2,
                ],dim=-1)



                
                dx1 = torch.einsum('btk,btek->bte', att,dx1k)
                dx2 = torch.einsum('btk,btek->bte', att,dx2k)
                dx3 = torch.einsum('btk,btek->bte', att,dx3k)
                # dx3 = torch.einsum('btk,btek->bte', att,dx3k)

                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
                dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            
                dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            

        # x = dx1 
        x = dx3 

        x0 = x

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn


        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        



class CCM39(GPT):
    '''
    using a matrix to select the context attention
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne, ng,bias=True),
            k_attn = nn.Linear(ne, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                # v_attn = nn.Linear(ne, ne,bias=False),
                # k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                # flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                flayer = nn.Linear(ne*(nh+2),    ne,bias=False),
                flayer2 = nn.Linear(ne*(nh+2),    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            # xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)

            y    = torch.einsum('bktp,bpe->btke',att,xn)

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        dx3 = xn
        # nh  = self.config.n_head
        # dx2  = torch.ones((b,t,e),device=device)
        dx1k = torch.zeros((b,t,e,ng),device=device)
        dx2k = torch.zeros((b,t,e,ng),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            for i, fblock in enumerate(( self.transformer.h)):
                
                '''
                step 1250: train loss 3.8440, val loss 4.7829

                '''
                ### action1 is to squash c,ac,m together to get new c
                xk    = md.k_attn(dx1).reshape((b,t,nh))
                attk  = xk.softmax(-1)
                # wk    = block0.k_attn.weight.T.reshape((ne,ne,nh))
                # wk    = torch.einsum("btk,efk->btef", attk, wk)
                # y1    = get_context(wk, dx1, xn)
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, dx1, xn)
                ### btke

                xk  = md.gate_attn(dx1).reshape((b,t,ng))
                att = xk
                att = att.softmax(-1)

                ### the k possibilities of the 

                ### first possibility is to update dx1
                ### second possibility to update memory

                # ctx = torch.cat([attk.unsqueeze(-1)*wk, dx1,dx2,y1,],dim=-1)
                ctx = torch.cat([attk.unsqueeze(-1)*y1, 
                dx1[:,:,None], 
                dx1[:,:,None], 
                # dx2[:,:,None],
                # dx[:,:,None],
                # dx3[:,:,None],
                ],dim=2).reshape((b,t,-1))

                dx1k = torch.stack([
                    block0.flayer(ctx),
                    dx1, 
                    # dx1, 
                ],dim=-1)




                dx3k  = torch.stack([
                    block0.flayer2(ctx),
                    dx3,
                    # dx2,
                ],dim=-1)



                
                dx1 = torch.einsum('btk,btek->bte', att,dx1k)
                # dx2 = torch.einsum('btk,btek->bte', att,dx2k)
                dx3 = torch.einsum('btk,btek->bte', att,dx3k)
                # dx3 = torch.einsum('btk,btek->bte', att,dx3k)

                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
                # dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            
                dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            

        # x = dx1 
        x = dx3 

        x0 = x

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn


        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        


class CCM40(GPT):
    '''
    using a matrix to select the context attention
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne, ng,bias=True),
            k_attn = nn.Linear(ne, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                flayer = nn.Linear(ne*(nh+2),    ne,bias=False),
                flayer2 = nn.Linear(ne*(nh+2),    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            # xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)

            y    = torch.einsum('bktp,bpe->btke',att,xn)

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        dx3 = xn
        dx4 = xn
        # nh  = self.config.n_head
        # dx2  = torch.ones((b,t,e),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            for i, fblock in enumerate(( self.transformer.h)):
                
                '''
                step 1250: train loss 3.8440, val loss 4.7829

                '''
                ### action1 is to squash c,ac,m together to get new c
                xk    = md.k_attn(dx1).reshape((b,t,nh))
                attk  = xk.softmax(-1)
                # attk  = (xk).sigmoid()
                
                ### (maybe this gate is more suitable for sigmoid?)
                # wk    = block0.k_attn.weight.T.reshape((ne,ne,nh))
                # wk    = torch.einsum("btk,efk->btef", attk, wk)
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, dx1, xn)
                ### btke


                xk  = md.gate_attn(dx1).reshape((b,t,ng))
                att = xk
                att = att.softmax(-1)

                ### the k possibilities of the 
                ### first possibility is to update dx1
                ### second possibility to update memory

                # ctx = torch.cat([attk.unsqueeze(-1)*wk, dx1,dx2,y1,],dim=-1)
                ctx = torch.cat([
                    attk.unsqueeze(-1)*y1, 
                    dx1[:,:,None], 
                    dx2[:,:,None],
                    # dx2[:,:,None],
                    # dx4[:,:,None],
                # dx[:,:,None],
                # dx3[:,:,None],
                ],dim=2).reshape((b,t,-1))

                # dx1k = torch.stack([
                #     block0.flayer(ctx),
                #     block0.flayer(ctx),
                #     block0.flayer(ctx),
                # ],dim=-1)


                # dx2k = torch.stack([
                #     torch.einsum('btk,btke->bte',attk,y1,),
                #     dx2,
                # ],dim=-1)

                # dx4k = torch.stack([
                #     dx4,
                #     torch.einsum('btk,btke->bte',attk,y1,),
                # ],dim=-1)

                # dx3k  = torch.stack([
                #     block0.flayer2(ctx),
                #     dx3,
                #     dx3,
                # ],dim=-1)

                xp = att[:,:,0:1,]
                dx1 = (1-xp)*dx1 + xp * block0.flayer(ctx) 
                
                # xp = attk[:,:,1:2,]
                xp = 1 - att[:,:,0:1,]
                dx2 = (1-xp)*dx2 + xp * torch.einsum('btk,btke->bte',attk,y1,)
                
                # xp = attk[:,:,2:3,]
                xp = att[:,:,0:1,]
                dx3 = (1-xp)*dx3 + xp * block0.flayer2(ctx)
                
                # dx4 = dx2
                # xp = attk[:,:,3:4,]
                # dx4 = (1-xp)*dx4 + xp * torch.einsum('btk,btke->bte',attk,y1,)

                # dx4 = dx2
                # dx1 = torch.einsum('btk,btek->bte', att,dx1k)
                # dx2 = torch.einsum('btk,btek->bte', att,dx2k)
                # dx3 = torch.einsum('btk,btek->bte', att,dx3k)
                # dx4 = torch.einsum('btk,btek->bte', att,dx4k)
                # dx3 = torch.einsum('btk,btek->bte', att,dx3k)

                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
                dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            
                dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                # dx4 = dx4/(0.001 + dx4.std(-1,keepdims=True))            
                # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            

        # x = dx1 
        x = dx3 

        x0 = x

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn


        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        


class CCM41(GPT):
    '''
    using a matrix to select the context attention
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 4
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne, ng,bias=True),
            gate_attn_2 = nn.Linear(ne, ng,bias=True),
            k_attn = nn.Linear(ne, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                flayer = nn.Linear(ne*(nh+3),    ne,bias=False),
                flayer2 = nn.Linear(ne*(nh+3),    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            # xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)

            y    = torch.einsum('bktp,bpe->btke',att,xn)

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        dx3 = xn
        dx4 = xn
        # nh  = self.config.n_head
        # dx2  = torch.ones((b,t,e),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            for i, fblock in enumerate(( self.transformer.h)):
                
                '''
                step 1250: train loss 3.8440, val loss 4.7829

                '''
                ### action1 is to squash c,ac,m together to get new c
                xk    = md.k_attn(dx1).reshape((b,t,nh))
                attk  = xk.softmax(-1)
                # attk  = (xk).sigmoid()
                
                ### (maybe this gate is more suitable for sigmoid?)
                # wk    = block0.k_attn.weight.T.reshape((ne,ne,nh))
                # wk    = torch.einsum("btk,efk->btef", attk, wk)
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, dx1, xn)
                ### btke


                xk  = md.gate_attn(dx1).reshape((b,t,ng))
                att = xk
                # att = att.softmax(-1)
                att = att.sigmoid()

                xk  = md.gate_attn_2(dx1).reshape((b,t,ng))
                # att2 = xk
                # att = att.softmax(-1)
                att2 = xk.sigmoid()

                # softmax(-1)

                ### the k possibilities of the 
                ### first possibility is to update dx1
                ### second possibility to update memory

                # ctx = torch.cat([attk.unsqueeze(-1)*wk, dx1,dx2,y1,],dim=-1)
                ctx = torch.cat([
                    attk.unsqueeze(-1)*y1, 
                    dx1[:,:,None] * att2[:,:,0:1,None], 
                    dx2[:,:,None] * att2[:,:,1:2,None],
                    # dx2[:,:,None],

                    dx4[:,:,None] * att2[:,:,2:3,None],

                # dx[:,:,None],
                # dx3[:,:,None],
                ],dim=2).reshape((b,t,-1))

                # dx1k = torch.stack([
                #     block0.flayer(ctx),
                #     block0.flayer(ctx),
                #     block0.flayer(ctx),
                # ],dim=-1)

                xp = att[:,:,0:1,]
                dx1 = (1-xp)*dx1 + xp * block0.flayer(ctx) 
                
                xp = att[:,:,1:2,]
                # xp = 1 - att[:,:,0:1,]
                dx2 = (1-xp)*dx2 + xp * torch.einsum('btk,btke->bte',attk,y1,)

                
                xp = att[:,:,2:3,]
                # xp = att[:,:,0:1,]
                dx3 = (1-xp)*dx3 + xp * block0.flayer2(ctx)
                
                # # dx4 = dx2
                xp = att[:,:,3:4,]
                dx4 = (1-xp)*dx4 + xp * torch.einsum('btk,btke->bte',attk,y1,)


                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
                dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            
                dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                dx4 = dx4/(0.001 + dx4.std(-1,keepdims=True))            
                # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            

        # x = dx1 
        x = dx3 

        x0 = x

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn


        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        


class CCM42(GPT):
    '''
    using a matrix to select the context attention
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne, ng,bias=True),
            k_attn = nn.Linear(ne, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                # v_attn = nn.Linear(ne, ne,bias=False),
                # k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                # flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                flayer = nn.Linear(ne*(nh+3),    ne,bias=False),
                flayer2 = nn.Linear(ne*(nh+3),    ne,bias=False),
                flayer3 = nn.Linear(ne*(nh+3),    ne,bias=False),
                flayer4 = nn.Linear(ne*(nh+3),    ne,bias=False),
                )
                ) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            # xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)

            y    = torch.einsum('bktp,bpe->btke',att,xn)

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        # dx1 = xn 
        dx1 = torch.ones((b,t,e),device=device)
        dx1 = dx1 / (0.001 + dx1.std(-1,keepdim=True))
        dx2 = xn
        dx3 = xn
        dx4 = xn
        # nh  = self.config.n_head
        # dx2  = torch.ones((b,t,e),device=device)
        dx1k = torch.zeros((b,t,e,ng),device=device)
        dx2k = torch.zeros((b,t,e,ng),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            for i, fblock in enumerate(( self.transformer.h)):
                
                '''
                step 1250: train loss 3.8440, val loss 4.7829

                '''
                ### action1 is to squash c,ac,m together to get new c
                xk    = md.k_attn(dx1).reshape((b,t,nh))
                attk  = xk.softmax(-1)
                # wk    = block0.k_attn.weight.T.reshape((ne,ne,nh))
                # wk    = torch.einsum("btk,efk->btef", attk, wk)
                # y1    = get_context(wk, dx1, xn)
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, dx1, xn)
                ### btke

                xk  = md.gate_attn(dx1).reshape((b,t,ng))
                att = xk
                att = att.softmax(-1)

                ### the k possibilities of the 

                ### first possibility is to update dx1
                ### second possibility to update memory

                # ctx = torch.cat([attk.unsqueeze(-1)*wk, dx1,dx2,y1,],dim=-1)
                ctx = torch.cat([attk.unsqueeze(-1)*y1, 
                dx1[:,:,None], 
                dx1[:,:,None], 
                dx3[:,:,None], 
                # dx1[:,:,None], 
                # dx2[:,:,None],
                # dx4[:,:,None],
                # dx[:,:,None],
                # dx3[:,:,None],
                ],dim=2).reshape((b,t,-1))

                # dx1k = torch.stack([
                #     block0.flayer(ctx),
                #     dx1, 
                #     # dx1, 
                # ],dim=-1)

                # dx3k  = torch.stack([
                #     block0.flayer2(ctx),
                #     dx3,
                #     # dx2,
                # ],dim=-1)

                dx1 = block0.flayer(ctx)
                # dx2 = block0.flayer2(ctx)
                dx3 = block0.flayer3(ctx)
                # dx4 = block0.flayer4(ctx)


                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
                # dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            
                dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                # dx4 = dx4/(0.001 + dx3.std(-1,keepdims=True))            
                # # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                dx3 = dx1

        # x = dx1 
        x = dx3 

        x0 = x

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn


        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        


class CCM43(GPT):
    '''
    CCM38 with a output aggregation gate
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne, ng,bias=True),
            k_attn = nn.Linear(ne, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                # v_attn = nn.Linear(ne, ne,bias=False),
                # k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                # flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                flayer = nn.Linear(ne*(nh+2),    ne,bias=False),
                flayer2 = nn.Linear(ne*(nh+2),    ne,bias=False),
                o_gate =  nn.Linear(ne*(nh+2), 1,bias=True),
                )
                # ) for _ in range(config.n_layer)
                ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            # xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)

            y    = torch.einsum('bktp,bpe->btke',att,xn)

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        dx3 = xn
        # nh  = self.config.n_head
        # dx2  = torch.ones((b,t,e),device=device)
        dx1k = torch.zeros((b,t,e,ng),device=device)
        dx2k = torch.zeros((b,t,e,ng),device=device)

        dx3h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            # for layer_i, fblock in enumerate(( self.transformer.h)):
            for layer_i in range(nl):
                # ( self.transformer.h)):
                
                '''
                step 1250: train loss 3.8440, val loss 4.7829

                '''
                ### action1 is to squash c,ac,m together to get new c
                xk    = md.k_attn(dx1).reshape((b,t,nh))
                attk  = xk.softmax(-1)
                # wk    = block0.k_attn.weight.T.reshape((ne,ne,nh))
                # wk    = torch.einsum("btk,efk->btef", attk, wk)
                # y1    = get_context(wk, dx1, xn)
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, dx1, xn)
                ### btke


                xk  = md.gate_attn(dx1).reshape((b,t,ng))
                att = xk
                att = att.softmax(-1)

                ### the k possibilities of the 

                ### first possibility is to update dx1
                ### second possibility to update memory

                # ctx = torch.cat([attk.unsqueeze(-1)*wk, dx1,dx2,y1,],dim=-1)
                ctx = torch.cat([attk.unsqueeze(-1)*y1, 
                dx1[:,:,None], 
                dx2[:,:,None],
                # dx[:,:,None],
                # dx3[:,:,None],
                ],dim=2).reshape((b,t,-1))

                dx1k = torch.stack([
                    block0.flayer(ctx),
                    dx1, ],
                    dim=-1,
                )


                dx2k = torch.stack([
                    dx2,
                    torch.einsum('btk,btke->bte',attk,y1,),
                    # dx2,
                ],dim=-1)

                dx3k  = torch.stack([
                    block0.flayer2(ctx),
                    dx3,
                    # dx2,
                ],dim=-1)

                out_gate = block0.o_gate(ctx)
                            
                dx1 = torch.einsum('btk,btek->bte', att,dx1k)
                dx2 = torch.einsum('btk,btek->bte', att,dx2k)
                dx3 = torch.einsum('btk,btek->bte', att,dx3k)
                # dx3 = torch.einsum('btk,btek->bte', att,dx3k)

                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
                dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            
                dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                dx3h[:,:,:,layer_i]   = dx3
                outh[:,:,:,layer_i]   = out_gate

        # x = dx1 
        # x = dx3 
        ### NH1
        ###          E1000  E1250
        ### w   gate  4.74  4.7358
        ### w/o gate  4.74  4.7359
        ### sigmoid gate 4.79, 4.78
        ### maximum gate 4.91, 4.87 

        # x = dx3
        # x = torch.einsum('btel,btl->bte',dx3h, outh.sigmoid().squeeze(2))
        # _, idx = outh.max(dim=-1)
        # x = torch.gather(dx3h, index=idx.unsqueeze(-1).expand((b,t,e,1)),dim=-1).squeeze(-1)

        x = torch.einsum('btel,btl->bte',dx3h, outh.softmax(-1).squeeze(2))

        x0 = x

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        ddx = (xn - x0)
        x = xn


        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss        

class CCM44(GPT):
    '''
    CCM38 with a output aggregation gate
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne, ng,bias=True),
            k_attn = nn.Linear(ne, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne*nh,bias=False),
                # v_attn = nn.Linear(ne, ne,bias=False),
                # k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                # flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(nh+1),    ne,bias=False),
                o_gate =  nn.Linear(ne*(nh+1), 1,bias=True),
                )
                # ) for _ in range(config.n_layer)
                ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            # xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)

            y    = torch.einsum('bktp,bpe->btke',att,xn)

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        dx3 = xn
        # nh  = self.config.n_head
        # dx2  = torch.ones((b,t,e),device=device)
        dx1k = torch.zeros((b,t,e,ng),device=device)
        dx2k = torch.zeros((b,t,e,ng),device=device)

        dx3h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            # for layer_i, fblock in enumerate(( self.transformer.h)):
            for layer_i in range(nl):
                # ( self.transformer.h)):
                
                '''
                step 1250: train loss 3.8440, val loss 4.7829

                '''
                ### action1 is to squash c,ac,m together to get new c
                xk    = md.k_attn(dx1).reshape((b,t,nh))
                attk  = xk.softmax(-1)
                # wk    = block0.k_attn.weight.T.reshape((ne,ne,nh))
                # wk    = torch.einsum("btk,efk->btef", attk, wk)
                # y1    = get_context(wk, dx1, xn)
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, dx1, xn)
                ### btke

                ## possible actions

                xk  = md.gate_attn(dx1).reshape((b,t,ng))
                att = xk
                # att = att.softmax(-1)
                att = att.sigmoid()

                ### the k possibilities of the 

                ### first possibility is to update dx1
                ### second possibility to update memory

                # ctx = torch.cat([attk.unsqueeze(-1)*wk, dx1,dx2,y1,],dim=-1)
                ym = attk.unsqueeze(-1)*y1
                ctx = torch.cat([
                    ym,
                    dx1[:,:,None], 
                ],dim=2).reshape((b,t,-1))




                out_gate = block0.o_gate(ctx)

                xp = att[:,:,0:1]
                dx1 = xp * block0.flayer(ctx) + (1-xp)*dx1
                dx3 = xp * block0.flayer2(ctx) + (1-xp)*dx3
                            
                # dx3 = torch.einsum('btk,btek->bte', att,dx3k)

                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
                # dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            
                dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                dx3h[:,:,:,layer_i]   = dx3
                outh[:,:,:,layer_i]   = out_gate

        # x = dx1 
        # x = dx3 
        ### NH1
        ###          E1000  E1250
        ### w   gate  4.74  4.7358
        ### w/o gate  4.74  4.7359
        ### sigmoid gate 4.79, 4.78
        ### maximum gate 4.91, 4.87 

        x = dx3
        # x = torch.einsum('btel,btl->bte',dx3h, outh.sigmoid().squeeze(2))
        # _, idx = outh.max(dim=-1)
        # x = torch.gather(dx3h, index=idx.unsqueeze(-1).expand((b,t,e,1)),dim=-1).squeeze(-1)

        # x = torch.einsum('btel,btl->bte',dx3h, outh.softmax(-1).squeeze(2))

        x0 = x


        # xn = x
        # xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # ddx = (xn - x0)
        # x = xn


        if 1:
            lp = outh.log_softmax(-1)[:,pad:,0]

            x = self.transformer.ln_f(dx3h.transpose(-1,-2))        
            


            x = x[:,pad:]

            if targets is not None:
                # if we are given some desired targets also calculate the loss
                logits = self.lm_head(x)

                ### (b,t,l)
                loss = F.cross_entropy(logits.transpose(1,-1), targets.unsqueeze(-1).expand((b,t-pad,nl)).transpose(1,-1), ignore_index=-1,reduction='none')
                # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,reduction='mean')
                loss = loss.transpose(1,-1)
                loss = (loss + lp).logsumexp(-1).mean((0,1))

                if self.training:
                    loss = loss + lreg_factor*lreg
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
                loss = None


        # x = self.transformer.ln_f(x)        
        
        # # lreg += self.lm_head.weight.square().sum()
        # lreg += regnorm(  self.lm_head.weight)
        # # fblock.flayer.weight )

        # x = x[:,pad:]

        # if targets is not None:
        #     # if we are given some desired targets also calculate the loss
        #     logits = self.lm_head(x)
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        #     if self.training:
        #         loss = loss + lreg_factor*lreg
        # else:
        #     # inference-time mini-optimization: only forward the lm_head on the very last position
        #     logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        #     loss = None

        return logits, loss              


class CCM45(GPT):
    '''
    CCM38 with a output aggregation gate
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne*3, ng,bias=True),
            k_attn = nn.Linear(ne*2, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne*2, ne*nh,bias=False),
                # v_attn = nn.Linear(ne, ne,bias=False),
                # k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                # flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                flayer = nn.Linear(ne*(2+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(2+1),    ne,bias=False),
                flayer3 = nn.Linear(ne*(2+1),    ne,bias=False),
                o_gate =  nn.Linear(ne*(2+1), 1,bias=True),
                )
                # ) for _ in range(config.n_layer)
                ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)

            y    = torch.einsum('bktp,bpe->btke',att,xn)

            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn*0+1
        dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        dx3 = xn
        # nh  = self.config.n_head
        # dx2  = torch.ones((b,t,e),device=device)
        dx1k = torch.zeros((b,t,e,ng),device=device)
        dx2k = torch.zeros((b,t,e,ng),device=device)

        dx3h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            # for layer_i, fblock in enumerate(( self.transformer.h)):
            for layer_i in range(nl):
                # ( self.transformer.h)):
                
                '''
                step 1250: train loss 3.8440, val loss 4.7829

                '''

                ctx0 = torch.cat([
                    dx1,dx2,],dim=2).reshape((b,t,-1))                

                ### attention of different head
                xk    = md.k_attn(ctx0).reshape((b,t,nh))
                attk  = xk.softmax(-1)
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, ctx0, xn)

                ym = attk.unsqueeze(-1)*y1
                ctx = torch.cat([ym.squeeze(2),ctx0],dim=-1)
                ### gating the event
                #  
                xk  = md.gate_attn(ctx).reshape((b,t,ng))
                att = xk
                # att = att.softmax(-1)
                att = att.sigmoid()


                out_gate = block0.o_gate(ctx)

                xp  = att[:,:,0:1]
                dx1 = xp * block0.flayer(ctx)  + (1-xp)*dx1
                xp  = 1 - att[:,:,0:1]
                dx2 = xp * block0.flayer2(ctx) + (1-xp)*dx2
                dx3 = block0.flayer3(ctx)
                            
                # dx3 = torch.einsum('btk,btek->bte', att,dx3k)

                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
                dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            
                dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                dx3h[:,:,:,layer_i]   = dx3
                outh[:,:,:,layer_i]   = out_gate

        ### NH1
        ###          E1000  E1250
        ### w   gate  4.74  4.7358
        ### w/o gate  4.74  4.7359
        ### sigmoid gate 4.79, 4.78
        ### maximum gate 4.91, 4.87 

        x = dx3
        # # x = torch.einsum('btel,btl->bte',dx3h, outh.sigmoid().squeeze(2))
        # # _, idx = outh.max(dim=-1)
        # # x = torch.gather(dx3h, index=idx.unsqueeze(-1).expand((b,t,e,1)),dim=-1).squeeze(-1)

        # x = torch.einsum('btel,btl->bte',dx3h, outh.softmax(-1).squeeze(2))

        x0 = x

        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss                      


class CCM46(GPT):
    '''
    CCM38 with a output aggregation gate
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 3
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne*(3+1), ng,bias=True),
            k_attn = nn.Linear(ne*3, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne*3, ne*nh,bias=False),
                # v_attn = nn.Linear(ne, ne,bias=False),
                # k_gate = nn.Linear(ne, ne*nh, bias=False),
                # k_gate = nn.Linear(ne, nh, bias=True),
                # flayer = nn.Linear(ne*(nh+1),    ne,bias=False),
                flayer = nn.Linear(ne*(3+1),    ne,bias=False),
                # flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2b = nn.Linear(ne*(2),    ne,bias=False),
                flayer3 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer4 = nn.Linear(ne*(3+1),    ne,bias=False),
                o_gate =  nn.Linear(ne*(3+1), 1, bias=True),
                )
                # ) for _ in range(config.n_layer)
                ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            y    = torch.einsum('bktp,bpe->btke',att,xn)
            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        # dx2 = xn*0+1
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        dx3 = xn
        dx0 = xn
        # nh  = self.config.n_head
        # dx2  = torch.ones((b,t,e),device=device)
        dx1k = torch.zeros((b,t,e,ng),device=device)
        dx2k = torch.zeros((b,t,e,ng),device=device)

        dx0h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            # for layer_i, fblock in enumerate(( self.transformer.h)):
            for layer_i in range(nl):                
                '''
                step 1250: train loss 3.8440, val loss 4.7829
                '''

                ctx0 = torch.cat([
                    dx1,dx2,dx3],dim=2).reshape((b,t,-1))                
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, ctx0, xn)

                ym = y1.squeeze(2)
                ctx = torch.cat([ym,ctx0],dim=-1)
                ### gating the event
                #  
                xk  = md.gate_attn(ctx).reshape((b,t,ng))
                att = xk
                att = att.softmax(-1)
                # att = att.sigmoid()


                out_gate = block0.o_gate(ctx)

                xp  = att[:,:,0:1]
                dx1 = xp * block0.flayer2(ctx)  + (1-xp)*dx1
                xp  = att[:,:,1:2]
                dx2 = xp * block0.flayer3(ctx)  + (1-xp)*dx2
                xp  = att[:,:,2:3]
                dx3 = xp * block0.flayer4(ctx)  + (1-xp)*dx3
                dx0 = block0.flayer(ctx)


                # dx3 = torch.einsum('btk,btek->bte', att,dx3k)

                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
                dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            
                dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                dx0 = dx0/(0.001 + dx0.std(-1,keepdims=True))            
                # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                dx0h[:,:,:,layer_i]   = dx0
                outh[:,:,:,layer_i]   = out_gate

        ### NH1
        ###          E1000  E1250
        ### w   gate  4.74  4.7358
        ### w/o gate  4.74  4.7359
        ### sigmoid gate 4.79, 4.78
        ### maximum gate 4.91, 4.87 

        # x = dx0
        x = torch.einsum('btel,btl->bte',dx0h, outh.softmax(-1).squeeze(2))

        x0 = x

        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss                      


class CCM47(GPT):
    '''
    CCM38 with a output aggregation gate
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 7
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            gate_attn = nn.Linear(ne*(ng+1), ng,bias=True),
            k_attn = nn.Linear(ne*ng, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne*ng, ne*nh,bias=False),
                flayer = nn.Linear(ne*(ng+1),    ne,bias=False),
                # flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(ng+1),    ne,bias=False),
                flayer2b = nn.Linear(ne*(2),    ne,bias=False),
                flayer3 = nn.Linear(ne*(ng+1),    ne,bias=False),
                flayer4 = nn.Linear(ne*(ng+1),    ne,bias=False),
                o_gate =  nn.Linear(ne*(ng+1), 1, bias=True),
                )
                # ) for _ in range(config.n_layer)
                ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            y    = torch.einsum('bktp,bpe->btke',att,xn)
            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        # dx2 = xn*0+1
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        dx3 = dx2
        dx4 = dx2
        dx5 = dx2
        dx6 = dx2
        dx7 = dx2
        dx0 = dx2
        # nh  = self.config.n_head
        # dx2  = torch.ones((b,t,e),device=device)
        dx0h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            # for layer_i, fblock in enumerate(( self.transformer.h)):
            for layer_i in range(nl):                
                '''
                step 1250: train loss 3.8440, val loss 4.7829
                '''

                # ctx0 = torch.cat([
                #     dx1,dx2,dx3],dim=2).reshape((b,t,-1))                
                # ctx0 = torch.cat([
                #     dx1,dx1,dx1],dim=2).reshape((b,t,-1))                
                # ctx0 = torch.cat([
                #     dx1,dx2,dx3,dx4,dx5],dim=2).reshape((b,t,-1))                
                ctx0 = torch.cat([
                    dx1,dx2,dx3,dx4,dx5,dx6,dx7],dim=2)#.reshape((b,t,-1))                
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, ctx0, xn)

                ym = y1.squeeze(2)
                ctx = torch.cat([ym,ctx0],dim=-1)
                ### gating the event
                #  
                xk  = md.gate_attn(ctx).reshape((b,t,ng))
                att = xk
                att = att.softmax(-1)
                # att = att.sigmoid()


                out_gate = block0.o_gate(ctx)

                xp  = att[:,:,0:1]
                # dx1 = xp * block0.flayer3(ctx)  + (1-xp)*dx1
                dx1 = xp * ym  + (1-xp)*dx1
                xp  = att[:,:,1:2]
                dx2 = xp * ym  + (1-xp)*dx2
                xp  = att[:,:,2:3]
                dx3 = xp * ym  + (1-xp)*dx3
                xp  = att[:,:,3:4]
                dx4 = xp * ym  + (1-xp)*dx4
                xp  = att[:,:,4:5]
                dx5 = xp * ym  + (1-xp)*dx5
                xp  = att[:,:,5:6]
                dx6 = xp * ym  + (1-xp)*dx6
                xp  = att[:,:,6:7]
                dx7 = xp * ym  + (1-xp)*dx7
                # xp  = att[:,:,3:2]
                # dx2 = xp * ym  + (1-xp)*dx2
                # dx4 = xp * block0.flayer3
                dx0 = block0.flayer(ctx)


                # dx3 = torch.einsum('btk,btek->bte', att,dx3k)

                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))            
                dx2 = dx2/(0.001 + dx2.std(-1,keepdims=True))            
                dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                dx4 = dx4/(0.001 + dx4.std(-1,keepdims=True))            
                dx5 = dx5/(0.001 + dx5.std(-1,keepdims=True))            
                dx6 = dx6/(0.001 + dx6.std(-1,keepdims=True))            
                dx7 = dx7/(0.001 + dx7.std(-1,keepdims=True))            
                dx0 = dx0/(0.001 + dx0.std(-1,keepdims=True))            
                # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                dx0h[:,:,:,layer_i]   = dx0
                outh[:,:,:,layer_i]   = out_gate

        ### NH1
        ###          E1000  E1250
        ### w   gate  4.74  4.7358
        ### w/o gate  4.74  4.7359
        ### sigmoid gate 4.79, 4.78
        ### maximum gate 4.91, 4.87 

        # x = dx0
        x = torch.einsum('btel,btl->bte',dx0h, outh.softmax(-1).squeeze(2))

        x0 = x

        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss                      


class CCM48(GPT):
    '''
    CCM38 with a output aggregation gate
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        config.optimizer ='adam'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        # config.dropout = 0.0
        # print (config)
        # config.n_head =  1
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 1
        ng = config.ng
        nh = config.n_head

        ### nh=1
        ### step 1500: train loss 3.7240, val loss 4.8227
        ### nh=2
        ### step 1250: train loss 3.7819, val loss 4.7292

        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            final_output = nn.Linear(ne*(nl), ne,bias=False),
            gate_attn = nn.Linear(ne*(ng+1), ng,bias=True),
            k_attn = nn.Linear(ne*ng, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne*ng, ne*nh,bias=False),
                flayer = nn.Linear(ne*(ng+1),    ne,bias=False),
                # flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(ng+1),    ne,bias=False),
                flayer2b = nn.Linear(ne*(2),    ne,bias=False),
                flayer3 = nn.Linear(ne*(ng+1),    ne,bias=False),
                flayer4 = nn.Linear(ne*(ng+1),    ne,bias=False),
                o_gate =  nn.Linear(ne*(ng+1), 1, bias=True),
                )
                # ) for _ in range(config.n_layer)
                ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 3

        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb[:pad]= pos_emb_2[:pad]
        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head



        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,-1,e))
            xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('btke,bpe->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            y    = torch.einsum('bktp,bpe->btke',att,xn)
            return y

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        # dx2 = xn*0+1
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        dx3 = dx2
        dx0 = dx2
        # nh  = self.config.n_head
        dx0h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        for ii in range(1):
            # dx2 = md.x2(xn)
            # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
            # for layer_i, fblock in enumerate(( self.transformer.h)):
            for layer_i in range(nl):                
                '''
                step 1250: train loss 3.8440, val loss 4.7829
                '''
                # ctx0 = torch.cat([
                #     dx1,dx2,dx3],dim=2).reshape((b,t,-1))                
                # ctx0 = torch.cat([
                #     dx1,dx2,dx3,dx4,dx5],dim=2).reshape((b,t,-1))                
                ctx0 = torch.cat([
                    dx1,],dim=2)
                
                wk    = block0.k_attn.weight.T
                y1    = get_context(wk, ctx0, xn)

                ym = y1.squeeze(2)
                ctx = torch.cat([ym,ctx0],dim=-1)
                ### gating the event
                xk  = md.gate_attn(ctx).reshape((b,t,ng))
                att = xk
                att = att.softmax(-1)
                # att = att.sigmoid()


                out_gate = block0.o_gate(ctx)

                xp  = att[:,:,0:1]
                # dx1 = block0.flayer3(ctx) 
                dx1 = xp * block0.flayer3(ctx) + (1-xp)*dx1
                # dx1 = xp * block0.flayer3(ctx)  + (1-xp)*dx1
                # dx1 = xp * ym  + (1-xp)*dx1
                xp  = att[:,:,1:2]
                dx0 = block0.flayer(ctx)


                # dx3 = torch.einsum('btk,btek->bte', att,dx3k)

                dx1 = dx1/(0.001 + dx1.std(-1,keepdims=True))              
                dx0 = dx0/(0.001 + dx0.std(-1,keepdims=True))            
                # dx3 = dx3/(0.001 + dx3.std(-1,keepdims=True))            
                dx0h[:,:,:,layer_i]   = dx0
                outh[:,:,:,layer_i]   = out_gate



        # x = dx0
        # x = torch.einsum('btel,btl->bte',dx0h, outh.softmax(-1).squeeze(2))
        # dx0h = dx0h * outh.softmax(-1)
        dx0h = dx0h * outh.sigmoid()
        x = md.final_output( dx0h.reshape((b,t,-1)) )
        
        x0 = x

        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)
        # fblock.flayer.weight )

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss                      


class CCM49(GPT):
    '''
    minimal interactive machine
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 10
        ng = config.ng
        nh = config.n_head


        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            final_output = nn.Linear(ne*(nl), ne,bias=False),
            gate_attn = nn.Linear(ne*(ng+1), ng,bias=True),
            k_attn = nn.Linear(ne*ng, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne,bias=False),
                k_attn_2 = nn.Linear(ne, ne,bias=False),
                flayer = nn.Linear(ne*(ng+1),    ne,bias=False),
                # flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(2+1),    ne,bias=False),
                flayer3 = nn.Linear(ne*(2+1),    ne,bias=False),
                flayer2b = nn.Linear(ne*(2),    ne,bias=False),
                flayer4 = nn.Linear(ne*(ng+1),    ne,bias=False),
                o_gate =  nn.Linear(ne*(ng+1), 1, bias=True),
                )
                # ) for _ in range(config.n_layer)
                ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0

        # pos = torch.cat([pos[:pad],pos],0)
        # idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb   = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb   = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        # pos_emb[:pad]= pos_emb_2[:pad]

        ### init internal embedding
        ### ng is the number of internal vectors
        int_emb   = self.transformer.wpe2(torch.zeros((b,t,ng),device=device).long() + pos[None,None,:ng])
        

        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx    = 0.
        nstep = 1



        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head

        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,e))
            xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('bte,bpke->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            y    = torch.einsum('btp,bpe->bte',att,xn)
            return y,att

        # for ii in range(nstep):
        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        # dx2 = xn*0+1
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        dx3 = dx2
        dx0 = dx2
        # nh  = self.config.n_head
        dx0h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        int_state = int_emb.clone()
        for ii in range(1):

            for layer_i in range(nl):                
                '''
                step 1250: train loss 3.8440, val loss 4.7829
                '''
                
                #### getting the two attended vector

                ### (b,t,e)
                ### getting input for both input and internal                

                wk    = block0.k_attn.weight.T
                xq    = dx1.matmul(wk)
                
                ### (b,t,k)
                
                engk = torch.einsum('bte,btke->btk',xq , int_state) 
                ### (b,t,p)
                engp = xq @ xn.transpose(1,2) 
                # engk, engp
                engp   = engp.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                att    = (torch.cat([engk,engp],dim=2)  / (ne ** 0.5) ).softmax(-1)
                yinput = torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) + att[:,:,ng:] @ xn


                wk    = block0.k_attn_2.weight.T
                xq    = dx1.matmul(wk)

                engk = torch.einsum('bte,btke->btk',xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                yinput2 = torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) 
                atto = att
                # print(layer_i, atto.mean((0,1)))

                # yinput,  att  = get_context(wk, dx1, torch.cat([ int_state, xn],dim=1))

                ### (b,t,e)
                ### (b,t,k)
                # yinput2, atto = get_context(wk, dx1, int_state)

                ctx=  torch.cat([yinput,yinput2,dx1],dim=2)
                yout = block0.flayer2(ctx)
                # yout = yout.unsqueeze(-2) + int_emb
                yout = yout.unsqueeze(-2) 
                # yout = yout.unsqueeze(-2) #+ int_emb
                dx1  = block0.flayer3(ctx)

                atto = atto.unsqueeze(-1)
                int_state = atto*yout + (1- atto) *int_state
                int_state = int_state / (0.001 + int_state.std(-1,keepdims=True))
                dx1       = dx1/(0.001 + dx1.std(-1,keepdims=True))              




        x = int_state[:,:,-1,:]
        x0 = x
        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()
        lreg += regnorm(  self.lm_head.weight)

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss                      


class CCM50(GPT):
    '''
    minimal interactive machine
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 12
        config.share_layer = False
        ng = config.ng
        nh = config.n_head


        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            final_output = nn.Linear(ne*(nl), ne,bias=False),
            gate_attn = nn.Linear(ne*(ng+1), ng,bias=True),
            k_attn = nn.Linear(ne*ng, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne,bias=False),
                k_attn_2 = nn.Linear(ne, ne,bias=False),
                k_attn_3 = nn.Linear(ne, ne,bias=False),
                flayer = nn.Linear(ne*(ng+1),    ne,bias=False),
                # flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(2+1),    ne,bias=False),
                flayer3 = nn.Linear(ne*(2+1),    ne,bias=False),
                # flayer2b = nn.Linear(ne*(2),    ne,bias=False),
                flayer4 = nn.Linear(ne*(2+1),    ne,bias=False),
                # o_gate =  nn.Linear(ne*(ng+1), 1, bias=True),
                )
                ) for _ in range(config.n_layer if not config.share_layer else 1) 
                # ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0

        # pos = torch.cat([pos[:pad],pos],0)
        # idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb   = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb   = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # pos_emb[:pad]= pos_emb_2[:pad]

        ### init internal embedding
        ### ng is the number of internal vectors
        int_emb   = self.transformer.wpe2(torch.zeros((b,t,ng),device=device).long() 
            + torch.arange(ng,device=device)[None,None,])

        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx    = 0.
        nstep = 1



        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head

        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        # dx2 = xn*0+1
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        # nh  = self.config.n_head
        dx0h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        int_state = int_emb.clone()
        for ii in range(1):

            for layer_i in range(nl):                
                '''
                step 1250: train loss 3.8440, val loss 4.7829
                '''
                if not self.config.share_layer:
                    ### variable layer
                    block0 = self.transformer.h[layer_i]
                else:
                    ### constant layer
                    block0 = self.transformer.h[0]
                
                #### getting the two attended vector

                ### (b,t,e)
                ### getting input for both input and internal                

                wk    = block0.k_attn.weight.T
                xq    = dx1.matmul(wk)
                
                ### (b,t,k)
                
                engk = torch.einsum('bte,btke->btk',xq , int_state) 
                ### (b,t,p)
                engp = xq @ xn.transpose(1,2) 
                # engk, engp
                engp   = engp.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                att    = (torch.cat([engk,engp],dim=2)  / (ne ** 0.5) ).softmax(-1)
                yinput = torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) + att[:,:,ng:] @ xn


                wk     = block0.k_attn_2.weight.T
                xq     = dx1.matmul(wk)
                engk   = torch.einsum('bte,btke->btk',xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                yinput2= torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) 
                atto   = att

                wk     = block0.k_attn_3.weight.T
                xq     = dx1.matmul(wk)
                engk   = torch.einsum('bte,btke->btk',xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                atto2  = att

                # yinput,  att  = get_context(wk, dx1, torch.cat([ int_state, xn],dim=1))

                ### (b,t,e)
                ### (b,t,k)
                # yinput2, atto = get_context(wk, dx1, int_state)

                ctx=  torch.cat([yinput,yinput2,dx1],dim=2)
                yout = block0.flayer2(ctx)
                # yout = yout.unsqueeze(-2) + int_emb
                yout = yout.unsqueeze(-2) 
                # yout = yout.unsqueeze(-2) #+ int_emb
                dx1  = block0.flayer3(ctx)

                atto = atto.unsqueeze(-1)
                # int_state = atto*yout + (1- atto) *int_state
                int_state = atto*yout + int_state + atto2.unsqueeze(-1) * block0.flayer4(ctx).unsqueeze(-2)
                # int_state = int_state + int_emb
                int_state = int_state / (0.001 + int_state.std(-1,keepdims=True))
                dx1       = dx1/(0.001 + dx1.std(-1,keepdims=True))              



        x = int_state[:,:,-1,:]
        x0 = x
        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss           

class CCM51(GPT):
    '''
    minimal interactive machine
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # ng = 2
        config.ng = 12
        ng = config.ng
        nh = config.n_head


        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            final_output = nn.Linear(ne*(nl), ne,bias=False),
            gate_attn = nn.Linear(ne*(ng+1), ng,bias=True),
            k_attn = nn.Linear(ne*ng, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne,bias=False),
                k_attn_2 = nn.Linear(ne, ne,bias=False),
                k_attn_3 = nn.Linear(ne, ne,bias=False),
                flayer = nn.Linear(ne*(ng+1),    ne,bias=False),
                # flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(2+1),    ne,bias=False),
                flayer3 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2b = nn.Linear(ne*(2),    ne,bias=False),
                flayer4 = nn.Linear(ne*(2+1),    ne,bias=False),
                o_gate =  nn.Linear(ne*(ng+1), 1, bias=True),
                )
                ) for _ in range(config.n_layer)
                # ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0

        # pos = torch.cat([pos[:pad],pos],0)
        # idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb   = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb   = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # pos_emb[:pad]= pos_emb_2[:pad]

        ### init internal embedding
        ### ng is the number of internal vectors
        int_emb   = self.transformer.wpe2(torch.zeros((b,t,ng),device=device).long() + pos[None,None,:ng])
        

        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx    = 0.
        nstep = 1



        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head

        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,e))
            xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('bte,bpke->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            y    = torch.einsum('btp,bpe->bte',att,xn)
            return y,att

        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        # dx2 = xn*0+1
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        # nh  = self.config.n_head
        dx0h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        int_state = int_emb.clone()
        for ii in range(1):

            for layer_i in range(nl):                
                '''
                step 1250: train loss 3.8440, val loss 4.7829
                '''
                
                #### getting the two attended vector
                block0 = self.transformer.h[layer_i]

                ### (b,t,e)
                ### getting input for both input and internal                
                wk    = block0.k_attn.weight.T
                xq    = dx1.matmul(wk)                
                ### (b,t,k)                
                engk = torch.einsum('bte,btke->btk',xq , int_state) 
                ### (b,t,p)
                engp = xq @ xn.transpose(1,2) 
                # engk, engp
                engp   = engp.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                att    = (torch.cat([engk,engp],dim=2)  / (ne ** 0.5) ).softmax(-1)
                yinput = torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) + att[:,:,ng:] @ xn


                wk     = block0.k_attn_2.weight.T
                xq     = dx1.matmul(wk)
                engk   = torch.einsum('bte,btke->btk',xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                yinput2= torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) 
                atto   = att

                wk     = block0.k_attn_3.weight.T
                xq     = dx1.matmul(wk)
                engk   = torch.einsum('bte,btke->btk',xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                y3      = torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) 
                atto2  = att

                # yinput,  att  = get_context(wk, dx1, torch.cat([ int_state, xn],dim=1))

                ### (b,t,e)
                ### (b,t,k)
                # yinput2, atto = get_context(wk, dx1, int_state)

                ctx=  torch.cat([yinput,yinput2,dx1],dim=2)
                yout = block0.flayer2(ctx)
                # yout = yout.unsqueeze(-2) + int_emb
                yout = yout.unsqueeze(-2) 
                # yout = yout.unsqueeze(-2) #+ int_emb
                # dx1  = block0.flayer3(ctx)
                dx1  = block0.flayer3(torch.cat([ctx,y3],dim=-1))

                atto = atto.unsqueeze(-1)
                # int_state = atto*yout + (1- atto) *int_state
                int_state = atto*yout + int_state + atto2.unsqueeze(-1) * block0.flayer4(ctx).unsqueeze(-2)
                # int_state = int_state + int_emb
                int_state = int_state / (0.001 + int_state.std(-1,keepdims=True))
                dx1       = dx1/(0.001 + dx1.std(-1,keepdims=True))              





        x = int_state[:,:,-1,:]
        x0 = x
        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss                      


class CCM52(GPT):
    '''
    minimal interactive machine
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # ng = 2
        config.ng = 12
        ng = config.ng
        nh = config.n_head


        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            final_output = nn.Linear(ne*(nl), ne,bias=False),
            gate_attn = nn.Linear(ne*(ng+1), ng,bias=True),
            k_attn = nn.Linear(ne*ng, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(2*ne, ne,bias=False),
                k_attn_2 = nn.Linear(2*ne, ne,bias=False),
                k_attn_3 = nn.Linear(2*ne, ne,bias=False),
                k_attn_4 = nn.Linear(2*ne, ne,bias=False),
                flayer = nn.Linear(ne*(ng+1),    ne,bias=False),
                # flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer3 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2b = nn.Linear(ne*(2),    ne,bias=False),
                flayer4 = nn.Linear(ne*(3+1),    ne,bias=False),
                o_gate =  nn.Linear(ne*(ng+1), 1, bias=True),
                )
                # ) for _ in range(config.n_layer)
                ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0

        # pos = torch.cat([pos[:pad],pos],0)
        # idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb   = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb   = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # pos_emb[:pad]= pos_emb_2[:pad]

        ### init internal embedding
        ### ng is the number of internal vectors
        int_emb   = self.transformer.wpe2(torch.zeros((b,t,ng),device=device).long() + pos[None,None,:ng])
        

        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx    = 0.
        nstep = 1



        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head

        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,e))
            xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('bte,bpke->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            y    = torch.einsum('btp,bpe->bte',att,xn)
            return y,att

        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        # dx2 = xn*0+1
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        # nh  = self.config.n_head
        dx0h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        int_state = int_emb.clone()
        for ii in range(1):

            for layer_i in range(nl):                
                '''
                step 1250: train loss 3.8440, val loss 4.7829
                '''
                
                #### getting the two attended vector

                ### (b,t,e)
                ### getting input for both input and internal                
                wk    = block0.k_attn.weight.T
                ctx0  = torch.cat([dx1,dx2],dim=2)
                xq    = ctx0.matmul(wk) 
                xq    = xq / (0.001 + xq.std(-1,keepdims=True))
                ### (b,t,k)                
                engk = torch.einsum('bte,btke->btk',xq , int_state) 
                ### (b,t,p)
                engp = xq @ xn.transpose(1,2) 
                # engk, engp
                engp   = engp.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                att    = (torch.cat([engk,engp],dim=2)  / (ne ** 0.5) ).softmax(-1)
                yinput = torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) + att[:,:,ng:] @ xn


                wk     = block0.k_attn_2.weight.T
                xq     = ctx0.matmul(wk)
                xq    = xq / (0.001 + xq.std(-1,keepdims=True))
                engk   = torch.einsum('bte,btke->btk',xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                yinput2= torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) 
                atto   = att

                wk     = block0.k_attn_3.weight.T
                xq     = ctx0.matmul(wk)
                xq    = xq / (0.001 + xq.std(-1,keepdims=True))

                engk   = torch.einsum('bte,btke->btk',xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                y3      = torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) 
                atto2  = att

                wk     = block0.k_attn_4.weight.T
                xq     = ctx0.matmul(wk)
                xq    = xq / (0.001 + xq.std(-1,keepdims=True))
                engk   = torch.einsum('bte,btke->btk',xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                y      = torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) 
                atto3  = att                

                # yinput,  att  = get_context(wk, dx1, torch.cat([ int_state, xn],dim=1))

                ### (b,t,e)
                ### (b,t,k)
                # yinput2, atto = get_context(wk, dx1, int_state)

                ctx  = torch.cat([yinput,yinput2,ctx0],dim=2)
                yout = block0.flayer2(ctx)
                # yout = yout.unsqueeze(-2) + int_emb
                yout = yout.unsqueeze(-2) 
                # yout = yout.unsqueeze(-2) #+ int_emb
                dx1  = block0.flayer3(ctx)
                dx2  = block0.flayer4(ctx)

                # atto = atto.unsqueeze(-1)
                # int_state = atto*yout + (1- atto) *int_state
                int_state = atto3.unsqueeze(-1)*yout + int_state + atto2.unsqueeze(-1) * block0.flayer4(ctx).unsqueeze(-2)
                # int_state = int_state + int_emb
                int_state = int_state / (0.001 + int_state.std(-1,keepdims=True))
                dx1       = dx1/(0.001 + dx1.std(-1,keepdims=True))              
                dx2       = dx2/(0.001 + dx2.std(-1,keepdims=True))              




        x = int_state[:,:,-1,:]
        x0 = x
        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss              

class CCM28(GPT):
    '''
    reparametrise the model to use product of SU(2)



    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # config.dropout = 0.0
        # print (config)
        config.n_head =  1
        # config.block_size = config.block_size + 3
        # config.g = 100
        # config.g = 700
        # config.g = config.n_embd
        config.g = 100
        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            wpe2 = nn.Embedding(config.block_size, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve   = nn.Embedding(config.g, config.n_embd*ng),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd*ng, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0
 
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask = torch.tril(att_mask, diagonal=-1)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True


        md = self.transformer

        knorm   = md.wke.weight
        # knorm = knorm - knorm.sum(dim=1,keepdim=True)
        # knorm = self.transformer.ln_f(knorm)
        # knorm = md.wke.weight.T
        vnorm   = md.wve.weight
        vnorm2  = md.wve2.weight
        # vnorm = self.transformer.ln_f(vnorm).T

        dx = 0.
        nstep = 2
        nstep = 5
        # nstep = 1
        # theta = torch.cos()

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        # torch.autograd.set_detect_anomaly(True)

        for ii in range(nstep):
            xh = x[:,:,:e//2]
            y  = torch.cat([torch.sin(xh),torch.cos(xh)],dim=-1) # /e**0.5
            # xn = x - x.mean(-1,keepdims=True)
            # xn = xn/(0.001 + xn.std(-1,keepdims=True))
            dx = 0.

            for i,block in enumerate(self.transformer.h):
                attn      = block.attn
                xq, k, v  = attn.c_attn(y).split(attn.n_embd, dim=2)
                att       = xq @ y.transpose(2,1)/attn.n_embd**0.5
                att       = att.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                dx        = dx + att.softmax(-1) @ v                
                # dx = dx + block.attn(xn)

            dx = dx+ block.mlp(y)
            dy = dx
            # dy = dy.clip(-0.3,0.3)
            # dxdy = (1.0001-y.square())*0.5
            dxdy =  torch.cat([torch.cos(xh),-torch.sin(xh)],dim=-1)
            
            # dx = torch.cos(x)*(dy)
            dx = dxdy*(dy)
            dx = dx[:,:,:e//2] + dx[:,:,e//2:]
            # dx = torch.cat((dx[:,:,:e//2]))
            dx = dx.clip(-0.5,0.5)
            ddx = torch.zeros((b,t,e),device=device)
            ddx[:,:,:e//2] = dx

            # x[:,:,:e//2]  = x[:,:,:e//2] + dx
            x = x + ddx

        xh = x[:,:,:e//2]
        y  = torch.cat([torch.sin(xh),torch.cos(xh)],dim=-1) # /e**0.5

        # x = torch.sin(x)/e**0.5

        x = self.transformer.ln_f(y)        
        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss



class CCM29(GPT):
    '''
    using gaussian displacement for updates

    step 1500: train loss 4.1113, val loss 4.9812

    gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # config.dropout = 0.0
        # print (config)
        config.n_head =  1
        # config.block_size = config.block_size + 3
        # config.g = 100
        # config.g = 700
        # config.g = config.n_embd
        config.g = 100
        # config.nh = 3
        # ng = 2
        config.ng = 2
        ng = config.ng
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            wpe2 = nn.Embedding(config.block_size, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve   = nn.Embedding(config.g, config.n_embd*ng),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd*ng, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0
 
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask = torch.tril(att_mask, diagonal=-1)
        # att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True


        md = self.transformer

        knorm   = md.wke.weight
        # knorm = knorm - knorm.sum(dim=1,keepdim=True)
        # knorm = self.transformer.ln_f(knorm)
        # knorm = md.wke.weight.T
        vnorm   = md.wve.weight
        vnorm2  = md.wve2.weight
        # vnorm = self.transformer.ln_f(vnorm).T

        dx = 0.
        nstep = 1 

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd

        # xn = x - x.mean(-1,keepdims=True)
        # xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # x = xn

        for ii in range(nstep):
            dx = 0.
            y = x.sigmoid() - 0.5

            for i,block in enumerate(self.transformer.h):
                attn      = block.attn
                xq, k, v  = attn.c_attn(y).split(attn.n_embd, dim=2)
                att       = xq@k.transpose(2,1)/attn.n_embd**0.5
                att       = att.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                dx        = dx + att.softmax(-1) @ v                
                # dx = dx + block.attn(xn)

            dx = dx+ block.mlp(y)


            ### step 1000: train loss 3.9426, val loss 4.7900
            ### step 1000: train loss 3.9426, val loss 4.7900


            ### correct the gradient on sphere
            # dx = dx - torch.einsum('bce,bc->bce', xn, torch.einsum('bce,bce->bc',dx,xn)).detach()
            # print([x.square().mean(-1).mean(), dx.square().mean(-1).mean()])
            # print(x.square().sum(-1).mean())
            dy = dx
            dx = dy * (0.5+y) * (0.5-y)
            # x  = x + dx
            # x  = x + dx.clip(-0.2,0.2)
  
            # xn = x - x.mean(-1,keepdims=True)
            # xn = xn/(0.001 + xn.std(-1,keepdims=True))
            # x = xn
        x = x.sigmoid() - 0.5

        x = self.transformer.ln_f(x)        
        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss



class CCM22(GPT):
    '''

    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        # super(GPT,self).__init__(config)
        super(GPT,self).__init__()

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
        config.nc = 2
        nc= config.nc
        # config.g = 45
        config.g = 15

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size+2, config.n_embd),
            wpe2 = nn.Embedding(config.block_size+2, config.n_embd),
            wpe3 = nn.Embedding(config.block_size+2, config.n_embd),
            whe  = nn.Embedding(config.nc, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output   = nn.Linear(ne, ne, bias=False),
            lin_output_2   = nn.Linear(ne, ne,bias=False),
            lin_output_3   = nn.Linear(ne, ne,bias=False),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,ne,bias=False),
            # lin_transit_k= nn.Linear(ne,config.g*ne,bias=True),
            lin_transit_k= nn.Linear(ne,config.g*ne,bias=False),
            lin_transit_v= nn.Linear(ne,config.g*ne,bias=False),
            # lin_transit_k  = nn.Embedding(config.n_embd, config.g  *config.n_embd,bias=True),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input   = nn.Linear(ne,ne,bias=False),
            lin_input_2   = nn.Linear(ne,ne,bias=False),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # self.att_bias = 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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

        md = self.transformer ### model dict
        nc = self.config.nc
        head = pos[:nc]
        # forward the GPT model itself
        pad = 6
        pos = torch.cat([pos[:pad],pos],0)
        idx = torch.cat([idx[:,:pad]*0,idx],1)


        tok_emb  = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb  = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2  = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        pos_emb_3  = self.transformer.wpe3(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2[:pad] = pos_emb_3[:pad]

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

        ### cpt is the post context at position 
        t = t+pad

        ### positional embedding is enough to store prior knowledge on causality between tokens
        cpt      = (torch.zeros((b,t,e),device=device) + pos_emb_2 )
 
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=0)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask = torch.tril(att_mask, diagonal=0)
        att_mask[0,0]=True
        md       = self.transformer

        # ### (tp,tc)        
        # att_mask = att_mask.T
        # cpt   = self.transformer.ln_f(F.softplus(cpt))
        cpt   = self.transformer.ln_f(cpt)

        # xnorm = self.transformer.ln_f(F.softplus(x))
        xnorm = self.transformer.ln_f(x)

        knorm = md.wke.weight
        # knorm = knorm - knorm.sum(dim=1,keepdim=True)
        knorm = self.transformer.ln_f(knorm).T

        # knorm = md.wke.weight.T
        

        # vnorm = md.wve.weight
        # vnorm = self.transformer.ln_f(vnorm).T
        vnorm = md.wve.weight.T

        g = self.config.g

        # for i in range(4):
        # for i in range(4):
        for i in range(3):
        # for i in range(1):
        # for i in range(2):
            ### (b,c,e),(e,k)->bcek
            ### (bcek)

            # cpred = cpt.unsqueeze(-1) + 0.5 * vnorm[None,None] 

            # cpred = (cpt + 0.5 * md.lin_transit_2(cpt)).unsqueeze(-1) 
            # cpred = md.lin_transit_k(cpt).reshape((b,t,e,g))
            # cpred = 0.5*cpred + cpt.unsqueeze(-1)
            # cpred = cpt.unsqueeze(-1) + 0.5 * vnorm[None,None] 
            # cpred = 

            ### using vk combination of matrices

            # cptk = md.lin_transit_k(cpt).reshape((b,t,e,g))
            # # cptv = md.lin_transit_v(cpt).reshape((b,t,e,g)).relu()
            # cptv = md.lin_transit_v(cpt).reshape((b,t,e,g))#.tanh()
            # att0 = torch.einsum('bpe,bpeg->bpg', cpt, cptk).softmax(-1)
            # cpred = 0.
            # cpred += torch.einsum('bpg,bpeg->bpe', att0, cptv)
            # cpred_d = cpred


            cpred_d = 0.

            cpred = cpt.matmul(md.lin_transit.weight)
            ### (b,tp,nh,tc)
            lp   = 0.
            #### gradient to predict the token
            lp += torch.einsum('bpe,bce ->bcp', cpred,  cpt )/e**0.5

            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp   = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))
            # att  = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)
            att  = lp.softmax(-1)
            # print(att.mean((0,1)).sum(0))


            dcpt = 0.
            dcpt += torch.einsum('bcp,bpe->bce', att, cpred)
            # dcpt += torch.einsum('bcp,bpe->bce', att2, cpred2)
            #  + xnorm.matmul(md.lin_output_2.weight)
            dcpt += 0.5 * (x + x.matmul(md.lin_output_2.weight))
            # dcpt += x 
            # x.matmul(md.lin_output_2.weight)
            dcpt += cpred_d
            dcpt = dcpt/3.
            # dcpt = dcpt * 0.5

            cpt  +=  0.7 * (dcpt - cpt)
            # cpt  +=  1.0 * (dcpt - cpt)
            cpt   = self.transformer.ln_f((cpt))
            cpt += torch.normal(0,0.05, cpt.shape,device=device)



        ### inducing the next token using the pos_embedding
        cpt_ind = cpt
        # y       = md.lin_output_3(cpt_ind) + cpt
        y       = md.lin_output_3(cpt_ind).relu() + cpt
        # y       = md.lin_output_3(cpt_ind)  

        # ### step 1750: train loss 3.9248, val loss 4.8322
        y     = y[:,pad:]
        y     = self.transformer.ln_f((y))
        # wt2   = F.softplus(self.lm_head.weight)

        # logits = y.matmul(wt2.T)
        # logits = logits.log_softmax(-1)

        logits = self.lm_head(y)

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
            if self.training:
                pass
                # loss = loss - lps
        else:
            loss  = None
        return logits, loss           




class CCM25(GPT):
    '''

using gaussian displacement for updates

step 1500: train loss 4.1113, val loss 4.9812


gpt01:4.70


    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1
        config.n_head =  1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # config.dropout = 0.0
        # print (config)
        config.n_head =  1
        # config.block_size = config.block_size + 3
        # config.g = 100
        # config.g = 700
        config.g = config.n_embd
        # config.g = 500
        # config.nh = 3
        # ng = 2
        config.ng = 4
        ng = config.ng
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            wpe2 = nn.Embedding(config.block_size, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve   = nn.Embedding(config.g, config.n_embd*ng),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd*ng, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself

        # assert 0
        # pad = 17
        pad = 0
        # pos = torch.cat([pos[:pad],pos],0)
        # idx = torch.cat([idx[:,:pad]*0,idx],1)

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        pos_emb_2 = self.transformer.wpe2(pos) # position embeddings of shape (t, n_embd)
        # pos_emb[:pad] = pos_emb_2[:pad]
        # tok_emb[:,:pad] = pos_emb_2[:pad]

        x = self.transformer.drop(tok_emb + pos_emb)
        
        # x = self.transformer.ln_f(x)
        # x = self.transformer.h[0].ln_1(x)
        

        md = self.transformer

        knorm = md.wke.weight
        # knorm = knorm - knorm.sum(dim=1,keepdim=True)
        # knorm = self.transformer.ln_f(knorm)

        # knorm = md.wke.weight.T
        vnorm = md.wve.weight
        vnorm2 = md.wve2.weight
        # vnorm = self.transformer.ln_f(vnorm).T

        dx = 0.
        nstep = 1

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        for ii in range(nstep):
            # xn = block.ln_1(x)
            xn = x - x.mean(-1,keepdims=True)
            xn = xn/(0.001 + xn.std(-1,keepdims=True))

            # for i,block in enumerate(self.transformer.h[:3]):
            for i,block in enumerate(self.transformer.h):
                dx = dx+ block.attn(xn)
            # dx = dx+ block.mlp(xn)
            knormm = self.transformer.ln_f(knorm)
            att = torch.einsum('bce,ke->bck', xn, knormm).softmax(-1)
            # dxx = torch.einsum( 'bck,ke->bce',att, vnorm.reshape((g,ng,e))[:,0], )

            dxx = torch.einsum('bck,kve,bce->bcvk', att, vnorm.reshape((g,ng,e)), xn)

            # dxx = torch.einsum('bck,kve,bce->bcvk', att, vnorm.reshape((g,ng,e)), xn - torch.einsum('bck,ke->bce',att, knormm ))
            dxx = torch.einsum('bcvk,kve->bce', dxx, vnorm2.reshape((g,ng,e)))

            # dx = dx + md.proj(dxx)
            dx += dxx

            x  = x+ dx


        x = self.transformer.ln_f(x)
        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss





import torch.nn as nn


# class CCM01(nn.Module):
class CCM21(GPT):
    '''

    ### maybe overfitting?

    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k    = 5
        config.g    = 200
        config.nrep = 2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        # config.n_internal = config.n_embd//16
        config.n_internal = config.n_embd//4


        # config.block_size = config.block_size+config.k
        # config.block_size = config.block_size*4
        config.method = -1
        config.n_head = 1

        config.suffix = f'{config.optimizer}-method{config.method}'
        config.is_causal=True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config


    def __init__(self, config):
        super(GPT,self).__init__()

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
            # k_vects = nn.Embedding(config.g, config.n_embd),
            # g_vects = nn.Embedding(config.g, config.n_embd),
            drop = nn.Dropout(config.dropout),
            lin_output   = nn.Linear(ne, nh*nei,bias=config.bias),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit   = nn.Linear(ne,nh*nei,bias=config.bias),
            # lin_transit_2 = nn.Linear(nh*nei, ne,bias=config.bias),
            lin_input     = nn.Linear(ne,nei,bias=config.bias),
            lin_input_2   = nn.Linear(ne,ne,bias=config.bias),
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
        cpt      = torch.zeros((b,t,e),device=device)+ x
        x_init   = x
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### need to concat a single prefix to avoid
        # att_mask = torch.tril(att_mask, diagonal=1)

        ### (tc,tp)
        att_mask = torch.tril(att_mask, diagonal=0)
        nei      = config.n_internal
        md       = self.transformer



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
            # cpred = cpred.sigmoid()
            return cpred

        for i in range(4):
            xpred = get_xpred(cpt)
            cpred = get_cpred(x,cpt)

            ### (b,tp,nh,tc)
            lp = 0.
            # lp += xpred.matmul(xrep) 
            lp += torch.einsum('bphe,bce->bphc', xpred, x.matmul(md.lin_internal_output.weight)) 
            lp += torch.einsum('bpehc,bce->bphc',cpred, cpt.matmul(md.lin_internal.weight))

            ### (b,tc,nh,tp)
            lp = lp.transpose(1,3)
            
            lp = lp.masked_fill(att_mask[None,:,None,:]==0,float('-inf'))

            ### (b,tc,nh,tp)
            # att = lp.reshape.softmax((2,3))
            att = lp.reshape((b,t,-1)).softmax(-1).reshape(lp.shape)
            # 2,3))
            # breakpoint()    
            dcpt = torch.einsum('bchp,bpehc->bce', att, cpred).matmul(md.lin_internal.weight.T)
            cpt = cpt + 0.1*dcpt

        # xpred = md.lin_output(dcpt).reshape((b,t,nh,nei))
        xpred = get_xpred(cpt)
        xpred = xpred.matmul(md.lin_internal_output.weight.T).reshape((b,t,nh,ne))
        x     = xpred

        x     = self.transformer.ln_f(x)            

        logits= self.lm_head(x).log_softmax(-1).logsumexp(2)-math.log(nh)


        # if targets is not None:
        #     lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=self.config.mask_index,reduction='none')
        #     # lp_external  = -F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        # else:
        #     lp_external = torch.zeros_like(logits[:,:,0])

        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        else:
            loss  = None
        return logits, loss         



class CCM53(GPT):
    '''
    minimal interactive machine
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # config.nh = 3
        # ng = 2
        config.ng = 12
        ng = config.ng
        nh = config.n_head


        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            final_output = nn.Linear(ne*(nl), ne,bias=False),
            gate_attn = nn.Linear(ne*(ng+1), ng,bias=True),
            k_attn = nn.Linear(ne*ng, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne,bias=False),
                k_attn_2 = nn.Linear(ne, ne,bias=False),
                k_attn_3 = nn.Linear(ne, ne,bias=False),
                k_attn_4 = nn.Linear(ne, ne,bias=False),
                k_attn_5 = nn.Linear(ne, ne,bias=False),
                flayer = nn.Linear(ne*(ng+1),    ne,bias=False),
                # flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(2+1),    ne,bias=False),
                flayer3 = nn.Linear(ne*(2+1),    ne,bias=False),
                flayer4 = nn.Linear(ne*(2+1),    ne,bias=False),
                flayer5 = nn.Linear(ne*(2+1),    ne,bias=False),
                flayer2b = nn.Linear(ne*(2),    ne,bias=False),
                o_gate =  nn.Linear(ne*(ng+1), 1, bias=True),
                )
                # ) for _ in range(config.n_layer)
                ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0

        # pos = torch.cat([pos[:pad],pos],0)
        # idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb   = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb   = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # pos_emb[:pad]= pos_emb_2[:pad]

        ### init internal embedding
        ### ng is the number of internal vectors
        int_emb   = self.transformer.wpe2(torch.zeros((b,t,ng),device=device).long() + pos[None,None,:ng])
        

        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx    = 0.
        nstep = 1



        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head

        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,e))
            xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('bte,bpke->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            y    = torch.einsum('btp,bpe->bte',att,xn)
            return y,att

        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        # dx2 = xn*0+1
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        # nh  = self.config.n_head
        dx0h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        
        int_emb = int_emb / (0.001 + int_emb.std(-1,keepdims=True))

        int_state = int_emb.clone()
        for ii in range(1):

            for layer_i in range(nl):                
                '''
                step 1250: train loss 3.8440, val loss 4.7829
                '''
                
                #### getting the two attended vector

                ### (b,t,e)
                ### getting input for both input and internal                

                # int_state = int_state + int_emb
                # int_state = int_state / (0.001 + int_state.std(-1,keepdims=True))


                wk    = block0.k_attn.weight.T
                xq    = dx1.matmul(wk)                
                ### (b,t,k)                
                engk = torch.einsum('bte,btke->btk',xq , int_state) 
                ### (b,t,p)
                engp = xq @ xn.transpose(1,2) 
                engp   = engp.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                att    = (torch.cat([engk,engp],dim=2)  / (ne ** 0.5) ).softmax(-1)
                yinput = torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) + att[:,:,ng:] @ xn



                wk     = block0.k_attn_2.weight.T
                xq     = dx1.matmul(wk)
                engk   = torch.einsum('bte,btke->btk',xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                yinput2= torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) 
                atto   = att


                wk    = block0.k_attn_3.weight.T
                xq    = dx1.matmul(wk)                
                ### (b,t,k)                
                engk = torch.einsum('bte,btke->btk',xq , int_state) 
                ### (b,t,p)
                engp = xq @ xn.transpose(1,2) 
                engp   = engp.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                att    = (torch.cat([engk,engp],dim=2)  / (ne ** 0.5) ).softmax(-1)
                yinput3 = torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) + att[:,:,ng:] @ xn

                wk     = block0.k_attn_4.weight.T
                xq     = dx1.matmul(wk)
                engk   = torch.einsum('bte,btke->btk',xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                atto4  = att

                wk     = block0.k_attn_5.weight.T
                xq     = dx1.matmul(wk)
                engk   = torch.einsum('bte,btke->btk',xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                atto5  = att


                # yinput,  att  = get_context(wk, dx1, torch.cat([ int_state, xn],dim=1))

                ### (b,t,e)
                ### (b,t,k)
                # yinput2, atto = get_context(wk, dx1, int_state)

                ctx=  torch.cat([yinput,yinput3,dx1],dim=2)
                # yout = yout.unsqueeze(-2) #+ int_emb
                dx1  = block0.flayer3(ctx)

                # int_state = atto*yout + (1- atto) *int_state
                int_state += atto.unsqueeze(-1)  * block0.flayer2(ctx).unsqueeze(-2)  
                # int_state = int_state / (0.001 + int_state.std(-1,keepdims=True))
                int_state += atto4.unsqueeze(-1) * block0.flayer4(ctx).unsqueeze(-2)
                # int_state = int_state / (0.001 + int_state.std(-1,keepdims=True))
                int_state += atto5.unsqueeze(-1) * block0.flayer5(ctx).unsqueeze(-2)
                # int_state = int_state + int_emb
                int_state = int_state / (0.001 + int_state.std(-1,keepdims=True))
                dx1       = dx1/(0.001 + dx1.std(-1,keepdims=True))              



        x = int_state[:,:,-1,:]
        x0 = x
        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss           



class CCM54(GPT):
    '''
    minimal interactive machine
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # ng = 2
        config.ng = 12
        ng = config.ng
        nh = config.n_head


        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            final_output = nn.Linear(ne*(nl), ne,bias=False),
            gate_attn = nn.Linear(ne*(ng+1), ng,bias=True),
            k_attn = nn.Linear(ne*ng, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne,bias=False),
                k_attn_2 = nn.Linear(ne, ne,bias=False),
                k_attn_3 = nn.Linear(ne, ne,bias=False),
                flayer = nn.Linear(ne*(ng+1),    ne,bias=False),
                # flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(2+1),    ne,bias=False),
                flayer3 = nn.Linear(ne*(3),    ne,bias=False),
                flayer2b = nn.Linear(ne*(2),    ne,bias=False),
                flayer4 = nn.Linear(ne*(2+1),    ne,bias=False),
                o_gate =  nn.Linear(ne*(ng+1), 1, bias=True),
                )
                # ) for _ in range(config.n_layer)
                ) for _ in range(1)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                # torch.nn.init.normal_(p, mean=0.0, std=1)
                # //math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0

        # pos = torch.cat([pos[:pad],pos],0)
        # idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0
        def regnorm(weight):
            return 0


        tok_emb   = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb   = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # pos_emb[:pad]= pos_emb_2[:pad]

        ### init internal embedding
        ### ng is the number of internal vectors
        int_emb   = self.transformer.wpe2(torch.zeros((b,t,ng),device=device).long() + pos[None,None,:ng])
        

        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)  
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md = self.transformer

        dx    = 0.
        nstep = 1



        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head

        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,e))
            xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('bte,bpke->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            y    = torch.einsum('btp,bpe->bte',att,xn)
            return y,att

        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        # dx2 = xn*0+1
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        # nh  = self.config.n_head
        dx0h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        int_state = int_emb.clone()

        ### (b, t, k, e)
        int_state[:,:,0] += xn 
        int_state = int_state / (0.001 + int_state.std(-1,keepdim=True))

        for ii in range(1):

            for layer_i in range(nl):                
                '''
                step 1250: train loss 3.8440, val loss 4.7829
                '''                
                ### getting the two attended vector...

                ### (b,t,e)
                ### getting input for both input and internal                
                wk    = block0.k_attn.weight.T

                xq    = int_state.matmul(wk) 
                xq     = xq[:,:,:1] 

                ### (b,t,k,k2)                
                engk   = torch.einsum('btke,btje->btkj', xq, int_state) 
                
                ### (b,t,k,p)
                engp   = torch.einsum('btke,bpe->btkp', xq, xn)
                # engp   = xq @ xn.transpose(1,2).unsqueeze(-1) 
                
                ### engk, engp
                engp   = engp.masked_fill(att_mask[None,:t,None,:t] == 0, float('-inf'))
                att    = (torch.cat([engk,engp],dim=-1)  / (ne ** 0.5) ).softmax(-1)
                yinput = (
                    torch.einsum('btkj,btje->btke', att[:,:,:,:ng], int_state) + 
                    torch.einsum('btkp,bpe->btke',att[:,:,:,ng:] , xn)
                )


                wk     = block0.k_attn_2.weight.T
                xq     = int_state.matmul(wk) 
                xq     = xq[:,:,:1] 
                engk   = torch.einsum('btke,btje->btkj', xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                yinput2= torch.einsum('btkj,btje->btke', att[:,:,:,:ng], int_state) 
                atto   = att

                wk     = block0.k_attn_3.weight.T
                xq     = int_state.matmul(wk) 
                xq     = xq[:,:,:1] 
                engk   = torch.einsum('btke,btje->btkj', xq , int_state)  / (ne ** 0.5)
                att    = engk.softmax(-1)
                y3     = torch.einsum('btkj,btje->btke', att[:,:,:,:ng], int_state) 
                atto2  = att

                # yinput,  att  = get_context(wk, dx1, torch.cat([ int_state, xn],dim=1))

                ### (b,t,e)
                ### (b,t,k)
                # yinput2, atto = get_context(wk, dx1, int_state)

                # ctx=  torch.cat([yinput,yinput2,int_state],dim=-1)
                ctx=  torch.cat([yinput,yinput2,int_state[:,:,:1]],dim=-1)
                # ctx = ctx[:,:,:1]
                # yout = block0.flayer2(ctx)
                # # yout = yout.unsqueeze(-2) + int_emb
                # yout = yout.unsqueeze(-2) 
                # yout = yout.unsqueeze(-2) #+ int_emb
                # dx1  = block0.flayer3(ctx)
                # dx1  = block0.flayer3(torch.cat([ctx,y3],dim=-1))

                int_state  += block0.flayer3(ctx)
                # int_state[:,:,:1]  = block0.flayer3(ctx)
                # int_state  = block0.flayer3(ctx)

                int_state += torch.einsum('btkj,btke->btje', atto2, block0.flayer4(ctx))

                # atto2.unsqueeze(-1) * block0.flayer4(ctx).unsqueeze(-2)
                int_state = int_state / (0.001 + int_state.std(-1,keepdims=True))


                # #   atto.unsqueeze(-1)

                # atto = atto.unsqueeze(-1)
                # # int_state = atto*yout + (1- atto) *int_state
                # int_state = atto*yout + int_state + atto2.unsqueeze(-1) * block0.flayer4(ctx).unsqueeze(-2)
                # # int_state = int_state + int_emb
                # int_state = int_state / (0.001 + int_state.std(-1,keepdims=True))
                # dx1       = dx1/(0.001 + dx1.std(-1,keepdims=True))              



        x = int_state[:,:,-1,:]
        x0 = x
        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss      


class CCM55(GPT):
    '''
    minimal interactive machine
    
    '''

    @classmethod
    def add_config(cls, config):
        config.mask_index = -1
        # config.optimizer ='rmsprop'
        config.optimizer ='adamw'

        config.k = 5
        # config.g = 10
        config.nrep=2
        config.use_dropout = 0
        ### lower rank for the internal embedding
        config.n_internal = config.n_embd//4

        config.method = -1

        config.suffix    = f'{config.optimizer}-method{config.method}'
        config.is_causal =True
        # assert config.method in [5,6,7,8,9,10,11,12,13]
        assert config.optimizer
        return config



    def __init__(self, config):
        super(GPT,self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        config.optimizer ='rmsprop'
        config.optimizer ='adamw'
        # config.optimizer ='adam'
        self.config = config
        pad = 20
        # config.block_size = config.block_size + pad
        config.g = 100
        nl = config.n_layer
        ne = config.n_embd
        # config.n_head =  1

        # ng = 2
        config.ng = 12
        ng = config.ng
        nh = config.n_head


        
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size + pad, config.n_embd),
            wpe2 = nn.Embedding(config.block_size + pad, config.n_embd),
            wke  = nn.Embedding(config.g, config.n_embd),
            wve  = nn.Embedding(config.g, config.n_embd),
            wge  = nn.Embedding(ng, config.n_embd),
            flayer  = nn.Linear((nh+1)*ne, ne*ng,bias=False),
            glayer  = nn.Linear((nl+1)*ne, ng, bias=False),
            wve2  = nn.Embedding(config.g, config.n_embd*ng),
            proj  = nn.Linear(config.n_embd, config.n_embd,bias=False),
            x2  = nn.Linear(ne,ne,bias=False),
            drop = nn.Dropout(config.dropout),
            
            # gate_attn = nn.Linear(ne, ne*(nh+1),bias=False),
            final_output = nn.Linear(ne*(nl), ne,bias=False),
            gate_attn = nn.Linear(ne*(ng+1), ng,bias=True),
            k_attn = nn.Linear(ne*ng, nh, bias=True),
            h = nn.ModuleList([ nn.ModuleDict(
                dict(
                k_attn = nn.Linear(ne, ne,bias=False),
                # k_attn_2 = nn.Linear(ne, ne,bias=False),
                # flayer = nn.Linear(ne*(ng+1),    ne,bias=False),
                # flayer2 = nn.Linear(ne*(3+1),    ne,bias=False),
                flayer2 = nn.Linear(ne*(2),    ne, bias=False),
                flayer3 = nn.Linear(ne*(1),    ne, bias=False),
                # flayer2b = nn.Linear(ne*(2),    ne,bias=False),
                # flayer4 = nn.Linear(ne*(2+1),    ne,bias=False),
                # o_gate =  nn.Linear(ne*(ng+1), 1, bias=True),
                )
                ) for _ in range(config.n_layer)
                ]),
            # h = nn.ModuleList([Block23(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))



    def forward(self, idx, targets=None):

        g  = self.config.g
        ng = self.config.ng
        e  = self.config.n_embd
        ne = self.config.n_embd

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        pad = 0

        # pos = torch.cat([pos[:pad],pos],0)
        # idx = torch.cat([idx[:,:pad]*0,idx],1)
        lreg = 0.
        lreg_factor = 0.0


        tok_emb   = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb   = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # pos_emb[:pad]= pos_emb_2[:pad]

        ### init internal embedding
        ### ng is the number of internal vectors
        int_emb   = self.transformer.wpe2(torch.zeros((b,t,ng),device=device).long() + pos[None,None,:ng])
        

        t = t + pad
        x = self.transformer.drop(tok_emb + pos_emb)
        # x = (tok_emb + pos_emb)
        
        
        att_mask = torch.ones((t,t),device=device).bool()
        ### (i,j)  0 for j>i 1 for j<=i  so i is ti, j is tp
        ### (tc,tp)
        # att_mask = torch.tril(att_mask, diagonal=-1)
        att_mask      = torch.tril(att_mask, diagonal=0)
        # att_mask    = torch.tril(att_mask, diagonal=0)
        att_mask[0,0] =True


        md    = self.transformer
        dx    = 0.
        nstep = 1

        xn = x
        xn = xn/(0.001 + xn.std(-1,keepdims=True))
        # xn = xn/(1. + xn.std(-1,keepdims=True))
        x = xn
        nl = self.config.n_layer
        nh = self.config.n_head

        def get_context(weight, xq, xn):
            '''
            ### for each round, getting nh neihgbors, and compute the next state
            '''
            # xk = torch.einsum('bte,btef->btf',xq,weight)
            xk = (xq@weight)
            xk = xk.reshape((b,t,e))
            xk = xk/(0.001+xk.std(-1,keepdims=True))
            att = torch.einsum('bte,bpke->bktp',xk, xn/ne**0.5)
            att = att.masked_fill(att_mask[None,None, :t,:t] == 0, float('-inf'))
            att = att.softmax(-1)

            ### (b,t,k)
            y    = torch.einsum('btp,bpe->bte',att,xn)
            return y,att

        dx = 0.

        block0 = self.transformer.h[0]
        # block1 = self.transformer.h[0]
        dx1 = xn
        dx2 = xn
        # dx2 = dx2 / (0.001 + dx2.std(-1,keepdims=True))
        # nh  = self.config.n_head
        dx0h = torch.zeros((b,t,e,nl),device=device)
        outh = torch.zeros((b,t,1,nl),device=device)
        int_state = int_emb.clone()
        for ii in range(1):

            for layer_i in range(nl):                
                block = self.transformer.h[layer_i]
                #### getting the two attended vector

                ### (b,t,e)
                ### getting input for both input and internal                
                wk    = block.k_attn.weight.T                
                xq    = dx1.matmul(wk)                
                ### (b,t,k)                
                # engk = torch.einsum('bte,btke->btk',xq , int_state) 
                ### (b,t,p)
                engp   = xq @ xn.transpose(1,2) 
                # engk, engp
                engp   = engp.masked_fill(att_mask[None,:t,:t] == 0, float('-inf'))
                att    = (torch.cat([engp],dim=2)  / (ne ** 0.5) ).softmax(-1)
                # yinput = torch.einsum('btk,btke->bte', att[:,:,:ng], int_state) + 
                yinput = att[:,:,] @ xn


                # yinput,  att  = get_context(wk, dx1, torch.cat([ int_state, xn],dim=1))

                ### (b,t,e)
                ### (b,t,k)
                # yinput2, atto = get_context(wk, dx1, int_state)

                ctx=  torch.cat([yinput,dx1],dim=2)
                dx1 = block.flayer2(ctx) + dx1
                # yout = yout.unsqueeze(-2) + int_emb
                # yout = yout.unsqueeze(-2) 
                # # yout = yout.unsqueeze(-2) #+ int_emb
                # # dx1  = block0.flayer3(ctx)
                # dx1  = block0.flayer3(torch.cat([ctx,y3],dim=-1))

                # atto = atto.unsqueeze(-1)
                # # int_state = atto*yout + (1- atto) *int_state
                # int_state = atto*yout + int_state + atto2.unsqueeze(-1) * block0.flayer4(ctx).unsqueeze(-2)
                # # int_state = int_state + int_emb
                # int_state = int_state / (0.001 + int_state.std(-1,keepdims=True))
                dx1       = dx1/(0.001 + dx1.std(-1,keepdims=True))              
        x = dx1

        # x = int_state[:,:,-1,:]
        # x0 = x
        x = self.transformer.ln_f(x)        
        
        # lreg += self.lm_head.weight.square().sum()

        x = x[:,pad:]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.training:
                loss = loss + lreg_factor*lreg
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss 

'''
GPT01

iter 1740: loss 1.1673, time 40.94ms, mfu 9.46%
step 1750: train loss 1.1054, val loss 1.4680
saving checkpoint to out-shakespeare-char


CM01
iter 730: loss 0.9343, time 88.90ms, mfu 0.47%
iter 740: loss 0.8686, time 88.92ms, mfu 0.47%
step 750: train loss 0.8607, val loss 0.8843


CM01
step 1750: train loss 4.1811, val loss 5.0537

step 1750: train loss 4.2450, val loss 5.0785
step 2000: train loss 4.1566, val loss 5.0758
step 2750: train loss 4.0060, val loss 5.0832

And do not in the like a man now!

CAPULET:
Thou hast you, be said, Aufidius,
Where it is known to you; for our brows;

GREMIO: I know not.

KING HEN ELIZEL:
I'll not your grace my lord;
To greetings, let's the boar!

LADY CAPULET:
He's eyes, my Lord of the Duke of love
And, for a mystery remained
Though we have done a dozen friends;
I say, I am here in his life,
The people,
And with my mother! Come hither;
One of all the field?

DUKE VINCENTIO:
In fortune,
Not you, if he be so,
Though you have you, dole, and here to the world:
The other's wrong before the Earl of their fingers,
With us?

KING RICHARD:
They have him or what they were so much I
---------------

Thou shalt be truths.
One, and
And yet not well-mast, being the hollow queen,
And I the nurse of those bloody eyes.

Provost:

'er I will be the palace of a mile on the year.

And there be full of love;
By our noble uncle, from the deed is my son, go with you,

'''

'''
CCM50,4.63


RICHARD:
Look, by my lady, and am I, my lord;
Which I have made thee doubt, if I follow
---------------

RICHARD:
My lord, my lord, at my sweet soul, and good lord,
That we, thus I know, I could make thee sleep!

GLOUCESTER:
Ah, they are my daughter, that hast thou blest,
But thou hast not leave to my heart in thee.

KING EDWARD IV:
I speak or well I could not do my life.

Second Citizen:
Go, be we must not live, and say the king.

First Citizen:
Ay, good for more. Go, God's mother: come.

Second Citizen:
Well, welcome!

AUTOLYCUS:
Shepherd:
Ay, sir; I'll hear it.

MENENIUS:
Now, good sir, sir, good sir;
And you have: if you do be spoke of one.

First Citizen:
They are our general.

COMINIUS:

'''

GPT = GPT
# GPT = CCM01
# # GPT = CCM02
# # GPT = CCM03
# # GPT = CCM04
# # GPT = CCM05
# # 
# GPT = CCM06

# GPT=GPT
