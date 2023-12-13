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

class GPT(nn.Module):

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
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

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



gpt01 2-layer
step 1000: train loss 4.0415, val loss 4.8985
step 1250: train loss 3.9042, val loss 4.8685
step 1500: train loss 3.7759, val loss 4.8632



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
            lin_output   = nn.Linear(ne, ne,bias=config.bias),
            lin_output_2   = nn.Linear(ne, ne,bias=config.bias),
            lin_output_3   = nn.Linear(ne, ne,bias=config.bias),
            lin_internal = nn.Linear(nei, ne,bias=config.bias),
            lin_internal_output = nn.Linear(nei, ne,bias=config.bias),
            lin_transit = nn.Linear(ne,nei,bias=config.bias),
            lin_transit_2 = nn.Linear(ne,ne,bias=config.bias),
            log_prior   = nn.Linear(10,2,bias=config.bias),
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


        cpt       = (torch.zeros((b,t,e),device=device)+ x)
        cpt2      = (torch.zeros((b,t,e),device=device)+ x)

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

        for i in range(4):
            xpred = get_xpred(cpt)
            cpred = get_cpred(x,cpt)

            ### (b,tp,nh,tc)
            lp = 0.
            #### gradient to predict the token
            lp += torch.einsum('bpe,bce ->bcp', xpred,   x)
            lp += torch.einsum('bpec,bce->bcp', cpred, cpt.matmul(md.lin_internal.weight))

            #### gradient to predict the dependency between context
            ### (b,tc,tp)
            lp = lp.masked_fill(att_mask[None,:,:]==0,float('-inf'))

            ### (b,tc,tp)
            att  = lp.softmax(-1)
            dcpt = torch.einsum('bcp,bpec->bce', att, cpred).matmul(md.lin_internal.weight.T)
            cpt  = cpt + 0.1*dcpt


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

            ### (b,tc,tp)
            att  = lp.softmax(-1)
            dcpt2 = torch.einsum('bcp,bpec->bce', att, cpt2_pred)
            cpt2  = cpt2 + 0.1*dcpt2

        
        x     = cpt2.matmul(md.lin_output_3.weight)
        # print(x.shape)

        x     = self.transformer.ln_f(x)            

        ###step 1750: train loss 4.5279, val loss 5.3416

        logits = self.lm_head(x).log_softmax(-1)
        # logits = logits.logsumexp(2)


        ### (b,t)
        if targets is not None:
            loss  = F.cross_entropy(logits.transpose(1,2), targets, ignore_index=-1,reduction='mean')
        else:
            loss  = None
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

'''

GPT = CCM01
# GPT = CCM02
# GPT = CCM03
# GPT = CCM04
# GPT = CCM05
# 
GPT = CCM06

# GPT=GPT
