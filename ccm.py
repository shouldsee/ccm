
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