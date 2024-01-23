# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-novel-word'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'novel'
wandb_run_name = 'novel'

dataset = 'novel'
# gradient_accumulation_steps = 1
# gradient_accumulation_steps = 6
gradient_accumulation_steps = 12
batch_size = 12
# block_size = 256 # context of up to 256 previous characters
# block_size = 256 # context of up to 256 previous characters
block_size = 64 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
# n_embd = 182
dropout = 0.2

# learning_rate = 1e-3 # with baby networks can afford to go a bit higher
learning_rate = 1e-4 # with baby networks can afford to go a bit higher

# max_iters = 5000
# lr_decay_iters = 5000 # make equal to max_iters usually

max_iters = 500000
lr_decay_iters = 500000 # make equal to max_iters usually

min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
# init_from ='resume'


'''
python3 -u sample.py config/train_novel.py --out_dir=out-novel-word_CCM232_c85667cd6eadbcf772d23fa97827c506 --model=CCM232
python3 -u sample.py config/train_novel.py --out_dir=out-novel-word_CCM251_4bfeb7b3e1b9c46a2542bb9406173e22 --model=CCM251
python3 -u sample.py config/train_novel.py --out_dir=out-novel-word_CCM253_01b622029bb00e0dc9206cae3b0ab559 --model=CCM253 
python3 -u sample.py config/train_novel.py --out_dir=out-novel-word_CCM276_01b622029bb00e0dc9206cae3b0ab559 --model=CCM276
python3 -u sample.py config/train_novel.py --out_dir=out-novel-word_CCM278_a97bfe077b316033e3604831d5ad371e --model=CCM278
python3 -u sample.py config/train_novel.py --out_dir=out-novel-word_CCM305_52efa1def80492f81622f536ac701226 --model=CCM305


out-novel-word_CCM276_01b622029bb00e0dc9206cae3b0ab559

'''
#### python3 -u sample.py config/train_novel.py --out_dir=out-novel-word_CCM232_c85667cd6eadbcf772d23fa97827c506 --model=CCM232