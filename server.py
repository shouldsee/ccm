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


def load_model(out_dir, model_name):

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
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    # ModelEncoding()
    return Env(model,encode,decode)

class Env(object):
    def __init__(self,model,encode,decode):
        self.model = model
        self.encode = encode
        self.decode = decode
        pass


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
fn = 'out-novel-char_LGT001_29fd8aaa7d6a706d3fe1fe4e    daa59f0d'
fn = 'out-novel-char_LGT001D_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT001E_29fd8aaa7d6a706d3fe1fe4edaa59f0d'
# fn = 'out-novel-char_LGT001_5f42cea4d6cf5b579c5219fb2b448174'

'''

却说王濬班师，迁吴主皓赴洛阳面君。皓登殿稽首以见晋帝。帝赐坐曰：“朕设此座以待卿久矣。”皓对曰：“臣于南方亦设此座以待陛下。”帝大笑。贾充问皓曰：“闻君在南方，每凿人眼目，剥人面皮，此何等刑耶？”皓曰：“人臣弑君及奸回不忠者，则加此刑耳。”充默然甚愧。帝封皓为归命侯，子孙封中郎，随降宰辅皆封列侯。丞相张悌阵亡，封其子孙。封王濬为辅国大将军，其馀各加封赏。

'''
mname = fn.split('_')[1]
model = load_model(fn,mname)
print(f'[loaded_model]{fn}')
config =dict(server_port=6006,server_name="0.0.0.0")



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
    add_analyse_interface(model)
    
demo.launch(**config)

# demo.launch(server_port=6006,server_name="0.0.0.0")
