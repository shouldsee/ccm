from transformers import AutoModel,AutoTokenizer,LlamaForCausalLM,AutoModelForCausalLM

# pth = '/root/autodl-tmp/LLaMA-Factory/models/llama-3-8b-Instruct-chinese_v2/'
pth = '/root/ccm/models/MiniCPM-1B-sft-bf16/'
model     = AutoModelForCausalLM.from_pretrained(pth,local_files_only=True,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(pth,local_files_only=True,trust_remote_code=True)


# from transformers import LlamaForCausalLM
x = tokenizer(['请续写如下小说：'],return_tensors='pt')
# x['labels']    = x['input_ids'][:,1:]
# x['input_ids'] = x['input_ids'][:,:-1]
# x['attention_mask'] = x['attention_mask'][:,1:]
# # x.pop('attention_mask')
# y = model(**x, output_hidden_states=True)
breakpoint()
