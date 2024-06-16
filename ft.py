import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

inputs = tokenizer('''

范闲没有发现，虽然他知道当时的范思辙一直在做什么准备，但是太子那里已经很困难了。

　　第七十四章 青州军

　　“我是谁？”

　　范闲皱眉说道：“这是如今北齐''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)