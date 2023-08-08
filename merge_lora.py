import torch
from peft import PeftModel,PeftConfig
from transformers import AutoModel,AutoTokenizer,AutoConfig

base_model_path ="xxx"
lora_model_path ="yyy"

merge_model_dir ="zzz"

print("loading lora model")

base_model =AutoModel.from_pretrained(
    base_model_path,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype =torch.float16
)
tokenizer =AutoTokenizer.from_pretrained(
    base_model_path,trust_remote_code=True
)

base_model_vocab_size =base_model.get_input_embeddings().weight.size(0)

print(base_model_vocab_size,len(tokenizer))
if base_model_vocab_size != len(tokenizer):
    base_model.resize_token_embeddings(len(tokenizer))
    ##！！！！！！！这里可能需要修改chatglm2 源码 看readme

lora_model =PeftModel.from_pretrained(
    base_model,
    lora_model_path,
    device_map='auto',
    torch_dtype =torch.float16
)
lora_model.eval()
print('merging...')
base_model =lora_model.merge_and_unload()
print('saving...')
tokenizer.save_pretrained(merge_model_dir)
base_model.save_pretrained(merge_model_dir)
print("done")