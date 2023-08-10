import torch
from peft import PeftModel,PeftConfig
from transformers import AutoModel,AutoTokenizer,AutoConfig
import argparse

def main():
    parser =argparse.ArgumentParser()
    parser.add_argument("--base_model_path",default=None,type=str,required=True)
    parser.add_argument("--lora_model_path", default=None, type=str, required=True)
    parser.add_argument("--merged_model_path", default=None, type=str, required=True)

    args =parser.parse_args()
    print(args)

    print("loading lora model")

    base_model =AutoModel.from_pretrained(
        args.base_model_path,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype =torch.float16
    )
    tokenizer =AutoTokenizer.from_pretrained(
        args.base_model_path,trust_remote_code=True
    )


    lora_model =PeftModel.from_pretrained(
        base_model,
        args.lora_model_path,
        device_map='auto',
        torch_dtype =torch.float16
    )
    lora_model.eval()
    print('merging...')
    base_model =lora_model.merge_and_unload()
    print('saving...')
    tokenizer.save_pretrained(args.merged_model_path)
    base_model.save_pretrained(args.merged_model_path)
    print("done")

if __name__ =='__main__':
    main()