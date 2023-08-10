from transformers import AutoTokenizer,AutoModel,TrainingArguments,AutoConfig
import torch
import torch.nn as nn
from peft import LoraConfig,TaskType,get_peft_model
from torchkeras import KerasModel
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_preprocess import load_json_data,get_dataloader
from steprunner import StepRunner
import argparse


def main():
    parser =argparse.ArgumentParser()
    parser.add_argument("--train_data_path",default=None,type=str,required=True)
    parser.add_argument("--test_data_path",default=None,type=str,required=False)
    parser.add_argument("--base_llm_path",default=None,type=str,required=True)
    parser.add_argument("--output_dir",default=None,type=str,required=True)

    parser.add_argument("--sft_mode",default=None,type=str,required=True)
    parser.add_argument("--lora_r",default=4,type=int,required=False)
    parser.add_argument("--lora_alpha", default=32, type=int, required=False)
    parser.add_argument("--lora_drouout", default=0.05, type=float, required=False)

    parser.add_argument("--lr", default=5e-4, type=float, required=False)
    parser.add_argument("--batch_size", default=2, type=float, required=False)
    parser.add_argument("--max_seq_len",default=2048,type=int,required=False)
    parser.add_argument("--epochs", default=3, type=int, required=False)


    args =parser.parse_args()
    print(args)


    ##data_preprocess
    train_dataset =load_json_data(args.train_data_path)
    test_dataset =load_json_data(args.train_data_path)
    train_loader,test_loader =get_dataloader(train_dataset,test_dataset,args.base_llm_path,args.batch_size,args.max_seq_len)

    print("finish data preprocess")
    ##config model
    config =AutoConfig.from_pretrained(args.base_llm_path,trust_remote_code=True)
    model =AutoModel.from_pretrained(args.base_llm_path,trust_remote_code=True,config=config)

    model.supports_gradient_checkpointing =True #开启gradient_ckpt 节约现存
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model.config.use_cache =False #训练不用kv缓存 推理可以用

    lora_config =LoraConfig(task_type=TaskType.CAUSAL_LM,inference_mode=False,r=args.lora_r,
                            lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout)

    model =model.half().cuda()
    model =get_peft_model(model,lora_config)
    model.print_trainable_parameters()    #显示可训练参数的量
    print("finish config lora")

    ##config training
    #rewrite StepRunner/save_ckpt/load_ckpt
    KerasModel.StepRunner =StepRunner
    KerasModel.save_ckpt =StepRunner.save_ckpt
    KerasModel.load_ckpt =StepRunner.load_ckpt

    lr_scheduler =CosineAnnealingLR(torch.optim.AdamW(model.parameters(),lr=args.lr),T_max=10)

    keras_model =KerasModel(model,loss_fn=None,optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr),
                            lr_scheduler=lr_scheduler)

    print("finish config keras model")
    keras_model.fit(train_data=train_loader,
                    val_data=test_loader,
                    epochs=args.epochs,
                    monitor='val_loss',
                    mode='min',
                    ckpt_path=args.output_dir,
                    mixed_precision='fp16')
    print("finish lora training!!!")

if __name__ =="__main__":
    main()

