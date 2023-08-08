from transformers import AutoTokenizer,AutoModel,TrainingArguments,AutoConfig
import torch
import torch.nn as nn
from peft import LoraConfig,TaskType,get_peft_model
from torchkeras import KerasModel
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_preprocess import load_json_data,get_dataloader
from steprunner import StepRunner
from Config import Config

conf =Config()
train_json_path ="xxx"
test_json_file ="xxx"
llm_file_path ="yyy"


##data_preprocess
train_dataset =load_json_data(train_json_path)
test_dataset =load_json_data(test_json_file)
train_loader,test_loader =get_dataloader(train_dataset,test_dataset)

print("finish data preprocess")
##config model
config =AutoConfig.from_pretrained(llm_file_path,trust_remote_code=True)
model =AutoModel.from_pretrained(llm_file_path,trust_remote_code=True,config=config)

model.supports_gradient_checkpointing =True #开启gradient_ckpt 节约现存
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.config.use_cache =False #训练不用kv缓存 推理可以用

lora_config =LoraConfig(task_type=TaskType.CAUSAL_LM,inference_mode=False,r=4,lora_alpha=32,lora_dropout=0.05)

model =model.half().cuda()
model =get_peft_model(model,lora_config)
model.print_trainable_parameters()    #显示可训练参数的量
print("finish config lora")

##config training
#rewrite StepRunner/save_ckpt/load_ckpt
KerasModel.StepRunner =StepRunner
KerasModel.save_ckpt =StepRunner.save_ckpt
KerasModel.load_ckpt =StepRunner.load_ckpt

lr_scheduler =CosineAnnealingLR(torch.optim.AdamW(model.parameters(),lr=conf.lr),T_max=10)

keras_model =KerasModel(model,loss_fn=None,optimizer=torch.optim.AdamW(model.parameters(),lr=conf.lr),
                        lr_scheduler=lr_scheduler)

ckpt_path ="zzz"
print("finish config keras model")
keras_model.fit(train_data=train_loader,
                val_data=test_loader,
                epochs=conf.epochs,
                monitor='train_loss',
                mode='mean',
                ckpt_path=ckpt_path,
                mixed_precision='fp16')
print("finish lora training!!!")



