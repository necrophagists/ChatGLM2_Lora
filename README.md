# ChatGLM2_Lora_Simplified

This repo was  a simple  way to implement Lora to fine-tuning ChatGLM2.

这个项目是用LORA微调chatglm2的简单实现。

##2023/08/09
数据格式
{
 {"input":"AAAAA",
  "output":"BBBBB"
 },
  {"input":"CCCCC",
   "output":"DDDDD"
  }
}

# 使用方式
## lora:
  python3 train_lora.py 
--train_data_path  "xxx"
--test_data_path   "xxx"
--base_llm_path  "xxx"
--output_dir "xxx"
--sft_mode "xxx"
--lora_r   4
--lr  5e-4
--batch_size 2
--max_seq_len 2048
--epochs 3
## 合并权重
python3 merge_lora.py 
--base_model_path  "xxx"
--lora_model_path "xxx"
--merged_model_path "xxx"

# To Do List

- 奖励模型
- RLHF
- P-tuning
- LLAMA2
   



