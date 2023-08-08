# ChatGLM2_Lora_Simplified
This repo was  a simple  way to implement Lora to fine-tuning ChatGLM2.这个项目是用LORA微调chatglm2的简单实现。

##2023/08/08
需要修改chatglm2的modeling_chatglm.py
第768行加上一个函数:
def set_input_embddings(self,values):
   self.embedding.word_embeddings =values

