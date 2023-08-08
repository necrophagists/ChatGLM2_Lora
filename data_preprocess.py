from transformers import AutoTokenizer
import torch
import json
import datasets
from datasets import load_dataset
from typing import Dict,List

from Config import Config
conf =Config()


llm_file_path ="xxx"
tokenizer =AutoTokenizer.from_pretrained(llm_file_path,trust_remote_code=True)
max_seq_length =2048    #max length of input or output   for padding
skip_over_length =True  #skip the string that over length


def str2ids(sample:Dict) ->Dict:
    context =sample['input']
    output =sample['output']

    context_ids =tokenizer.encode(
        context,
        max_length =max_seq_length,
        truncation =True   # keep maxium length <=max_Seq_length
    )

    output_ids =tokenizer.encode(
        output,
        max_length =max_seq_length,
        truncation =True   # keep maxium length <=max_Seq_length
    )

    input_ids =context_ids+output_ids+[tokenizer.eos_token_id]    #  sentence =input +output+eos

    return {"input_ids":input_ids,"context_len":len(context_ids),"output_len":len(output_ids)}

def collate_fn(batch_data:List):
    batch_seq_len =[len(data["input_ids"]) for data in batch_data]
    max_batch_seq_len =max(batch_seq_len)

    input_ids,output_ids=[],[]
    #按长度排序
    for seq_len,data in sorted(zip(batch_seq_len,batch_data),key =lambda x:-x[0]): #降序
        ids =data["input_ids"]
        input_len =data["context_len"]

        outputs =([-100]*(input_len-1)+ids[input_len-1:])+[-100]*(max_batch_seq_len-input_len)
        ids =ids+[tokenizer.pad_token_id]*(max_batch_seq_len -seq_len)

        input_ids.append(torch.LongTensor(ids))
        output_ids.append(torch.LongTensor(outputs))  #???
    #这里的批处理是这样的:
    #(1) 按长度降序排列  (2)input其实是context+output+eos拼成的一个序列,并且需要做right padding。
    #(3) label和input等长,无效部分(right padding、以及context除最后一个token之外的位置)均用-100来填充
    #-100在 hf里是无效token的默认值，在计算loss的时候会忽略掉这部分token。

    #ps:decoder only的模型一般不需要bos token 直接根据当前的token做生成就可以了 不需要起始符。
    #并且这种把context和output拼接作为输入序列的方式可以看做"强制教学"方式的一种,
    #然后对于label label和input登场,

    #为什么label中用-100,而input中pad还是0? 因为对于input pad位置是需要参与计算的,只不过后面mask掉就行了。
    #对于label -100的部分是不用去算概率的 在计算交叉熵时会跳过-100的位置。

    input_ids =torch.stack(input_ids)
    outputs =torch.stack(output_ids)

    return {
        "input_ids":input_ids,
        "labels":outputs,
    }
def load_json_data(file_path:str):
    dataset=[]
    with open(file_path) as f:
        data =json.load(f)

    for x in data:
        dataset.append({"context":x['input'],"output":x['output']})

    return datasets.Dataset.from_list(dataset)

def get_dataloader(train_dataset,test_dataset):
    train_ds =train_dataset.map(str2ids).select_columns(['input_ids','context_len','output_len'])
    if skip_over_length ==True:      #过滤超过max_length的句子
        train_ds =train_ds.filter(
            lambda x:x['context_len']<max_seq_length and x['output_len']<max_seq_length
        )
    train_loader =torch.utils.data.DataLoader(train_ds,batch_size =conf.bs,num_workers=2,pin_memory=True,shuffle=True,collate_fn=collate_fn)

    test_ds =test_dataset.map(str2ids).select_columns(['input_ids','context_len','output_len'])
    if skip_over_length ==True:      #过滤超过max_length的句子
        test_ds =test_ds.filter(
            lambda x:x['context_len']<max_seq_length and x['output_len']<max_seq_length
        )
    test_loader =torch.utils.data.DataLoader(test_ds,batch_size =conf.bs,num_workers=2,pin_memory=True,shuffle=True,collate_fn=collate_fn)
    return train_loader,test_loader