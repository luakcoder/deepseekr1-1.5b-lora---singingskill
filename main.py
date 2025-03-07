import json
from data_pre import samples
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM , AutoModelForMaskedLM,AutoModelForCausalLM
import torch
model_name = "deepseekr1-1.5b"
model_path = r"E:\llm\r1-1.5b"#模型路径(换成自己存放模型的路径)
tokenizer = AutoTokenizer.from_pretrained(model_path)
#1.
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     trust_remote_code=True  # 必须添加此参数
# )
# print("模型加载成功")
#2.
# with open(r"datasets.jsonl", "w",encoding="utf-8") as f:
#     for s in samples:
#         json_line = json.dumps(s, ensure_ascii=False)
#         f.write(json_line+"\n")
#     else:
#         print("prepare data finished")

#3.
from datasets import load_dataset
dataset = load_dataset(
    "json",
    data_files={"train":"datasets.jsonl"},  # 直接指定文件路径
    split="train"
)
print(len(dataset))
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]



#4
def tokenize_function(many_samples):
    texts=[f"{prompt}\n{completion}" for prompt,completion in
           zip(many_samples["prompt"],many_samples["completion"])]
    tokens = tokenizer(texts, truncation=True, max_length=512,padding="max_length")
    tokens["labels"]=tokens["input_ids"].copy()
    return tokens
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
# tokenized_train_dataset.set_format(type="torch")  # 新增

tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
# tokenized_eval_dataset.set_format(type="torch")  # 新增
#print(tokenized_train_dataset[0])

#5 量化设置
from transformers import BitsAndBytesConfig
#,device_map="auto"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_path, config=quantization_config).to("cuda")
#print(model.hf_device_map)  # 查看各层分配到的设备
from peft import get_peft_model,LoraConfig,TaskType

lora_config = LoraConfig(
    r=8,lora_alpha=16,lora_dropout=0.05,task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model,lora_config)
model.print_trainable_parameters()
#print("lora  set finshed")

#7.set train
from transformers import TrainingArguments,Trainer

training_args = TrainingArguments(
    output_dir="./finetuned_models",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=10,
    learning_rate=3e-5,
    logging_dir="./logs",
    run_name="deepseek-r1-distill-finetune"
)

print("训练参数完毕")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)
print("开始训练")

trainer.train()

print("训练完成")

#8.save   lora model

save_path="./saved_models"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("lora has saved")
final_save_path="./final_models"

from peft import PeftModel
best_model = AutoModelForCausalLM.from_pretrained(model_path)
model = PeftModel.from_pretrained(best_model,save_path)
model = model.merge_and_unload()


model.save_pretrained(final_save_path)

tokenizer.save_pretrained(final_save_path)

print("全量模型已保存")
