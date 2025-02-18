from transformers import  AutoTokenizer,AutoModelForCausalLM

final_save_path="./final_models"
model = AutoModelForCausalLM.from_pretrained(final_save_path)
tokenizer = AutoTokenizer.from_pretrained(final_save_path)

print("构建推理pipline")
from transformers import pipeline

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

prompt = "tell me some sing skills"

generated_text = pipe(prompt,max_length=512,num_return_sequences=1)


print("开始回答-------",generated_text[0]["generated_text"])