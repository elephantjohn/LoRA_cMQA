from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# lora微调过的模型
model = AutoPeftModelForCausalLM.from_pretrained("/root/work/lora/lora_output/checkpoint-6000", device_map="auto",trust_remote_code=True).eval()

# 对比未经lora微调的基础模型
#model = AutoModelForCausalLM.from_pretrained("/root/work/qwen/Qwen/models/", device_map="auto",trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained("/root/work/qwen/Qwen/models/", trust_remote_code=True)


#response, history = model.chat(tokenizer, "我家宝宝六个月，吃奶、睡觉常常满头冒汗，以前是不出汗的。请问这是不是病，如何医治？", history=None)
response, history = model.chat(tokenizer, "手毛，脚毛都很多，很密，都不知道怎么办，有什么永远脱毛方法，不想激光脱毛？", history=None)

print(response)
