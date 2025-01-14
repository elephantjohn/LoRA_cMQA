# LoRA_cMQA
## 一、训练adapter过程
### 环境配置
python3.11
requirements.txt
```python 
accelerate==0.21.0
annotated-types==0.7.0
Brotli==1.0.9
certifi==2024.12.14
charset-normalizer==3.3.2
deepspeed==0.16.2
einops==0.8.0
filelock==3.13.1
flash-attn==2.7.3
fsspec==2024.12.0
gmpy2==2.1.2
hjson==3.1.0
huggingface-hub==0.27.1
idna==3.10
Jinja2==3.1.4
MarkupSafe==3.0.2
mkl_fft==1.3.11
mkl_random==1.2.8
mkl-service==2.4.0
mpmath==1.3.0
msgpack==1.1.0
networkx==3.3
ninja==1.11.1.3
numpy==1.26.4
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-ml-py==12.560.30
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
packaging==24.2
pandas==2.2.3
peft==0.6.2
pillow==11.0.0
pip==24.2
psutil==6.1.1
py-cpuinfo==9.0.0
pydantic==2.10.5
pydantic_core==2.27.2
PySocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2024.2
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
safetensors==0.5.2
scipy==1.15.1
setuptools==75.1.0
six==1.17.0
sympy==1.13.3
tiktoken==0.8.0
tokenizers==0.15.2
torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1
tqdm==4.67.1
transformers==4.37.2
transformers-stream-generator==0.0.4
triton==3.1.0
typing_extensions==4.12.2
tzdata==2024.2
urllib3==2.2.3
wheel==0.44.0
```
### 训练参数
```python
export CUDA_VISIBLE_DEVICES=0

lora_output_qwen=/root/work/lora/lora_output
python ../finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --bf16 True \
  --output_dir $lora_output_qwen \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 10 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora

# If you use fp16 instead of bf16, you should use deepspeed
# --fp16 True --deepspeed finetune/ds_config_zero2.json
```
finetune.py来源于QWen库

### 数据集
医疗问答数据集，Github地址：https://github.com/wjjingtian/cMQA 

### 关键优化
在3090 24G上用finetune_lora_single_gpu.sh无法运行，GPU Out Of Memory； 需要修改model里的config.json里的"num_hidden_layers": 32 改为16；
在  --per_device_train_batch_size 2  --per_device_eval_batch_size 1 的情况下，在使用bf16且没有deepspeed的情况下，用时8小时16分；
在  --per_device_train_batch_size 2  --per_device_eval_batch_size 1 的情况下，在使用fp16且用了deepspeed的情况下，跑不起来；


### 训练过程
<img width="757" alt="截屏2025-01-13 22 16 18" src="https://github.com/user-attachments/assets/5d593f1f-e045-4a75-b91d-17b36aaa8ece" />

### GPU占用
<img width="1008" alt="截屏2025-01-13 22 16 42" src="https://github.com/user-attachments/assets/c2da1851-50cd-4105-b9e8-6aad984c863c" />

### adapter结构
```json
.
├── adapter_config.json
├── adapter_model.safetensors
├── checkpoint-1000
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── qwen.tiktoken
│   ├── README.md
│   ├── rng_state.pth
│   ├── scheduler.pt
│   ├── special_tokens_map.json
│   ├── tokenization_qwen.py
│   ├── tokenizer_config.json
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-2000
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── qwen.tiktoken
│   ├── README.md
│   ├── rng_state.pth
│   ├── scheduler.pt
│   ├── special_tokens_map.json
│   ├── tokenization_qwen.py
│   ├── tokenizer_config.json
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-3000
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── qwen.tiktoken
│   ├── README.md
│   ├── rng_state.pth
│   ├── scheduler.pt
│   ├── special_tokens_map.json
│   ├── tokenization_qwen.py
│   ├── tokenizer_config.json
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-4000
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── qwen.tiktoken
│   ├── README.md
│   ├── rng_state.pth
│   ├── scheduler.pt
│   ├── special_tokens_map.json
│   ├── tokenization_qwen.py
│   ├── tokenizer_config.json
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-5000
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── qwen.tiktoken
│   ├── README.md
│   ├── rng_state.pth
│   ├── scheduler.pt
│   ├── special_tokens_map.json
│   ├── tokenization_qwen.py
│   ├── tokenizer_config.json
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-6000
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── qwen.tiktoken
│   ├── README.md
│   ├── rng_state.pth
│   ├── scheduler.pt
│   ├── special_tokens_map.json
│   ├── tokenization_qwen.py
│   ├── tokenizer_config.json
│   ├── trainer_state.json
│   └── training_args.bin
├── qwen.tiktoken
├── README.md
├── special_tokens_map.json
├── tokenization_qwen.py
├── tokenizer_config.json
├── trainer_state.json
└── training_args.bin
```
## 二、使用adapter过程
checkpoint目录下有adapter_config.json里配置了基础模型的路径；
运行load_lora_adapter.py脚本
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# lora微调过的模型, 注意使用peft的AutoPeftModelForCausalLM加载模型
model = AutoPeftModelForCausalLM.from_pretrained("/root/work/lora/lora_output/checkpoint-6000", device_map="auto",trust_remote_code=True).eval()

# 对比未经lora微调的基础模型，注意使用transformer的AutoModelForCausalLM加载
#model = AutoModelForCausalLM.from_pretrained("/root/work/qwen/Qwen/models/", device_map="auto",trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained("/root/work/qwen/Qwen/models/", trust_remote_code=True)

#response, history = model.chat(tokenizer, "我家宝宝六个月，吃奶、睡觉常常满头冒汗，以前是不出汗的。请问这是不是病，如何医治？", history=None)
response, history = model.chat(tokenizer, "手毛，脚毛都很多，很密，都不知道怎么办，有什么永远脱毛方法，不想激光脱毛？", history=None)

print(response)
```
经过LoRA微调后的Qwen模型，还能够比较准确的保持其原有的通用知识生成能力，除此之外，LoRA的优势更在于其推理阶段的优势。

## 三、总结
lora在推理阶段是直接使用训练好的A、B低秩矩阵去替换原预训练模型的对应参数，就可以避免因增加网络的深度所带来的推理延时和额外的计算量。所以特别适用于对推理速度和模型性能都有较高要求的应用场景。

LoRA（Low-Rank Adaptation）是一种在大模型上进行参数高效微调的方法。它通过在模型的某些层引入低秩矩阵来学习微调任务的增量权重，并将这些增量权重与原始模型的权重相加/合并，从而完成“微调”。当我们拿微调完的 LoRA 模型进行推理时，相比直接用基础模型（base model）推理，通常会带来以下好处：

1. **更好的任务表现**  
   - **个性化/领域化性能提升**：LoRA 只需在相对小规模的低秩参数中“学习”新任务的特征，就能显著提高模型在特定任务或领域上的表现。基础模型往往是通用性较强，但对特定任务或领域的适应性不足，难以达到理想的效果。  
   - **微调效果可快速验证**：由于 LoRA 的训练开销较小，易于快速迭代，可以更快验证微调思路，得到更好的推理结果。

2. **内存与存储开销较小**  
   - **只需存储少量增量参数**：相比传统的“全参数微调”，LoRA 在推理时只需要加载基础模型 + 少量 LoRA 参数（通常是几百万甚至更小的参数量），无需像全量微调那样生成一个体量和基座模型差不多大小的新模型。这在多任务、多领域应用时尤其明显，一套基础模型可以搭配多个 LoRA 微调模块，大幅减少存储与管理成本。  
   - **易于部署和热插拔**：由于 LoRA 参数规模较小，“加载/卸载”不同领域的微调模块十分便捷，在推理框架中能相对灵活地实现不同场景的切换。

3. **推理速度几乎不变或变化很小**  
   - LoRA 参数的合并通常只在网络部分矩阵上进行低秩运算，对推理速度影响较小；在某些高效实现（如 FlashAttention、Fused Kernel 等）里，额外的计算成本可以进一步优化。整体上，相比全量微调后体量变大的模型，LoRA 的推理一般能保持与原有基础模型大致相当的速度。

4. **易于多任务、多语言或跨领域扩展**  
   - 在需要适配不同任务（或多语言、多领域）时，可以为每个场景分别训练一个小规模的 LoRA 适配器。在推理时，只需组合对应的 LoRA 权重即可，让同一个基础模型发挥多用途，而无需维护多个完整模型。

简而言之，使用 LoRA 进行微调并在推理时加载 LoRA 权重，可以在保持基础模型规模与推理速度大致不变的前提下，针对特定任务或领域获得更好的推理效果，并且大幅降低额外的存储与部署成本。这也是 LoRA 在大模型应用中越来越受青睐的主要原因。

