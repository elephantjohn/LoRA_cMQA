# LoRA_cMQA
### 环境配置

### 关键优化
在3090 24G上用finetune_lora_single_gpu.sh无法运行，GPU Out Of Memory； 需要修改model里的config.json里的"num_hidden_layers": 32 改为16；

### 训练过程
<img width="757" alt="截屏2025-01-13 22 16 18" src="https://github.com/user-attachments/assets/5d593f1f-e045-4a75-b91d-17b36aaa8ece" />

### GPU占用
<img width="1008" alt="截屏2025-01-13 22 16 42" src="https://github.com/user-attachments/assets/c2da1851-50cd-4105-b9e8-6aad984c863c" />
