# LoRA_cMQA
### 环境配置
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

### 关键优化
在3090 24G上用finetune_lora_single_gpu.sh无法运行，GPU Out Of Memory； 需要修改model里的config.json里的"num_hidden_layers": 32 改为16；

### 训练过程
<img width="757" alt="截屏2025-01-13 22 16 18" src="https://github.com/user-attachments/assets/5d593f1f-e045-4a75-b91d-17b36aaa8ece" />

### GPU占用
<img width="1008" alt="截屏2025-01-13 22 16 42" src="https://github.com/user-attachments/assets/c2da1851-50cd-4105-b9e8-6aad984c863c" />
