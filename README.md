# Duke_AI_advisor
基于与训练大语言模型，创造Duke学生选课助手

## 目录
文件结构
```
.
├── data             # 数据
│   ├── finetune_data
│   └── rag_data
├── deployment       # 部署代码
│   ├── back_end
│   └── front_end
├── fine_tune        # 微调代码
│   └── mac
└── models           # 模型
    ├── Qwen2-0.5B-Instruct
    ├── Qwen2-1.5B-Instruct
    ├── bge-small-en-v1.5
    └── gte-base-en-v1.5
```


## 快速部署
### 环境
```
pip install -r requirements.txt
```

### 本地部署
``` bash
# backend
cd 4.deployment/        
uvicorn fastapi_v1:app --reload  

# frontend
streamlit run ./4.deployment/debug_frontend.py
```