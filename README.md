# Duke_AI_advisor
智能领域顾问系统，集成检索增强生成（RAG）与函数调用能力，为复杂决策提供AI支持。


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
streamlit run ./4.deployment/hello.py
```
