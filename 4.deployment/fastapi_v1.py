import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch
import openai

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Milvus client
client = MilvusClient("../data/rag_data/dukies.db")

# RAG model
rag_model = SentenceTransformer("../models/gte-base-en-v1.5", trust_remote_code=True).to(device)

# Rerank model
rerank_tokenizer = AutoTokenizer.from_pretrained("../models/gte-multilingual-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    "../models/gte-multilingual-reranker-base", trust_remote_code=True, torch_dtype=torch.float16
).to(device)

# LLM model
llm_model = AutoModelForCausalLM.from_pretrained("../models/Qwen2.5-0.5B-Instruct", torch_dtype="auto", device_map="auto").to(device)
llm_tokenizer = AutoTokenizer.from_pretrained("../models/Qwen2.5-0.5B-Instruct")

# OpenAI API key
openai_api_key = os.getenv('API_KEY')
openai.api_key = openai_api_key

# Define request and response schemas
class QueryRequest(BaseModel):
    user_input: str
    rag_num_return: int = 10

class GenerateRequest(BaseModel):
    final_prompt: str

@app.post("/rag_search")
def rag_search(query: QueryRequest):
    """Retrieve documents using the RAG model."""
    try:
        query_vectors = rag_model.encode([query.user_input])
        res = client.search(
            collection_name="rag_collection",
            data=query_vectors,
            limit=query.rag_num_return,
            output_fields=["text", "subject"]
        )
        res_list = [res[0][i]['entity']['text'] for i in range(len(res[0]))]
        return {"results": res_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rerank")
def rerank(query: QueryRequest, retrieved_docs: list[str]):
    """Rerank retrieved documents using the rerank model."""
    try:
        pairs = [[query.user_input, doc] for doc in retrieved_docs]
        inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
        reranked_docs = [retrieved_docs[i] for i, score in enumerate(scores) if score > 0]
        return {"reranked_docs": reranked_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_response_qwen")
def generate_response_qwen(request: GenerateRequest):
    """Generate a final response using the LLM."""
    try:
        messages = [
            {"role": "system", "content": "You are Dukies, a helpful advisor."},
            {"role": "user", "content": request.final_prompt}
        ]
        text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = llm_tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = llm_model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_response_openai")
def generate_response_openai(request: GenerateRequest):
    """Generate a response using the OpenAI API."""
    try:
        opanai_client = openai.OpenAI(api_key=openai_api_key)
        response = opanai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                            {"role": "system", "content": "You are Dukies, a helpful advisor."},
                            {"role": "user", "content": request.final_prompt}
                        ],
            response_format={
                "type": "text"
            },
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_all")
def process_all(query: QueryRequest):
    """Chained endpoint: Calls RAG, Rerank, and preprocesses final_prompt."""
    try:
        # Step 1: RAG search
        rag_response = rag_search(query)
        retrieved_docs = rag_response["results"]

        # Step 2: Rerank
        rerank_response = rerank(query, retrieved_docs=retrieved_docs)
        reranked_docs = rerank_response["reranked_docs"]

        # Step 3: Choose model
        qwen_used = True
        if not reranked_docs:
            qwen_used = False
        
        # use qwen
        if qwen_used:
            rag_str = '\n\n'.join(reranked_docs)
            final_prompt = f"Answer the following question based on:\n{rag_str}\nQuestion: {query.user_input}"

            llm_response = generate_response_qwen(GenerateRequest(final_prompt=final_prompt))
            return {"response":llm_response['response'], "retrieved_docs":retrieved_docs, "reranked_docs":reranked_docs, "model_used":"Qwen"}
        else:
            final_prompt = query.user_input
            llm_response = generate_response_openai(GenerateRequest(final_prompt=final_prompt))
            llm_response = generate_response_qwen(GenerateRequest(final_prompt=final_prompt))
            return {"response":llm_response['response'], "retrieved_docs":retrieved_docs, "reranked_docs":reranked_docs, "model_used":"ChatGPT"}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
