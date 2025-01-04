import json
import os
from typing import Tuple
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
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
llm_model = AutoModelForCausalLM.from_pretrained("../models/Qwen2.5-0.5B-Finetuned", torch_dtype="auto", device_map="auto").to(device)
llm_tokenizer = AutoTokenizer.from_pretrained("../models/Qwen2.5-0.5B-Finetuned")

# OpenAI API key
openai_api_key = os.getenv('API_KEY')
openai.api_key = openai_api_key

# Define request and response schemas
class QueryRequest(BaseModel):
    user_input: str
    rag_num_return: int = 10

class GenerateRequest(BaseModel):
    final_prompt: str

class ToolRequest(BaseModel):
    TOOLS: list
    query: str

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
        class StopWordsCriteria(StoppingCriteria):
            def __init__(self, stop_words_ids):
                super().__init__()
                self.stop_words_ids = stop_words_ids

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                # Check if any stop word is present in the generated sequence
                for stop_word in self.stop_words_ids:
                    if input_ids[0][-len(stop_word):].tolist() == stop_word:
                        return True
                return False
        stop_words = ['Observation:', 'Observation:\n']
        stop_words_ids = [llm_tokenizer.encode(word, add_special_tokens=False) for word in stop_words]

        # Define the stopping criteria with stop words
        stopping_criteria = StoppingCriteriaList([StopWordsCriteria(stop_words_ids)])
        messages = [
            {"role": "system", "content": "You are Dukies, a helpful advisor."},
            {"role": "user", "content": request.final_prompt}
        ]
        text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = llm_tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = llm_model.generate(**model_inputs, max_new_tokens=512, stopping_criteria=stopping_criteria)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/tool_use_qwen")
def tool_use_qwen(request: ToolRequest):
    try:
        TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""

        REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

        {tool_descs}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {query}"""

        tool_descs = []
        tool_names = []
        TOOLS = request.TOOLS
        query = request.query
        for info in TOOLS:
            tool_descs.append(
                TOOL_DESC.format(
                    name_for_model=info['name_for_model'],
                    name_for_human=info['name_for_human'],
                    description_for_model=info['description_for_model'],
                    parameters=json.dumps(
                        info['parameters'], ensure_ascii=False),
                )
            )
            tool_names.append(info['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)

        prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=query)
        llm_response = generate_response_qwen(GenerateRequest(final_prompt=prompt)) 
        llm_response = llm_response['response']

        def parse_latest_plugin_call(text: str) -> Tuple[str, str]:
            i = text.rfind('\nAction:')
            j = text.rfind('\nAction Input:')
            k = text.rfind('\nObservation:')
            if 0 <= i < j:  # If the text has `Action` and `Action input`,
                if k < j:  # but does not contain `Observation`,
                    # then it is likely that `Observation` is ommited by the LLM,
                    # because the output text may have discarded the stop word.
                    text = text.rstrip() + '\nObservation:'  # Add it back.
                    k = text.rfind('\nObservation:')
            if 0 <= i < j < k:
                plugin_name = text[i + len('\nAction:'):j].strip()
                plugin_args = text[j + len('\nAction Input:'):k].strip()
                data = json.loads(plugin_args)
                return plugin_name, data
            return '', ''
        
        function_name, parameters = parse_latest_plugin_call(llm_response)
        tool_used = True
        final_response = ''
        if not function_name and not parameters:
            tool_used = False
            if '\nFinal Answer: ' in llm_response:
                final_response = llm_response[llm_response.rfind('\nFinal Answer: ')+len('\nFinal Answer: '):]
            else:
                final_response = llm_response
        return {'tool_used': tool_used, 'function_name':function_name, 'parameters':parameters, 'original_response':llm_response, 'final_response': final_response}
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
            return {"response":llm_response['response'], "retrieved_docs":retrieved_docs, "reranked_docs":reranked_docs, "model_used":"ChatGPT"}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
