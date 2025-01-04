import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from FlagEmbedding import FlagModel
import os
import numpy as np
import gc
import sys

# Initialize FastAPI app
app = FastAPI()

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")

def get_variable_sizes():
    variables = globals()
    for var_name, var_value in variables.items():
        print(f"{var_name}: {sys.getsizeof(var_value)} bytes")



# Check for device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load rag embedding model
embedding_model_name = "bge-small-en-v1.5"
embedding_model = FlagModel(f'./models/{embedding_model_name}', 
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation



# setup rag vector database
rag_directory_path = './data/rag_data'

# Lists to hold the content of all txt files and their names
txt_files_content = []
file_names = []

# Iterate over all files in the directory
for filename in sorted(os.listdir(rag_directory_path)):
    if filename.endswith('.txt'):
        file_path = os.path.join(rag_directory_path, filename)
        with open(file_path, 'r') as file:
            txt_files_content.append(file.read())
            file_names.append(filename)

file_len = len(file_names)
embedding_file_path = os.path.join(rag_directory_path, f'passage_embeddings_{embedding_model_name}.pt')
reload_embedding = True

if os.path.exists(embedding_file_path):
    # Load the tensor if the file exists
    passage_embeddings = torch.load(embedding_file_path)
    if len(passage_embeddings) == file_len:
        reload_embedding = False
        print(f"reuse embedding, read it from .pt file. in total, we have {file_len} files")

if reload_embedding:
    print(f"start embedding")
    passage_embeddings = embedding_model.encode(txt_files_content, convert_to_numpy=False)[0]
    torch.save(passage_embeddings, embedding_file_path)
    print(f"new embedding, embedding file saved. in total, we have {len(passage_embeddings)}")


# Load model and tokenizer
model_path = "./models/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device != torch.device("cpu") else torch.float32,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define request body
class RequestBody(BaseModel):
    messages: list

@app.post("/generate/")
async def generate_text(request_body: RequestBody):
    try:
        messages = request_body.messages 
        prompt = messages[-1]['content']

        prompt_embeddings = embedding_model.encode_queries([prompt], convert_to_numpy=False)[0]
        scores = prompt_embeddings @ passage_embeddings.T
        scores = scores[0].cpu()
        max_index = np.argmax(scores)
        file_name = file_names[max_index]
        file_content = txt_files_content[max_index] 

        prompt = 'refer to this content while answering the question: ' + file_content + 'question: ' + prompt
        messages[-1]['content'] = prompt
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        attention_mask = model_inputs['input_ids'].ne(tokenizer.pad_token_id).long().to(device)
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {"response": response, "file_name": file_name, "file_content": file_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)