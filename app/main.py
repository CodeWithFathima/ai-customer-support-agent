from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class chatRequest(BaseModel):
    question : str
@app.post('/chat')
def chat(request:chatRequest):
    return {'answer':f'your asked  {request.question}'}