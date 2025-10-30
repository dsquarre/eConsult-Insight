import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=True)

#from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import os
import asyncio
from fastapi import FastAPI, Form, HTTPException

app = FastAPI()
from transformer import Analysis
from tfidf import Extract
#app.mount("/static",StaticFiles(directory="build/static"),name="static")

input_file = None

headers = {'Access-Control-Expose-Headers': 'Content-Disposition','Access-Control-Allow-Origin':'http://localhost:3000'}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=headers
)

@app.on_event("startup")  
async def startup_event():
    global input_file
    input_file = None

@app.post("/upload/")
async def getFile(file: UploadFile):
    global input_file
    import time
    time.sleep(2)
    input_file = file
    return {"message":"File received"}
    


@app.get("/result/")
async def sendData():
    global input_file
    sentences,sentiment0,best = Analysis(input_file)
    word_cloud,sentiment = Extract(sentences,sentiment0)
    import time
    time.sleep(2)
    
    return word_cloud,sentiment,best
@app.get("/")
def read_index():
    #return FileResponse("build/index.html")
    return {"Hello":"World"}


#add front and back