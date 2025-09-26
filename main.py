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

#app.mount("/static",StaticFiles(directory="build/static"),name="static")



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
    pass

@app.post("/upload/")
async def getLink(text: str = Form(...)):
    return {"message":"Link received"}
    


@app.get("/result/")
async def sendData():
    return FileResponse("package.json")

@app.get("/")
def read_index():
    #return FileResponse("build/index.html")
    return {"Hello":"World"}
