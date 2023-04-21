"""Main entrypoint for the app."""
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import Chroma
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from embedding import persist_embedding
from query_data import get_chain
from schemas import ChatResponse

# os.environ["OPENAI_API_KEY"] = ""
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
vectorstore: Optional[Chroma] = None


class Info(BaseModel):
    doc: str
    patent_id: str
    doc_type: str


ids = []


@app.post("/init")
async def init(*, info: Info):
    docs = info.doc
    patent_id = info.patent_id
    doc_type = info.doc_type
    item = patent_id + '@' + doc_type
    if item in ids:
        print("ids contains")
        return 'success'
    persist_embedding(docs, item)
    ids.append(item)
    return 'success'


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat/{patent_id}/{doc_type}")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    patent_id = websocket.path_params.get("patent_id")
    doc_type = websocket.path_params.get("doc_type")
    logging.info("loading vectorstore")
    directory_path = os.path.dirname(os.path.abspath(__file__))
    item = patent_id + '@' + doc_type
    if item not in ids:
        raise ValueError("vectorstore_path does not exist, please init first")
    vectorstore_path = directory_path + "/db/" + item + "-vectorstore.pkl"
    print("current path:" + vectorstore_path)
    if not Path(vectorstore_path).exists():
        raise ValueError("vectorstore.pkl does not exist, please init first")
    with open(vectorstore_path, "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)

    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())
            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):  # 如果是文件夹那么递归调用一下
            del_file(c_path)
        else:  # 如果是一个文件那么直接删除
            os.remove(c_path)
    print('文件已经清空完成')


if __name__ == "__main__":
    import uvicorn

    # clear db file
    # path = os.path.dirname(os.path.abspath(__file__)) + "/db"
    # del_file(path)

    # prepare ids
    path = os.path.dirname(os.path.abspath(__file__)) + "/db"
    ls = os.listdir(path)
    for i in ls:
        ids.append(i.removesuffix('-vectorstore.pkl'))
    uvicorn.run(app, host="0.0.0.0", port=9000)
