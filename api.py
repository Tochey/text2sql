"""exposes our agent to chatfsp """
from text2sql import Text2Sql
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import Optional, AsyncGenerator
from fastapi.responses import StreamingResponse
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Union
from langchain_core.messages import HumanMessage, AIMessage
import uvicorn


load_dotenv()  # for the openai key

ROUTE_PREFIX = "legal-team-prompt-helper"


app = FastAPI()
agent = Text2Sql()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class StreamingItem(BaseModel):
    model: str
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    stop: list[str]
    user: str
    stream: bool
    messages: list

    def to_dict(self):
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop,
            "user": self.user,
            "stream": self.stream,
            "messages": self.messages,
        }

    def map_messages_to_langchain(self) -> List[Union[HumanMessage, AIMessage]]:
        """
        Maps the list of message dictionaries to Langchain HumanMessage and AIMessage objects.

        Returns:
            List[Union[HumanMessage, AIMessage]]: The list of Langchain message objects.
        """
        langchain_messages = []
        for message in self.messages:
            if message.get("role") == "user":
                langchain_messages.append(HumanMessage(content=message.get("content")))
            elif message.get("role") == "assistant":
                langchain_messages.append(AIMessage(content=message.get("content")))
            # Handle other cases as necessary
        return langchain_messages


def create_message(text: str, idx: int) -> str:
    message = {
        "id": "empty",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "Text2Sql Agent",
        "choices": [{"index": idx, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(message)}\n\n"


async def streaming_data(
    item: StreamingItem, message_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    text = item.messages[-1]["content"]
    print(item)
    async for chat_str in agent.invoke(
        text, item.map_messages_to_langchain(), message_id
    ):
        message = create_message(chat_str, 1)
        yield message
    yield "data: [DONE]\n\n"


@app.post("/text2sql/stream")
async def get_stream(item: StreamingItem, message_id: Optional[str] = None):
    return StreamingResponse(
        streaming_data(item, message_id), media_type="text/event-stream"
    )
    

uvicorn.run(app, host="0.0.0.0", port=int(8081), workers=1)
