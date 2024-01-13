import os
from openai import AzureOpenAI

import asyncio
import json
from typing import AsyncIterable, Awaitable, List, Optional
from uuid import UUID
import os

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatLiteLLM
from langchain.chat_models.base import BaseChatModel
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from logger import get_logger

from langchain.chat_models import AzureChatOpenAI



from typing import List, Tuple

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain.schema import HumanMessage
# from langchain_community.chat_models import ChatLiteLLM

from dotenv import load_dotenv  # type: ignore
load_dotenv()

def format_chat_history(history) -> List[Tuple[str, str]]:
    """Format the chat history into a list of tuples (human, ai)"""

    return [(chat.user_message, chat.assistant) for chat in history]


def format_history_to_openai_mesages(
    tuple_history: List[Tuple[str, str]], system_message: str, question: str
) -> List[BaseMessage]:
    """Format the chat history into a list of Base Messages"""
    messages = []
    messages.append(SystemMessage(content=system_message))
    for human, ai in tuple_history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    messages.append(HumanMessage(content=question))
    return messages




def azureTest():
    # gets the API Key from environment variable AZURE_OPENAI_API_KEY
    client = AzureOpenAI(
        api_key='70f860eb151b4b3fbb25767f5520d3a4',
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        api_version="2023-05-15",
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint="https://20230620asc.openai.azure.com/",
        azure_deployment='gpt-35-turbo-0613',
    )

    completion = client.chat.completions.create(
        model="deployment-name",  # e.g. gpt-35-instant
        messages=[
            {
                "role": "user",
                "content": " What's LiteLLM ChatLiteLLM and openai ?",
            },
        ],
    )
    print(completion.model_dump_json(indent=2))
    
def azureTest1(question):
    
    messages = 'what is ollama'
    # azureTest2(messages)
    
    chat = ChatLiteLLM(model="gpt-3.5-turbo")
    
    messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
        )
    ]
    res = chat(messages)
    print(res)
    
    
def azureTest2(question):
        messages = format_history_to_openai_mesages(
            [], 'as a science mentor', question
        )
        use_azure_chatgpt =  os.getenv('use_azure_chatgpt')
        print('use_azure_chatgpt : ', use_azure_chatgpt)
        OPENAI_PROXY =  os.getenv('OPENAI_PROXY')
        azure_deployment_id =  os.getenv('azure_deployment_id')
        azure_api_version =  os.getenv('azure_api_version')
        OPENAI_API_BASE  =  os.getenv('open_ai_api_base')
        model =  os.getenv('azure_deployment_id')
        azure_api_key =  os.getenv('open_ai_api_key')
        if use_azure_chatgpt:
            api_base = OPENAI_API_BASE
            model_kwargs = {}
            # model_kwargs["azure_deployment_id"] = azure_deployment_id
            model_kwargs["deployment_id"] = azure_deployment_id
            # model_kwargs["azure_api_version"] = azure_api_version
            model_kwargs["api_version"] = azure_api_version
            callbacks = [] # on_chat_model_start NotImplementedError: StdOutCallbackHandler does not implement `on_chat_model_start`
            callbacks = None
            answering_llm =  ChatLiteLLM(
                model=model,
                streaming=False,
                verbose=True,
                callbacks=callbacks,
                max_tokens=2048,
                api_base=api_base,
                # api_version=azure_api_version,
                azure_api_key=azure_api_key,
                # openai_api_key=azure_api_key,
                
                model_kwargs=model_kwargs,
            )
            model_prediction = answering_llm.predict_messages(messages)
            answer = model_prediction.content
            print('answer : ', answer)
        
        
    
def azureTest4(question):
    
    for x in  os.environ:
        print('env is : ', x, os.getenv(x))

    use_azure_chatgpt =  os.getenv('use_azure_chatgpt')
    print('use_azure_chatgpt : ', use_azure_chatgpt)
    OPENAI_PROXY =  os.getenv('OPENAI_PROXY')
    azure_deployment_id =  os.getenv('azure_deployment_id')
    azure_api_version =  os.getenv('azure_api_version')
    OPENAI_API_BASE  =  os.getenv('open_ai_api_base')
    model =  os.getenv('azure_deployment_id')
    azure_api_key =  os.getenv('open_ai_api_key')
    
    # model = AzureChatOpenAI(
    #     openai_api_version= azure_api_version,
    #     deployment_name=azure_deployment_id,
    #     azure_endpoint=OPENAI_API_BASE,
    #     openai_api_key=azure_api_key,
    # )
    
    model = AzureChatOpenAI(
        deployment_name=azure_deployment_id,
    )
    
    message = HumanMessage(
        content="Translate this sentence from English to Japanese. I love u."
    )

    messages = format_history_to_openai_mesages(
        [], 'as a science mentor', question
    )
    
    res = model(messages)
    
    print('answer : ', res)
    
    
if __name__ == "__main__":
    # azureTest2('does ChatLiteLLM with azure support streaming  ?')
    # azureTest2(" litellm.exceptions.APIConnectionError: 'ChatCompletionChunk' object has no attribute 'system_fingerprint' ")
    # azureTest2(" httpx.ConnectError: [Errno 11001] getaddrinfo failed  ")
    # azureTest2(" httpx.ConnectTimeout: _ssl.c:985: The handshake operation timed out")
    azureTest2(" Caught exception: cannot access local variable 'custom_llm_provider' where it is not associated with a value ")
    # OSError: [WinError 64] 指定的网络名不再可用。
    # 2024-01-12 14:09:41,242 [ERROR] llm.qa_headless: Caught exception: cannot access local variable 'custom_llm_provider' where it is not associated with a value
    
    # 
#   reverseProxyUrl: "https://api.openai.com/v1/chat/completions",
#   proxy: "http://localhost:7890/",