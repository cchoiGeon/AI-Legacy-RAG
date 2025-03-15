import os
import aiofiles
import asyncio
from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# ThreadPoolExecutor 전역 선언 (재사용)
executor = ThreadPoolExecutor()

app = FastAPI()

async def save_user_data(uuid, userData):
    """사용자의 자서전 데이터를 비동기적으로 파일로 저장"""
    user_dir = os.path.join(uuid)
    os.makedirs(user_dir, exist_ok=True)

    file_path = os.path.join(user_dir, "bio.txt")
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(userData)

async def load_user_data(uuid):
    """사용자의 저장된 자서전 데이터를 비동기적으로 불러옴"""
    file_path = os.path.join(uuid, "bio.txt")
    if not os.path.exists(file_path):
        return None

    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        return await f.read()

def check_user_data(uuid):
    """사용자의 저장된 자서전 데이터 존재 여부 확인"""
    file_path = os.path.join(uuid, "bio.txt")
    return os.path.exists(file_path)

async def save_user_chat_data(uuid, question, response):
    """사용자의 저장된 채팅 데이터를 비동기적으로 저장"""
    file_path = os.path.join(uuid, "chat.txt")

    async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
        await f.write(f'question: {question}\n')
        await f.write(f'response: {response}\n')

async def load_user_chat_data(uuid):
    """사용자의 저장된 채팅 데이터를 비동기적으로 불러옴"""
    file_path = os.path.join(uuid, "chat.txt")
    if not os.path.exists(file_path):
        return None

    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        return await f.read()

async def split_text(text, chunk_size=1000, chunk_overlap=50):
    """문자열을 특정 크기로 비동기 분할"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

async def get_user_life_legacy_vector_store(uuid, userData):
    """사용자의 자서전 데이터의 벡터스토어를 생성 또는 로드 (비동기 처리)"""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore_path = os.path.join(uuid, "life_legacy_data_index")

    loop = asyncio.get_running_loop()

    if os.path.exists(vectorstore_path):
        # 비동기적으로 FAISS 로드
        vectorstore = await loop.run_in_executor(executor, lambda: FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True))
    else:
        # 텍스트 비동기 분할
        split_documents = await split_text(userData)

        # 비동기적으로 벡터스토어 생성 및 저장
        vectorstore = await loop.run_in_executor(executor, lambda: FAISS.from_texts(split_documents, embeddings))
        await loop.run_in_executor(executor, lambda: vectorstore.save_local(vectorstore_path))

    return vectorstore

async def get_user_chat_vector_store(uuid, user_chat_data):
    """사용자의 채팅 데이터의 벡터스토어를 생성 또는 로드 (비동기)"""
    if user_chat_data is None:
        return None

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore_path = os.path.join(uuid, "chat_data_index")

    loop = asyncio.get_running_loop()

    if os.path.exists(vectorstore_path):
        vectorstore = await loop.run_in_executor(executor, lambda: FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True))
    else:
        # 텍스트 비동기 분할
        split_documents = await split_text(user_chat_data)

        # 비동기적으로 벡터스토어 생성 및 저장
        vectorstore = await loop.run_in_executor(executor, lambda: FAISS.from_texts(split_documents, embeddings))
        await loop.run_in_executor(executor, lambda: vectorstore.save_local(vectorstore_path))

    return vectorstore

def get_retriever(vectorstore):
    """FAISS 벡터스토어에서 검색기 생성"""
    return vectorstore.as_retriever() if vectorstore else None

def get_prompt(role: str):
    """사용자의 자서전을 기반으로 특정 역할(role)에서 답변하는 프롬프트 생성"""
    return PromptTemplate.from_template(
        f"""
        당신은 사용자의 자서전과 이전 대화를 바탕으로, "{role}"의 입장에서 질문에 답변하는 역할을 합니다.
        "{role}"의 말투와 사고방식을 반영하여 답변을 제공해야 합니다.

        만약 질문이 자서전 데이터와 이전 대화에 포함된 정보와 관련이 있다면, "{role}"의 시각에서 직접 답변하세요.  
        만약 질문이 자서전 데이터와 이전 대화에 포함되지 않았다면, 자서전 내용을 바탕으로 "{role}"의 가치관과 성향을 유추하여 답변하세요.  
        그래도 답할 수 없는 경우, "죄송합니다. 해당 정보는 자서전에서 찾을 수 없습니다."라고 정중하게 답변하세요.

        또한, 이전 대화 기록을 참고하여 동일한 질문이 다시 나오면 이전 답변과 **완전히 같은 답변을 하지 마세요**.  
        질문이 비슷하더라도, 새로운 시각에서 답변을 제공하거나 **더 깊이 있는 설명**을 추가하세요.  
        답변이 중복되지 않도록 신경 쓰고, 사용자가 새로운 정보를 얻을 수 있도록 하세요.

        # 사용자 자서전:
        {{context}}

        # 이전 대화:
        {{chat}}

        # 현재 역할:
        {role}

        # 현재 질문:
        {{question}}

        # 답변 ("{role}"의 입장에서, 이전 답변과 다르게 답변하세요):
        """
    )

def get_llm():
    """GPT 모델 생성"""
    return ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

def get_chain(life_legacy_retriever, chat_retriever, prompt, llm):
    """LLM 체인 생성"""
    return (
        {"context": life_legacy_retriever or RunnablePassthrough(), "question": RunnablePassthrough(), "chat": chat_retriever or RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

class BioData(BaseModel):
    uuid: str
    name: str
    data: str

class ChatData(BaseModel):
    uuid: str
    role: str
    question: str

async def get_response(uuid, role, question):
    """비동기적으로 응답 생성"""
    user_life_legacy_data = await load_user_data(uuid)
    user_chat_data = await load_user_chat_data(uuid)

    if user_life_legacy_data is None:
        return ValueError('유저 데이터가 존재하지 않음 -> 잘못된 경로')

    # 벡터스토어 로드
    life_legacy_vectorstore = await get_user_life_legacy_vector_store(uuid, user_life_legacy_data)
    chat_vectorstore = await get_user_chat_vector_store(uuid, user_chat_data)

    life_legacy_retriever = get_retriever(life_legacy_vectorstore)
    chat_retriever = get_retriever(chat_vectorstore)

    prompt = get_prompt(role)
    llm = get_llm()
    chain = get_chain(life_legacy_retriever, chat_retriever, prompt, llm)

    response = chain.invoke(question)  # ✅ 수정

    return response

@app.post("/upload_bio")
async def save_bio(bio_data: BioData):
    if not await check_user_data(bio_data.uuid):
        await save_user_data(bio_data.uuid, bio_data.data)
    return {"success": True}

@app.post("/chat")
async def save_chat(chat_data: ChatData):
    response = await get_response(chat_data.uuid, chat_data.role, chat_data.question)
    await save_user_chat_data(chat_data.uuid, chat_data.question, response)
    return {"response": response}
