import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
# 환경 변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

def load_fix_introduction_data():
    file_path = os.path.join("userCase","1.txt")
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_user_chat_data(uuid, introduction, question):
    """사용자의 자기소개 데이터를 저장함"""
    file_path = os.path.join(uuid, "introduction.txt")

    with open(file_path, "a", encoding="utf-8") as f:  # "a" 모드로 추가 저장
        f.write(f'user_introduction: {introduction}\n')
        f.write(f'ai_question: {question}\n')

def load_user_chat_data(uuid):
    """사용자의 저장된 채팅 데이터를 불러옴"""
    file_path = os.path.join(uuid, "introduction.txt")
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text, chunk_size=1000, chunk_overlap=50):
    """문자열을 특정 크기로 분할"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


def get_user_introduction_vector_store(uuid, user_introduction):
    """사용자의 자서전 데이터의 벡터스토어를 생성 또는 로드"""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore_path = os.path.join(uuid, "introduction_index")

    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # 텍스트 분할
        split_documents = split_text(user_introduction)

        # 벡터스토어 생성 및 저장
        vectorstore = FAISS.from_texts(split_documents, embeddings)
        vectorstore.save_local(vectorstore_path)

    return vectorstore

def get_user_chat_vector_store(uuid, user_chat_data):
    """사용자의 채팅 데이터의 벡터스토어를 생성 또는 로드"""
    if user_chat_data is None: 
        return None

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore_path = os.path.join(uuid, "introduction_index")

    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # 텍스트 분할
        split_documents = split_text(user_chat_data)

        # 벡터스토어 생성 및 저장
        vectorstore = FAISS.from_texts(split_documents, embeddings)
        vectorstore.save_local(vectorstore_path)

    return vectorstore

def get_retriever(vectorstore):
    """FAISS 벡터스토어에서 검색기 생성"""
    if vectorstore is None:
        return None
    return vectorstore.as_retriever()


def get_prompt():
    return PromptTemplate.from_template(
        f"""
        당신은 사용자의 자기소개 데이터(introduction)와 이전 대화(chat)를 바탕으로, 사용자에게 맞춤형 목차 및 질문을 생성하는 역할을 합니다.

        1. 사용자의 자기소개 데이터를 분석하여, 기존에 설정된 유저별 목차 및 질문(data) 중 적절한 것을 부여하세요.
        2. 만약 사용자의 자기소개 데이터만으로 맞춤형 목차 및 질문을 부여하기 어렵다면, 어떤 정보가 더 필요한지 사용자에게 질문하세요.
        3. 사용자의 이전 대화(chat) 기록을 참고하여, 이미 주어진 정보가 있다면 이를 활용하여 중복 질문을 피하세요.
        4. 목차는 10개. 질문은 목차별 5개씩 부여하세요.
        5. 왜 해당 사용자에게 해당 맞춤형 목차 및 질문을 생성했는지 타당성 있는 글을 첨부하세요.
        6. 응답(response)은 반드시 다음 두 가지 데이터를 포함해야 합니다:
        (1) 'ok' 또는 'no': 현재 사용자의 자기소개 데이터만으로 맞춤형 목차 및 질문을 생성할 수 있는지 여부를 나타냅니다.
        (2) 세부 데이터:
            - 'ok'이면, 생성된 맞춤형 목차 및 질문을 제공하세요.
            - 'no'이면, 추가적으로 필요한 정보를 얻기 위한 질문을 포함하세요.

        # 사용자 자기소개 데이터:
        {{introduction}}

        # 이전 대화 기록:
        {{chat}}

        # 기존 유저별 목차 및 질문 데이터:
        {{data}}

        # 응답 형식:
        "status": "ok" 또는 "no",
        "details": "목차 및 질문 리스트 (ok일 경우) 또는 추가 질문 (no일 경우)"
        "reason": "해당 목차 및 질문 리스트를 부여한 이유"
        """
    )


def get_llm():
    """GPT 모델 생성"""
    return ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)


def get_chain(fix_introduction_retriever, chat_retriever, prompt, llm):
    """LLM 체인 생성 (NoneType 오류 방지)"""
    
    chat_runnable = chat_retriever if chat_retriever is not None else RunnablePassthrough()
    
    introduction_runnable = fix_introduction_retriever if fix_introduction_retriever is not None else RunnablePassthrough()

    return (
        {"data": introduction_runnable, "introduction": RunnablePassthrough(), "chat": chat_runnable}
        | prompt
        | llm
        | StrOutputParser()
    )



def main(uuid,user_introduction):
    """메인 실행 함수"""
    user_chat_data = load_user_chat_data(uuid)
    fix_introduction = load_fix_introduction_data()

    # 벡터스토어 로드 또는 생성
    fix_introduction_vectorstore = get_user_introduction_vector_store(uuid, fix_introduction)
    chat_vectorstore = get_user_chat_vector_store(uuid,user_chat_data)

    fix_introduction_retriever = get_retriever(fix_introduction_vectorstore)
    chat_retriever = get_retriever(chat_vectorstore)

    prompt = get_prompt()
    llm = get_llm()

    chain = get_chain(fix_introduction_retriever, chat_retriever, prompt, llm)

    # 질문 실행
    response = chain.invoke(user_introduction)

    # 채팅 저장
    save_user_chat_data(uuid, user_introduction, response)
    return response


# 테스트 실행 예제
if __name__ == "__main__":
    test_uuid = "test1"
    test_introduction = """
안녕 나는 서울에 살고있는 홍길동이라고해.
나는 남자고 55살이야. 나는 고등학교를 졸업했고, 건설쪽을 하고 있어.
결혼한 지 23년차야. 나는 5살 아들과 3살 딸이 있어."""
    response = main(test_uuid,test_introduction)
    print(response)
