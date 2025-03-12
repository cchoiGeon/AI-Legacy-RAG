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

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")


def save_user_data(uuid, userData):
    """사용자의 자서전 데이터를 파일로 저장"""
    user_dir = os.path.join(uuid)
    os.makedirs(user_dir, exist_ok=True)  # 사용자 폴더 생성
    file_path = os.path.join(user_dir, "bio.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(userData)

    return load_user_data(uuid)


def load_user_data(uuid):
    """사용자의 저장된 자서전 데이터를 불러옴"""
    file_path = os.path.join(uuid, "bio.txt")
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text, chunk_size=1000, chunk_overlap=50):
    """문자열을 특정 크기로 분할"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


def get_user_vector_store(uuid, userData=None):
    """사용자의 벡터스토어를 생성 또는 로드"""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore_path = os.path.join(uuid, "faiss_index")

    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # 사용자 데이터가 없으면 불러오기
        if userData is None:
            userData = load_user_data(uuid)
            if userData is None:
                raise ValueError("사용자 데이터가 존재하지 않습니다. userData를 입력해주세요.")

        # 텍스트 분할
        split_documents = split_text(userData)

        # 벡터스토어 생성 및 저장
        vectorstore = FAISS.from_texts(split_documents, embeddings)
        vectorstore.save_local(vectorstore_path)

    return vectorstore


def get_retriever(vectorstore):
    """FAISS 벡터스토어에서 검색기 생성"""
    return vectorstore.as_retriever()


def get_prompt():
    """질문을 위한 프롬프트 템플릿 생성"""
    return PromptTemplate.from_template(
        """
        당신은 질문에 답변하는 인공지능 상담사입니다. 
        아래의 사용자 자서전 정보를 참고하여 질문에 답변하세요. 
        만약 답을 모르면 데이터를 통해 추론을 해서 생각을 해보고 도저히 모르겠으면 모르겠습니다.라고 대답하세요.
        답변은 상담사가 해주는 식으로 답변해세요.
        질문이 정보를 토대로 하는 질문이 아니라면 사용자의 정보를 토대로 답변을 하세요.

        # 정보:
        {context}

        # 질문:
        {question}

        # 답변:
        """
    )


def get_llm():
    """GPT 모델 생성"""
    return ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)


def get_chain(retriever, prompt, llm):
    """LLM 체인 생성"""
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main(question, uuid, userLifeLegacyData):
    """메인 실행 함수"""
    userData = load_user_data(uuid)
    if userData is None:
        userData = save_user_data(uuid, userLifeLegacyData)
    
    # 벡터스토어 로드 또는 생성
    vectorstore = get_user_vector_store(uuid, userData)
    retriever = get_retriever(vectorstore)
    prompt = get_prompt()
    llm = get_llm()
    chain = get_chain(retriever, prompt, llm)

    # 질문 실행
    response = chain.invoke(question)
    return response


# 테스트 실행 예제
if __name__ == "__main__":
    test_uuid = "test1"
    test_data = "나는 봄을 좋아한다. 왜냐하면 따뜻한 바람이 불고 꽃이 피기 때문이다."
    test_question = "나는 왜 봄을 좋아할까?"
    response = main(test_question, test_uuid, test_data)
    print(response)
