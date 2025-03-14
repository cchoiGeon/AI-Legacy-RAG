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

def save_user_chat_data(uuid, question, response):
    """사용자의 저장된 채팅 데이터를 저장함"""
    file_path = os.path.join(uuid, "chat.txt")

    with open(file_path, "a", encoding="utf-8") as f:  # "a" 모드로 추가 저장
        f.write(f'question: {question}\n')
        f.write(f'response: {response}\n')

def load_user_chat_data(uuid):
    """사용자의 저장된 채팅 데이터를 불러옴"""
    file_path = os.path.join(uuid, "chat.txt")
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text, chunk_size=1000, chunk_overlap=50):
    """문자열을 특정 크기로 분할"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


def get_user_life_legacy_vector_store(uuid, userData):
    """사용자의 자서전 데이터의 벡터스토어를 생성 또는 로드"""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore_path = os.path.join(uuid, "life_legacy_data_index")

    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # 텍스트 분할
        split_documents = split_text(userData)

        # 벡터스토어 생성 및 저장
        vectorstore = FAISS.from_texts(split_documents, embeddings)
        vectorstore.save_local(vectorstore_path)

    return vectorstore

def get_user_chat_vector_store(uuid, user_chat_data):
    """사용자의 채팅 데이터의 벡터스토어를 생성 또는 로드"""
    if user_chat_data is None: 
        return None

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore_path = os.path.join(uuid, "chat_data_index")

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
    """LLM 체인 생성 (NoneType 오류 방지)"""
    
    # chat_retriever가 None이면 빈 문자열을 기본값으로 설정
    chat_runnable = chat_retriever if chat_retriever is not None else RunnablePassthrough()
    
    # life_legacy_retriever도 같은 방식으로 처리
    life_legacy_runnable = life_legacy_retriever if life_legacy_retriever is not None else RunnablePassthrough()

    return (
        {"context": life_legacy_runnable, "question": RunnablePassthrough(), "chat": chat_runnable}
        | prompt
        | llm
        | StrOutputParser()
    )



def main(question, uuid, new_user_life_legacy_data,role):
    """메인 실행 함수"""
    user_life_legacy_data = load_user_data(uuid)
    user_chat_data = load_user_chat_data(uuid)
    if user_life_legacy_data is None:
        user_life_legacy_data = save_user_data(uuid, new_user_life_legacy_data)
    

    # 벡터스토어 로드 또는 생성
    life_legacy_vectorstore = get_user_life_legacy_vector_store(uuid, user_life_legacy_data)
    chat_vectorstore = get_user_chat_vector_store(uuid,user_chat_data)

    life_legacy_retriever = get_retriever(life_legacy_vectorstore)
    chat_retriever = get_retriever(chat_vectorstore)

    prompt = get_prompt(role)
    llm = get_llm()

    chain = get_chain(life_legacy_retriever, chat_retriever, prompt, llm)

    # 질문 실행
    response = chain.invoke(question)

    # 채팅 저장
    save_user_chat_data(uuid, question, response)
    return response


# 테스트 실행 예제
if __name__ == "__main__":
    test_uuid = "test1"
    test_data = """
1.
탄생과 유아기 시절
Q.
언제 어디서 태어나셨나요? 탄생에 얽힌 이야기가 있나요?
A. 나는 1969년 5월 15일, 대한민국 서울에서 태어났습니다. 부모님께서는 결혼 후 몇 년 동
안 자녀를 갖지 못해 많은 노력을 하셨고, 그래서 내 탄생이 더욱 특별했다고 합니다.
어머니
는 나를 품에 안은 순간 눈
물을 흘리셨다고 하셨고, 아버지는 직접 내 이름을 지으며 기뻐하
셨다고 합니다. 할아버지는 내가 태어나던 날 동네에서 돼지머리를 잡고 잔치를 열었다고 합
니다. 집안에서 오랫동안 기다리던 첫 손자였기에, 나의 탄생은
가족
들에
게 큰 기쁨이었습니
다.
Q. 형제자매 중 몇째로 태어나셨고, 그것이 당
신에
게
어떤 영향을 미쳤나요?
A. 나는 삼남매 중 장남으로 태어났습니다. 장남이라는 위치 덕분에 자연스럽
게
책임감이 생
겼고, 동생들을 보살피는 것이 당
연한 일처럼 느껴졌습니다. 부모님이 맞벌이를 하셨기 때문
에 나는 일찍부터 독립심을 길러야 했습니다. 동생들이 어려운 일이 있으면 먼저 나서서 해
결해 주려고 노력했고,
가끔은 엄한 형이 되어 훈육도 했습니다. 이런 환경 덕분에 어른이 되
어도 책임을 회피하지 않는 성격을 갖
게
되었습니다.
Q. 유아기 때 가장 좋아했던 장난감이나 놀이가 있었나요?
A. 나는 나무로 만든 장난감 기차를 가장 좋아했습니다. 아버지가 직접 만들어 주신 장난감
이었기에 더욱 소중했고, 기차를 선로 위에 놓고 상상의 세계를 펼치며 오랜 시간 혼자서도
재미있
게
놀았습니다. 또한, 동네 친구들과 함께 구슬치기, 딱지치기,
숨바꼭질을 하면서 시
간을 보내곤 했습니다.
가끔은 온 동네를 놀이터 삼아 뛰어다니며 탐
험을 하는 것이 무척 신
나는 일이었습니다. 요즘 아이들처럼
게
임기나 스마트폰은 없었지만,
골목길에서의 놀이가
내 어린 시절의 가장 소중한 추억이 되었습니다.
Q. 부모님이 유아기 때 당
신을 위해 특별히 해주셨던 일이 있나요?
A. 어머니는 늘 책을 읽어주셨습니다. 잠들기 전이면 꼭 한 권씩 동화책을 읽어주셨고, 덕분
에 나는 글을 배우는 것이 어렵지 않았습니다. 아버지는 바쁜 와중에도 일요일이면 공원에
데려가
자전거를 타는 법을 가르쳐 주셨습니다. 몇 번이나 넘어져서 울면서도 다시 도전했
고, 결국 혼자 타게
되었을 때 아버지는 나를 꼭 안아주며 칭찬해 주셨습니다. 이런 부모님의
사랑 덕분에 나는 도전하는 것을 두려워하지 않는 성격이 되었습니다.
Q.
어릴 때 살던 곳과 그곳에서의 추억이 있나요?
A. 나는 서울의 작은 주택가에서 자랐습니다.
골목이 좁고 마당이 있는 집이었는데, 마당에
는 감나무가 있어서 가을이면 감이 주렁주렁 열렸습니다. 동네 친구들과 함께 감나무 아래에
서 놀다가
감을 따 먹는 재미가 있었습니다. 겨울이면 마당에 작은
눈사람을 만들고, 동생들
예시 데이터 PDF 1
과 눈
싸움을 하면서 즐거운 시간을 보냈습니다. 지금은 고층 아파트가 들어선 그곳이지만,
내 기억 속에는 여전히 따뜻한 고향처럼 남아 있습니다.
2.
가족환경과 성장환경
Q. 부모님에 대한 기억은
어떤가요?
A. 아버지는 엄격한 분이셨지만 가정에서는 누구보다도 따뜻한 사람이었습니다. 직장에서는
철저한 원칙주의자였지만,
집에 오면 나와 동생들에
게
장난을 치며 웃음을 주셨습니다.
어머
니는 인내심이 강하고 부드러운 성품을 가지신 분이셨습니다. 힘든 일이 있어도 잘 표현하지
않으셨고,
언제나 가족을 먼저 생
각하셨습니다. 부모님 두 분 모두 나에
게 성실함과
가족의
소중함을 몸소 보여주셨습니다.
Q.
가족
과 함께 했던 여행 중 가장 기억에 남는 장소는 어디인가요?
A. 어릴 때 부모님과 함께 다녀온 강릉 여행이 가장 기억에 남습니다. 처음으로 바
다를 본 날
이었는데, 끝없이 펼쳐진 파란 바
다를 보며 신기해했던 기억이 납니다. 아버지는 내 손을 잡
고 모래사장에서 달리기를 하셨고,
어머니는 집에서 싸온 김밥을 꺼내 모두에
게
나눠주셨습
니다. 해변에서
조
개껍데기를 주워 모으는 것도 큰 재미였고, 밤에는 바
닷
가
근처에서 회를
먹으며 특별한 저녁 시간을 보냈습니다. 그날 이후로 바
다는 나에
게
항상 설레는 장소로 남
아 있습니다.
Q.
가족 중에서 가장 존경했던 인물은 누구인가요?
A. 나는 할아버지를 가장 존경했습니다. 할아버지는 한평생 성실하
게
농사를 지으셨고,
가족
을 위해 희생하는 삶을 사셨습니다.
어릴 때 방학이면 시골에 내려가 할아버지를 도와 밭일
을 했는데, 그때마다 삶에 대한 지혜를 많이 배웠습니다. 항상 남을 돕고, 겸손함을 잃지 않는
태도가 나에
게 큰 가르침이 되었습니다. 할아버지께서는 늘 "성실하
게
사는 것이 가장 큰 재
산이다"라고 말씀하셨고, 나는 그 말을 가슴에 새기고 살아가고 있습니다.
Q.
어렸을 적 형제자매와의 관계는 어땠나요? 특별히 친했던 형제자매는 누구였나요?
A. 나는 둘째 동생과
가장 친했습니다. 나이 차이가 별로 나지 않아서 같이 놀기도 많이 했
고, 고민을 나누는 친구 같은 존재였습니다. 하지만 때로는 장남으로서 동생들을 이끌어야
했기에, 엄한 형이 되기도 했습니다. 동생이 실수를 하면 부모님께 혼나지 않도록 내가 먼저
조
언을 해주기도 했고, 힘든 일이 있으면 함께 해결하려고 노력했습니다. 지금도 우리는 자
주 연락하며 서로의 인생을 응원하는 가까운 사이입니다.
Q. 형제자매와의 경쟁이나 다툼이 있었나요? 그로 인해 배운 점이 있나요?
A. 어릴 때 동생과 자주 경쟁했습니다. 공부 성적이 더 좋은지, 운동을 더 잘하는지 서로 비
교하며 때로는 다투기도 했습니다. 하지만 이런 경쟁이 있었기에 우리는 서로를 더욱 발전시
킬 수 있었습니다. 경쟁 속에서 협력의 중요성도 배우
게
되었고,
가족
은 결국 경쟁자
가 아니
라 서로를 돕는 존재라는 것을 깨닫
게
되었습니다. 지금은 서로의 성공을 기쁘
게
응원하는
형제입니다.
예시 데이터 PDF 2
3.
청소년기와 학창시절
Q.
청소년기에 살았던 곳과 그곳에서의 추억은 무엇인가요?
A. 청소년기에는 서울의 번화가
근처로 이사하
게
되어 도시 생활을 경험하
게
되었습니다.
새
로운 환경에서 친구들과 함께 극장에 가서 영화를 보거나, 음악다방에서 음악을 들으며 시간
을 보내곤 했습니다. 특히, 한강 둔치에서 자전거를 타며 친구들과 경쟁하던 기억이 생생합
니다. 이러한 도시의 다양한 문화와 활동은 제 시야를 넓혀주었고,
새로운 취미를 발견하는
계기가 되었습니다.
Q. 학창시절 부모님과의 관계는 어땠나요?
A. 부모님과의 관계는 여전히 돈독했지만, 사춘기의 영향으로 가끔 의견 충돌이 있었습니
다. 특히 진로 선택에 있어 아버지와의 의견 차이가 있었지만, 결국 서로의 입장을 이해하며
조
율해 나갔습니다.
어머니는 항상 제 편이 되어주셨고, 감정적으로 힘든 시기에 큰 위로가
되었습니다. 부모님의 이러한 지지와 이해 덕분에 저는 어려운 시기를 잘 극복할 수 있었습
니다.
Q. 사춘기 시절 위험하거나 반항적인 행동을 한 적이 있나요?
A. 사춘기 시절
, 친구들과
어울려 가출을 계획한 적이 있었습니다.
당시에는 부모님의 간섭
이 답답하
게
느껴
져
자유를 꿈꾸었지만, 실제로 실행에 옮기지는 않았습니다. 또한, 학교 규
정을 어기고 밤늦
게
까지 놀다가 부모님께 혼난 적도 있습니다. 이러한 경험을 통해 책임의
중요성과
가족의 소중함을 깨닫
게
되었습니다.
Q.
청소년기 가장 친한 친구는 누구였으며, 그 친구와의 추억은
어떤가요?
A. 중학교 시절
만난 동갑내기 친구 철수와
가장 친했습니다. 우리는 함께 축구를 하며 주말
을 보냈고, 시험 기간에는 서로의 집에서 함께 공부하며 경쟁하곤 했습니다. 한 번은 둘이서
무작정 기차를 타고 바
다를 보러 간 적이 있었는데, 그때의 모험이 아직도 기억에 남습니
다. 철수와의 우정은 제 청소년기를 더욱 풍요롭
게
만들어주었습니다.
Q. 학창시절에 가장 자랑스러웠던 순간은
언제였나요?
A. 고등학교 2학년 때, 전국 과학 경시대회에서 수상한 것이 가장 자랑스러웠습니다. 밤늦
게
까지 실험하고 연구한 결과
가 인정받아 뿌듯했고, 학교에서도 큰 축하를 받았습니다. 이 경
험
은 제 자
신감을 크
게 높여주었고, 이후 진로 선택에도 긍정적인 영향을 미쳤습니다.
Q.
청소년기에 가장 열정을 가졌던 활동이나 취미는 무엇이었나요?
A. 청소년기에는 음악에 대한 열정이 컸습니다. 특히 기타 연주에 빠져 방과 후에는 음악실
에서 연습하며 시간을 보냈습니다. 학교 밴드에 가입하여 공연을 준비하고, 친구들과 함께
곡을 만들며 즐거움을 느꼈습니다. 이러한 음악 활동은 제 감성을 풍부하
게 해주었고, 스트
레스를 해소하는 데 큰 도움이 되었습니다.
예시 데이터 PDF 3
4. 20대 시절
-대학생활
Q1. 대학에 입학하
게
된 계기는 무엇인가요? 왜 그 학교를 선택했나요?
A1. 어렸을 때부터 과학과 기술에 대한 깊은
관심이 있었습니다. 특히, 고등학교 시절
참여한
과학 경시대회에서의 경험
은 이러한 열정을 더욱 확고히 해주었습니다. 이러한 이유로 공학
분야로 유명한 서울의 A대학교에 지원하
게
되었습니다. A대학교는 우수한 교수진과 최첨단
연구 시설을 보유하고 있어 제 학문적 목표와 부합한다고 생
각했습니다. 또한, 서울이라는
대도시에서 다양한 문화와 경험을 쌓을 수 있다는 점도 큰 매력으로 다가왔습니다.
Q2. 대학에서 가장 좋아했던 과목이나 교수는 누구였나요? 그 이유는 무엇인가요?
A2. '재료
공학' 과목을 가르치시던 김 교수님을 가장 존경했습니다. 김 교수님은 이론과 실
습을
조
화롭
게
가르치셨으며, 학생들의 참여를 적극적으로 유도하셨습니다. 특히, 산업 현장
의 실제 사례를 수업에 도입하여 학문과 실무의 연계를 강
조
하셨습니다. 이러한 교수님의 열
정적인 강의는 제 전공에 대한 흥미를 더욱 높여주었으며, 학문적 탐구심을 자극했습니다.
Q3. 대학 시절 가장 친했던 친구와의 추억은 무엇인가요?
A3. 동기인 민수와의 우정은 대학 생활의 큰 부분을 차지했습니다. 함께 동아리에 가입하여
다양한 활동을 했으며, 방학 때는 배낭여행을 다니며 추억을 쌓았습니다. 특히, 제주도로의
자전거 여행은 힘들었지만 아름다운 경치를 보며 서로를 격려했던 소중한 기억으로 남아 있
습니다. 이러한 경험들은 우리의 우정을 더욱 깊
게
만들어주었습니다.
Q4. 대학 생활 중 가장 힘들었던 순간은
언제였나요? 그것을 어떻
게
극복했나요?
A4. 3학년 때, 전공
과목의 프로젝트와 아르바이트를 병행하느라 체력적으로나 정신적으로
많이 지쳤습니다.
과제 마감과 업무 일정이 겹쳐 잠을 줄여가며 과제를 수행해야 했고, 성적
에 대한 압박
감도 컸습니다. 그러나 친구들과 스터디 그룹을 만들어 서로 도움을 주고받
으며
어려움을 극복했습니다. 또한, 주말에는 시간을 내어 취미 활동을 하며 스트레스를 해소하려
노력했습니다.
Q5. 대학 시절에 얻은
가장 큰 교훈이나 깨달음은 무엇이었나요?
A5. 대학 시절
다양한 경험을 통해 '도전의 중요성'을 깨달았습니다.
새로운 분야에 대한 도
전과 실패를 두려워하지 않고 적극적으로 임하는 자
세
가 얼마나 큰 성장을 가져오는지 경험
했습니다. 이러한 깨달음은 이후의 삶에서도 새로운 기회에 대한 두려움을 극복하고, 지속적
인 자기 발전을 추구하는 원동력이 되었습니다.
5. 20대 시절
- 직장생활
Q. 첫 직장을 어떻
게
구하
게
되었나요? 그 과정에 대해 이야기해주세요.
예시 데이터 PDF 4
A. 대학 졸업 후
, 전공을 살려 반도체 회사에 지원하
게
되었습니다.
당시 취업 경쟁이 치열했
지만, 학교에서 진행한 산학협력 프로그램을 통해 인턴십을 경험하며 실무 능력을 쌓았습니
다. 이러한 경험이 면접에서 긍정적으로 작용하여 최종 합격할 수 있었습니다. 첫 출근 날의
설렘
과 긴장은 아직도 생생하
게
기억납니다.
Q. 직장 생활에서 가장 기억에 남는 프로젝트는 무엇이었나요? 그 프로젝트의 성공 요인은 무
엇이었나요?
A. 입사 3년 차에 참여한 '신제품 개발 프로젝트'가 가장 기억에 남습니다.
팀
원들과 밤낮없
이 노력하며 기술적 난관을 극복했고, 결국 성공적인 제품 출시를 이뤄냈습니다.
팀
원 간의
원활한 소통과 협업, 그리고 끈기가 성공의 주요 요인이었다고 생
각합니다. 이 경험
은 제 경
력에 큰 전환점을 가져다주었습니다.
Q. 직장에서의 인간관계는 어땠나요? 특별히 친했던 동료
나 멘토가 있나요?
A. 처음에는 낯설었지만, 시간이 지나면서 동료
들과 두터운 친분을 쌓았습니다. 특히 선배인
김 대리님은 업무뿐만 아니라 인생 상담까지 해주시는 멘토 같은 분이었습니다. 그의
조
언과
지지는 제가 어려움을 극복하는 데 큰 힘이 되었습니다. 또한, 동료
들과의 팀워크는 업무 효
율성을 높이는 데 큰 역할을 했습니다.
Q. 직장 생활 중 가장 힘들었던 순간은
언제였나요? 그 어려움을 어떻
게
극복했나요?
A. 프로젝트 마감 기한이 촉박
해 연일 야근을 해야 했던 시기가 있었습니다. 체력적, 정신적
으로 지쳤지만, 팀
원들과 서로 격려하며 버텼습니다. 또한, 주말에는 짧은 휴식을 취하며 재
충전하려 노력했습니다. 이러한 노력 덕분에 프로젝트를 성공적으로 마무리할 수 있었습니
다.
Q. 직장 생활을 통해 얻은
가장 큰 교훈은 무엇인가요?
A. 인내와 협업의 중요성을 깊이 깨달았습니다.
어려운 상황에서도 포기하지 않고 끝까지 노
력하는 자
세
가 필요하다는 것을 배웠습니다. 또한, 다양한 사람들과의 협업을 통해 시야를
넓히고 성장할 수 있었습니다. 이러한 경험들은 제 인생의 소중한 자산이 되었습니다.
6. 결혼 이야기
Q. 결혼 전 배우자와의 첫 만남은
어땠나요? 그때의 느낌을 자
세히 말씀해주세요.
A. 친구의 소개로 한 카페에서 처음 만났습니다. 그녀의 밝은 미소와 따뜻한 눈빛에 첫눈에
끌렸습니다. 서로의 관심사와 취미가 비슷해 대화가 끊이지 않았고, 시간 가는 줄 몰랐습니
다. 그날 이후 그녀 생
각에 잠 못 이루는 밤이 많았습니다.
Q. 결혼을 결심하
게
된 이유는 무엇인가요?
A. 함께하는 시간이 늘어날수록 그녀의 배려심과
긍정적인 에너지에 감동받았습니다.
어떤
어려움이 닥쳐도 함께라면 이겨낼 수 있다는 확신이 들었습니다. 또한, 그녀와 함께하는 미
래를 그리며 행복감을 느꼈기에 결혼을 결심하
게
되었습니다.
예시 데이터 PDF 5
Q. 신혼생활 중 가장 기억에 남는 일은 무엇이었나요?
A. 첫 부부 여행으로 제주도를 방문했을 때가 가장 기억에 남습니다. 렌터카를 타고 섬 곳곳
을 탐
험하며 추억을 쌓았습니다. 특히, 해변에서 함께 본 일몰은 잊을 수 없는 장면입니다. 이
여행은 우리에
게 큰 행복과 추억을 선사했습니다.
Q. 결혼 생활 중 힘들었던 순간과 행복했던 순간은
언제였나요?
A. 초기에는 서로의 생활 패
턴 차이로 작은 다툼이 있었습니다. 그러나 대화를 통해 서로를
이해하고 배려하며 문제를 해결해 나갔습니다. 행복했던 순간은 첫 아이의 탄생으로 부모가
되었을 때입니다. 작은 생명이 주는 기쁨은 말로 표현할 수 없을 정도였습니다.
Q. 결혼 생활에서 가장 중요한 가치는 무엇이라고 생각하시나요?
A. 서로에 대한 존중과
신뢰가 가장 중요하다고 생
각합니다.
어떤 상황에서도 상대를 배려하
고 이해하려는 노력이 필요합니다. 또한, 함께 성장하고자 하는 의지와 사랑이 결혼 생활을
더욱 풍요롭
게
만들어줍니다.
7 .
자녀 이야기
Q. 첫 아이가 태어난 날에 대한 기억을 말씀해주세요.
A. 첫 아이가 태어난 날은 제 인생에서 가장 감동적인 순간 중 하나였습니다. 아내의 손을 꼭
잡고 분만실에서 기다리던 그 떨림과 설렘
은 아직도 생생합니다. 아이가 첫 울음을 터뜨렸을
때, 말로 표현할 수 없는 기쁨과 함께 부모로서의 책임감을 깊이 느꼈습니다. 그 작은 생명이
우리에
게
주는 행복은 이루 말할 수 없었습니다.
Q. 아이들과 함께한 추억 중 가장 의미 있는 것은 무엇인가요?
A. 가족 캠핑 여행은 우리 가족에
게
특별한 추억으로 남아 있습니다.
자연 속에서 함께 텐트
를 치고, 모닥불을 피우며 도란도란 이야기를 나누던 시간은 아이들에
게
도, 저에
게
도 소중한
기억입니다. 특히, 아이들이 자연의 아름다움과 소중함을 직접 느끼고 배울 수 있었던 시간
이었습니다.
Q. 부모로서 가장 자랑스러웠던 순간은
언제였나요?
A. 아이가 학교에서 주최한 과학 경진대회에서 최우수상을 받았을 때 정말 자랑스러웠습니
다.
자
신의 노력과 열정으로 성과를 이뤄낸 모습을 보며, 부모로서 큰 기쁨과 뿌듯함을 느꼈
습니다. 이러한 경험이 아이에
게
도 자
신감을 심어주었기를 바랍니다.
Q. 아이를 돌보면서 가장 어려웠던 점
은 무엇이었나요?
A. 사춘기 시절 아이와의 의사소통이 가장 큰 도전이었습니다. 서로의 생
각
과
감정을 이해하
는 데 어려움이 있었지만, 꾸준한 대화와
신뢰를 통해 이겨낼 수 있었습니다. 이 과정을 통해
부모로서의 인내와 사랑의 깊이를 배울 수 있었습니다.
Q. 자녀들에
게
가장 중요하
게
가르치고 싶었던 가치는 무엇이었나요?
예시 데이터 PDF 6
A. 정직과 배려의 가치를 가장 중요하
게
가르치고자 했습니다.
자
신에
게
솔직하고, 타인을
배려하는 마음이 인생을 살아가는 데 있어 가장 기본이자 중요한 덕목이라고 생
각합니다. 이
러한 가치를 통해 아이들이 올바른 인성을 갖추길
바
랐습니다.
Q. 자녀들이 성장하면서 부모로서 가장 보람을 느꼈던 순간은
언제였나요?
A. 아이들이 자
신만의 길을 찾아 열심히 노력하는 모습을 볼 때마다 큰 보람을 느꼈습니
다. 특히, 사회에 나가
자
신들의 역할을 다하며 책임감 있
게
살아가는 모습을 볼 때 부모로서
의 역할을 잘 해냈다는 생
각이 들었습니다.
8. 현재의 삶
Q. 현재의 생활에서 가장 만족스러운 부분은 무엇인가요?
A. 가족
과 함께하는 시간이 많아진 것이 가장 만족스럽습니다. 아이들이 성장하여 각
자의 삶
을 살아가고 있지만, 주기적으로 모여 함께 식사하고 이야기를 나누는 시간이 큰 행복을 줍
니다.
Q. 현재 집중하고 있는 취미나 활동은 무엇인가요? 그것이 당
신에
게
어떤 의미를 주고 있나
요?
A. 최근에는 그림 그리기에 흥미를 느껴
취미로 삼고 있습니다. 이 활동은 저에
게
창의적인
표현의 기회를 주고, 마음의 평안을 찾는 데 큰 도움이 됩니다. 또한,
새로운 기술을 배우는
즐거움도 느끼고 있습니다.
Q. 현재의 생활에서 가장 큰 도전은 무엇인가요? 그 도전을 어떻
게
극복하고 있나요?
A. 건강 관리는 현재 가장 큰 도전 중 하나입니다. 나이가 들면서 체력과 건강에 신경을 쓰
게
되는데, 규칙적인 운동과 건강한 식습관을 통해 이를 극복하려 노력하고 있습니다. 또한, 정
기적인 건강 검진을 통해 예방에 힘쓰고 있습니다.
Q. 지금의 삶에서 가장 감사하
게
생각하는 사람이나 사건은 무엇인가요?
A. 가족
들의 변함없는 지지와 사랑에 가장 감사함을 느낍니다. 특히,
어려운 시기에 함께해
준 아내와 아이들의 존재는 제 삶의 큰 힘이 되었습니다. 그들의 응원과 사랑이 있었기에 지
금의 제가 있을 수 있었습니다.
Q. 현재의 삶에서 이루고 싶은 목표나 계획이 있나요? 그 목표를 이루기 위해 어떤 준비를 하
고 있나요?
A. 앞으로는 지역 사회에 봉사하며 나눔의 삶을 살고 싶습니다. 이를 위해 봉사 단체에 가입
하여 다양한 활동에 참여하고 있으며, 필요한 역량을 키우기 위해 관련 교육도 받고 있습니
다. 이러한 활동을 통해 사회에 긍정적인 영향을 주고 싶습니다.
9. 인생의 교훈과 회고
예시 데이터 PDF 7
Q. 지금까지의 인생에서 가장 힘들었던 순간은
언제였고, 그 순간을 어떻
게
극복했나요?
A. 직장 생활 중 예상치 못한 구
조조
정으로 인해 실직을 경험한 적이 있습니다.
가족에 대한
책임감과 미래에 대한 불안으로 힘든 시기였지만, 긍정적인 마인드를 유지하며 새로운 기회
를 찾기 위해 노력했습니다. 그 결과
, 이전보다 더 나은 직장을 찾을 수 있었고, 이 경험을 통
해 역경 속에서도 희망을 잃지 않는 법을 배웠습니다.
Q. 인생에서 가장 후회되는 결정은 무엇인가요? 그 결정을 통해 배운 점이 있나요?
A. 젊
은 시절
, 해외 유학의 기회가 있었지만 두려움과 익숙한 환경을 떠나기 싫어 그 기회를
놓친 것이 후회로 남습니다. 이 경험을 통해 새로운 도전에 대한 두려움이 성장의 기회를 막
을 수 있다는 것을 깨달았고, 이후에는 도전에 적극적으로 임하는 자
세를 갖추
게
되었습니
다.
Q. 인생에서 가장 큰 기쁨을 느꼈던 순간은
언제였나요? 그 순간이 왜 특별했나요?
A. 첫 아이가 태어난 순간은 말로 표현할 수 없는 기쁨이었습니다.
새로운 생명의 탄생은 제
게
삶의 의미와 책임감을 동시에 안겨주었으며,
가족의 소중함을 깊이 느끼
게 해주었습니다.
Q. 지금까지의 인생에서 가장 감사하
게
생각하는 사람은 누구이며, 그 이유는 무엇인가요?
A. 부모님께 가장 큰 감사를 느낍니다. 그분들의 희생과 사랑이 없었다면 지금의 제가 존재
하지 않았을 것입니다.
어려운 환경 속에서도 항상 지지해 주시고 올바른 길로 인도해 주신
부모님의 은혜를 잊을 수 없습니다.
Q. 다음 세대에
게
꼭 전하고 싶은 교훈이나
조
언이 있다면 무엇인가요?
A. 실패를 두려워하지 말고 도전에 임하라는 말을 전하고 싶습니다. 실패는 성공의 밑거름이
며, 도전 없는 삶은 정체될 수밖에 없습니다. 또한, 주변 사람들과의 관계를 소중히 여기고,
작은 것에도 감사하는 마음을 가지길
바랍니다.
10. 미래의 계획과 희망
Q. 앞으로 이루고 싶은 목표나 꿈이 있나요? 그 목표를 이루기 위해 어떤 계획을 세우고 계신
가요?
A. 앞으로는 지역 사회에 기여하는 봉사 활동을 적극적으로 하고 싶습니다. 이를 위해 현재
관련 단체에 가입하여 활동을 시작하였으며, 필요한 역량을 키우기 위해 관련 교육도 받고
있습니다.
Q. 앞으로의 삶에서 가장 중요하
게
생각하는 가치는 무엇인가요?
A. 정직과 배려를 가장 중요하
게
생
각합니다.
자
신에
게
솔직하고 타인을 배려하는 마음이 인
생을 풍요롭
게
만든다고 믿습니다.
Q.
새로운 기술이나 취미를 배우고 싶은 계획이 있나요?
A. 최근에는 요리에 관심이 생겨 요리 강좌를 수강할 계획입니다. 이를 통해 가족
과 친구들
에
게
맛있는 음식을 대접하며 즐거움을 나누고 싶습니다.
Q. 앞으로의 여행 계획이 있다면 어디를 가고 싶으
신
가요?
A. 자연의 아름다움을 만끽할 수 있는 뉴질랜드로 여행을 가고 싶습니다. 그곳의 풍경을 직
접 눈
으로 보고 느끼며 삶의 활력을 얻고자 합니다.
Q. 미래의 자
신에
게
가장 하고 싶은 말은 무엇인가요?
A. 지금까지 잘 해왔고, 앞으로도 긍정적인 마음으"""
    test_question = "아빠 그때 기억나? 가족끼리 여행을 하다가 사고났을 때 아빠는 나만 걱정했잖아. 그때 아빠도 아팠을텐데"
    response = main(test_question, test_uuid, test_data,role='아빠')
    print(response)
