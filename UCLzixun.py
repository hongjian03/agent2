import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI

# 初始化会话状态变量
if 'prompt_template' not in st.session_state:
    st.session_state.prompt_template = """
    你是一位专业的 UCL 大学留学顾问，你的主要任务是帮助学生选择适合他们的 UCL 专业。
    请基于以下信息给出专业建议：
    
    1. 学生个人情况：{student_info}
    
    2. 你可以使用以下工具搜索知识库获取相关信息：
    - 搜索历年录取数据
    - 查询专业录取要求
    
    请遵循以下步骤：
    1. 分析学生的背景、成绩和兴趣
    2. 查询符合学生背景的 UCL 专业
    3. 对比多个可选专业的录取难度和匹配度
    4. 给出至少3个推荐专业，并详细说明理由
    5. 提供申请建议
    
    回答须为中文，表达专业且友好。
    """

# 加载知识库
@st.cache_resource
def create_vector_db():
    try:
        # 创建向量数据库，这里使用示例数据目录
        data_dir = "./knowledge_base"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            # 创建示例文件
            with open(f"{data_dir}/sample_data.txt", "w") as f:
                f.write("UCL Computer Science 专业往年录取平均分数：托福总分不低于92分，雅思不低于6.5分。GPA要求3.3/4.0以上。")
                f.write("UCL Economics 专业往年录取要求学生有较强的数学背景，通常需要GPA 3.5以上，雅思不低于7.0。")
        
        # 加载文档
        loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        # 分割文档
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # 创建向量数据库
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(documents=texts, embedding=embeddings)
        
        return vector_db
    except Exception as e:
        st.error(f"创建知识库时出错: {str(e)}")
        return None

# 创建搜索工具
def create_search_tools(vector_db):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    def search_admission_data(query):
        """搜索历年录取数据"""
        results = retriever.get_relevant_documents(f"录取数据: {query}")
        return "\n".join([doc.page_content for doc in results])
    
    def search_program_requirements(query):
        """查询专业录取要求"""
        results = retriever.get_relevant_documents(f"录取要求: {query}")
        return "\n".join([doc.page_content for doc in results])
    
    tools = [
        Tool(
            name="搜索历年录取数据",
            func=search_admission_data,
            description="用于搜索UCL各专业历年录取的学生分数和背景信息"
        ),
        Tool(
            name="查询专业录取要求",
            func=search_program_requirements,
            description="用于查询UCL各专业的具体录取要求和标准"
        )
    ]
    
    return tools

# 创建Agent
def create_ucl_advisor_agent(prompt_template, tools):
    try:
        # 使用OpenRouter作为LLM
        llm = ChatOpenAI(
            temperature=0.7,
            model=st.secrets["OPENROUTER_MODEL"],
            api_key=st.secrets["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            streaming=True
        )
        
        # 创建提示词模板
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["student_info"]
        )
        
        # 创建Agent
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        
        # 创建Agent执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        return agent_executor
    except Exception as e:
        st.error(f"创建Agent时出错: {str(e)}")
        return None

# 主应用
def main():
    st.title("UCL 专业选择顾问")
    
    # 创建知识库和工具
    vector_db = create_vector_db()
    if vector_db is None:
        st.error("无法创建知识库，请检查日志并重试")
        return
    
    tools = create_search_tools(vector_db)
    
    # 创建标签页
    tab1, tab2 = st.tabs(["专业顾问", "提示词设置"])
    
    with tab1:
        st.header("UCL专业选择分析")
        
        student_info = st.text_area(
            "请输入您的信息（包括教育背景、GPA、语言成绩、兴趣方向等）：",
            height=200,
            placeholder="例如：我是计算机专业的本科生，GPA 3.6/4.0，托福总分98（阅读25，听力26，口语23，写作24）。对人工智能和数据科学比较感兴趣，希望申请UCL的相关专业。"
        )
        
        if st.button("分析适合的专业"):
            if not student_info:
                st.warning("请先输入您的信息")
                return
            
            with st.spinner("正在分析中，请稍候..."):
                try:
                    # 使用当前session state中的提示词模板创建agent
                    agent_executor = create_ucl_advisor_agent(st.session_state.prompt_template, tools)
                    
                    if agent_executor:
                        # 执行分析
                        response = agent_executor.invoke({"student_info": student_info})
                        
                        # 显示结果
                        st.subheader("分析结果")
                        st.markdown(response["output"])
                        st.subheader("分析过程")
                        for step in response["intermediate_steps"]:
                            st.write(f"工具: {step[0].tool}")
                            st.write(f"输入: {step[0].tool_input}")
                            st.write(f"结果: {step[1]}")
                            st.write("---")
                    else:
                        st.error("无法创建顾问Agent，请检查设置后重试")
                except Exception as e:
                    st.error(f"分析过程中出错: {str(e)}")
    
    with tab2:
        st.header("提示词模板设置")
        
        new_template = st.text_area(
            "自定义提示词模板",
            value=st.session_state.prompt_template,
            height=400
        )
        
        if st.button("更新提示词"):
            st.session_state.prompt_template = new_template
            st.success("提示词模板已更新！")
            st.info("请返回'专业顾问'标签页，使用新的提示词进行分析。")

if __name__ == "__main__":
    main()
