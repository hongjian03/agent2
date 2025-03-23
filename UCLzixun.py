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
import pandas as pd
from typing import List
from langchain.docstore.document import Document

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

def excel_to_documents(excel_path: str) -> List[Document]:
    """将Excel数据转换为Document格式"""
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        documents = []
        
        # 将每行数据转换为结构化文本
        for _, row in df.iterrows():
            # 假设Excel中有这些列：专业名称、GPA、语言成绩、录取结果等
            content = f"""
            专业: {row['专业名称']}
            录取要求:
            - GPA: {row['GPA']}
            - 语言成绩: {row['语言成绩']}
            录取结果: {row['录取结果']}
            其他信息: {row.get('其他信息', '')}
            """
            
            # 创建Document对象
            doc = Document(
                page_content=content,
                metadata={
                    "program": row['专业名称'],
                    "year": row.get('年份', 'Unknown'),
                    "source": "UCL历史录取数据"
                }
            )
            documents.append(doc)
            
        return documents
    except Exception as e:
        raise Exception(f"Excel数据转换错误: {str(e)}")

# 加载知识库
@st.cache_resource
def create_vector_db():
    try:
        # 指定Excel文件路径
        excel_path = "./knowledge_base/ucl_admissions.xlsx"
        
        # 如果Excel文件不存在，创建示例数据
        if not os.path.exists("./knowledge_base"):
            os.makedirs("./knowledge_base")
        
        if not os.path.exists(excel_path):
            # 创建示例Excel数据
            sample_data = {
                '专业名称': ['Computer Science', 'Economics', 'Data Science'],
                'GPA': ['3.3/4.0', '3.5/4.0', '3.4/4.0'],
                '语言成绩': ['托福92/雅思6.5', '雅思7.0', '托福95/雅思6.5'],
                '录取结果': ['Conditional Offer', 'Conditional Offer', 'Conditional Offer'],
                '其他信息': ['需要较强编程背景', '需要较强数学背景', '需要数据分析经验']
            }
            df = pd.DataFrame(sample_data)
            df.to_excel(excel_path, index=False)
        
        # 将Excel转换为documents
        documents = excel_to_documents(excel_path)
        
        # 分割文档（如果需要的话）
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        texts = text_splitter.split_documents(documents)
        
        # 创建向量数据库
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_db = FAISS.from_documents(
            documents=texts,
            embedding=embeddings
        )
        
        return vector_db
    except Exception as e:
        st.error(f"创建知识库时出错: {str(e)}")
        return None

# 创建搜索工具
def create_search_tools(vector_db):
    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 3,
            "search_type": "mmr",  # 使用最大边际相关性搜索
            "fetch_k": 10  # 初始获取的文档数量
        }
    )
    
    def search_admission_data(query: str) -> str:
        """搜索历年录取数据"""
        try:
            results = retriever.get_relevant_documents(f"录取数据: {query}")
            formatted_results = []
            
            for doc in results:
                formatted_results.append(doc.page_content.strip())
            
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"搜索过程中出现错误: {str(e)}"
    
    def search_program_requirements(query: str) -> str:
        """查询专业录取要求"""
        try:
            results = retriever.get_relevant_documents(f"录取要求: {query}")
            formatted_results = []
            
            for doc in results:
                formatted_results.append(doc.page_content.strip())
            
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"搜索过程中出现错误: {str(e)}"
    
    tools = [
        Tool(
            name="搜索历年录取数据",
            func=search_admission_data,
            description="用于搜索UCL各专业历年录取的学生分数和背景信息，输入专业名称或具体查询条件"
        ),
        Tool(
            name="查询专业录取要求",
            func=search_program_requirements,
            description="用于查询UCL各专业的具体录取要求和标准，输入专业名称即可"
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
