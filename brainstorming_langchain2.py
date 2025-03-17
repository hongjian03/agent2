import streamlit as st
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import SequentialChain, LLMChain
import os
from typing import Dict, Any, List
import logging
import sys
from docx import Document
import io
# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
from demo import DEMO1, DEMO2, DEMO3, DEMO4, DEMO5
#DEMO1 常见问题及话术   DEMO2 三段目标咨询法的结构与底层逻辑   DEMO3 优秀案例分析   DEMO4 优秀案例分析   DEMO5 优秀案例分析

logger = logging.getLogger(__name__)

# 记录程序启动
logger.info("程序开始运行")

# 只在第一次运行时替换 sqlite3
if 'sqlite_setup_done' not in st.session_state:
    try:
        logger.info("尝试设置 SQLite")
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        st.session_state.sqlite_setup_done = True
        logger.info("SQLite 设置成功")
    except Exception as e:
        logger.error(f"SQLite 设置错误: {str(e)}")
        st.session_state.sqlite_setup_done = True


class PromptTemplates:
    def __init__(self):
        # 定义两套模板
        self.demo1 = DEMO1
        self.demo2 = DEMO2
        self.demo3 = DEMO3
        self.demo4 = DEMO4
        self.demo5 = DEMO5

        self.default_templates = {
            # Agent2 的模板
            'agent2': {
                'role2': """
                你是一个专业的对话分析专家，可以区分对话的说话角色是老师还是学生。
                你可以根据对话内容判断对话是否符合三段目标咨询法、热门问题、标准话术，是否流畅，互动频率如何。
                你可以根据分析结果给出评分和改进建议。
                """,
                'output_format2': """
                请按照以下格式输出分析结果：
                1、评分：
                2、改进建议：
                3、评分雷达图：
                """,
                'task2': """
                聊天记录：{document_content}
                1. 根据聊天记录区分哪些是老师说的话，哪些是学生说的话，并且修正笔误。
                2. 理解热门问题和标准话术{demo1}，判断对话是否符合热门问题、标准话术
                3. 理解三段目标咨询法{demo2}，判断对话是否符合三段目标咨询法
                4. 以下案例是标准案例：{demo3}，{demo4}，{demo5}，根据标准案例判断对话是否流畅，互动频率如何
                5. 根据以上分析结果，给出评分和改进建议。以下是评分维度：
                    * 三段式逻辑连贯性
                    * 话术复刻匹配度
                    * 互动引导有效性
                    * 咨询流畅度
                6. 根据评分画出评分雷达图
                """
            },
            # Agent1 的模板
            'agent1': {
                'role': """
                # 角色
                你是你是留学咨询机构的资深老师，负责培训留学顾问。
                你的目标是对顾问的咨询质量进行分析和评价，让顾问明白自己在咨询中哪些方面要继续保持、哪些方面要改进优化。
                """,
                
                'output_format': """
                请按照以下格式输出分析结果：
                输出内容必须是对原文档的修改，输出时必须输出原文档修改后的内容。
                严禁只输出修改部分，必须是整篇文档一起输出。
                对于修改的内容加以标注，标注格式为：
                [原文|修改]

                """,
                
                'task': """
                以下是两个优秀案例的分析示例：

                示例1：
                {demo1}

                示例2：
                {demo2}

                请根据以上示例的分析方式，分析下面的案例：
                基于提供的沟通目的：{communication_purpose}
                沟通记录：{document_content}
                
                ##工作步骤  
                1.根据顾问输入的咨询目的，理解该段咨询应达到的最终结果和重点评估维度。
                2.按细分维度为文档内容进行分析和打分。根据不同的咨询目的，各维度的评分权重可以上下浮动。
                3.在文档的原文基础上做优化提升建议，包括更好的表达方式、可继续追问的话术、可增加的交谈内容等。你的建议需要直接标注在文档的原文段落后面。

                ##限制
                1.做优化提升建议时，你是在原文档内做注释，而不是只引用其中的一句话。你可以理解为你在给原文档做沟通技巧的润色。
                2. 如果文档内的沟通内容已经很优秀了，你不用硬写建议。

                ##可参考的评分维度
                1. 专业力评估
                **细分维度1- 信息框架构建**
                - 是否通过有效提问，让客户提供充足的个人信息（当前学术背景、软性背景、留学预期、家庭背景、预算等）
                - 是否在15分钟内建立清晰咨询结构（背景采集→痛点确认→方向探索）
                - 在咨询过程中，是否根据学生碎片化表达归纳出结构化画像（每准确提炼1项核心特征+3分）
                **细分维度2- 留学专业信息展示**
                - 推荐院校范围或具体的院校，并分析客户背景与院校契合的原因
                - 推荐专业范围或具体的专业，并分析客户背景与专业契合的原因 
                - 展示方式：仅需口头描述客观信息+1分，辅以案例对比+3分
                **细分维度3- 风险预判提示**
                - 主动指出学生背景中的关键短板（如GPA波动/课程匹配度问题）
                - 提示后续可能注意的问题节点（如实习证明时效性）
                **细分维度4- 路径引导力**
                - 是否规划可明确执行的准备动作（如8月前完成GRE首考）
                - 给出学生可自主验证的信息渠道（官方资源链接/自查清单）
                **细分维度5- 答疑准确度**
                - 是否准确理解了客户问题的疑问点、并做了明确的解答
                - 解答内容是否坚定且准确、没有来回多次摇摆
                - 遇到当下无法回答的问题，是否做了合理的解释并约定了以后回应的方式

                2.咨询力评估
                **细分维度1- 共情表达**
                - 使用场景化语言回应焦虑（如"你的情况 我去年的xx学生A也遇到过"）
                - 准确复述客户隐性需求次数（如"你其实更关注专业对进大厂的帮助对吗？"）
                - 肯定客户的优点和优势
                **细分维度2- 提问深度**
                - 提出超越基本背景的洞察性问题（如"为什么特别排斥营销岗？是否有相关负面经历？"）
                - 每轮对话中封闭式问题占比不超过40%（开放式追问≥3次/10分钟）
                **细分维度3- 价值锚点植入**
                - 创造2个以上我司品牌相关的记忆点
                - 创造2个以上我司产品和服务内容相关的记忆点
                - 教学式传递1个以上行业认知（如"金融一级市场对communication skills的需求你可能不知道..."）
                **细分维度4- 临场掌控**
                - 有效阻断客户的无效发散表述（如"这部分我们稍后详谈，先聚焦专业方向"）
                - 突发问题解决时效（如现场回应客户对公司品牌或网络口碑的质疑、客户要求顾问做过度承诺等）
                **细分维度5- 结束对话与近一步邀约**
                - 结束本次对话时，是否约定了下一次沟通的时间和方式
                - 是否与客户约定了下一次沟通的主题和重要事项
                """
            }
        }
        
        # 初始化 session_state 中的模板
        if 'templates' not in st.session_state:
            st.session_state.templates = self.default_templates.copy()

    def get_template(self, agent_type: str, template_name: str) -> str:
        return st.session_state.templates.get(agent_type, {}).get(template_name, "")

    def update_template(self, agent_type: str, template_name: str, new_content: str) -> None:
        if agent_type not in st.session_state.templates:
            st.session_state.templates[agent_type] = {}
        st.session_state.templates[agent_type][template_name] = new_content

    def reset_to_default(self, agent_type: str):
        st.session_state.templates[agent_type] = self.default_templates[agent_type].copy()

class BrainstormingAgent:
    def __init__(self, api_key: str, prompt_templates: PromptTemplates):
        self.llm = ChatOpenAI(
            streaming=True,
            temperature=0.7,
            model=st.secrets["OPENROUTER_MODEL"],
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.prompt_templates = prompt_templates
        # 保存示例数据的引用
        self.demo3 = prompt_templates.demo3
        self.demo4 = prompt_templates.demo4
        self.demo5 = prompt_templates.demo5
        self.setup_chain()

    def setup_chain(self):
        # 创建咨询分析链
        consultant_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_templates.get_template('agent1', 'role')),
            ("human", self.prompt_templates.get_template('agent1', 'task')),
            ("system", self.prompt_templates.get_template('agent1', 'output_format'))
        ]).partial(
            demo1=self.demo3,
            demo2=self.demo4,
            demo3=self.demo5
        )
        
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=consultant_prompt,
            output_key="analysis_result",
            verbose=True
        )

    def process(self, document_content: str, communication_purpose: str, callback=None) -> Dict[str, Any]:
        try:
            logger.info(f"Processing document with purpose: {communication_purpose[:100]}...")
            logger.info(f"Document content: {document_content[:100]}...")
            
            # 准备输入
            chain_input = {
                "document_content": document_content,
                "communication_purpose": communication_purpose,
            }
            
            # 执行分析
            result = self.analysis_chain(
                chain_input,
                callbacks=[callback] if callback else None
            )
            
            logger.info("Analysis completed successfully")
            return {
                "status": "success",
                "analysis_result": result["analysis_result"]
            }
                
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

class Agent2:
    def __init__(self, api_key: str, prompt_templates: PromptTemplates):
        self.llm = ChatOpenAI(
            streaming=True,
            temperature=0.7,
            model=st.secrets["OPENROUTER_MODEL"],
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.prompt_templates = prompt_templates
        self.demo1 = prompt_templates.demo1
        self.demo2 = prompt_templates.demo2
        self.demo3 = prompt_templates.demo3
        self.demo4 = prompt_templates.demo4
        self.demo5 = prompt_templates.demo5
        self.setup_chain()

    def setup_chain(self):
        # 创建三段目标咨询分析链
        analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_templates.get_template('agent2', 'role2')),
            ("human", self.prompt_templates.get_template('agent2', 'task2')),
            ("system", self.prompt_templates.get_template('agent2', 'output_format2'))
        ]).partial(
            demo1=self.demo1,
            demo2=self.demo2,
            demo3=self.demo3,
            demo4=self.demo4,
            demo5=self.demo5
        )
        
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=analyzer_prompt,
            output_key="analysis_result",
            verbose=True
        )

    def process(self, document_content: str, communication_purpose: str, callback=None) -> Dict[str, Any]:
        try:
            logger.info("开始三段目标咨询法分析...")
            logger.info(f"文档内容前100字符: {document_content[:100]}...")
            
            # 准备输入
            chain_input = {
                "document_content": document_content,
            }
            
            # 执行分析
            result = self.analysis_chain(
                chain_input,
                callbacks=[callback] if callback else None
            )
            
            logger.info("三段目标咨询法分析完成")
            return {
                "status": "success",
                "analysis_result": result["analysis_result"]
            }
                
        except Exception as e:
            logger.error(f"三段目标咨询法分析过程中出错: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

def add_custom_css():
    st.markdown("""
    <style>
    /* 标题样式 */
    h1, h2, h3 {
        color: #1e3a8a;
        font-weight: 600;
    }
    
    /* 卡片样式 */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-top: 10px;
    }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #2e4a9a;
    }
    
    /* 输入框样式 */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    
    /* 文件上传区域样式 */
    .stFileUploader>div>button {
        background-color: #f1f3f9;
        color: #1e3a8a;
        border: 1px dashed #1e3a8a;
        border-radius: 5px;
    }
    
    /* 成功消息样式 */
    .stSuccess {
        background-color: #d1fae5;
        color: #065f46;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* 警告消息样式 */
    .stWarning {
        background-color: #fef3c7;
        color: #92400e;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* 错误消息样式 */
    .stError {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* 下拉选择框样式 */
    .stSelectbox>div>div {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    
    /* 页面标题样式 */
    .page-title {
        text-align: center;
        font-size: 2rem;
        margin-bottom: 20px;
        color: #1e3a8a;
        font-weight: bold;
    }
    
    /* 卡片容器样式 */
    .card-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 20px;
        width: 100%;
    }
    
    /* 分隔线样式 */
    hr {
        margin-top: 20px;
        margin-bottom: 20px;
        border: 0;
        border-top: 1px solid #eee;
    }
    
    /* 模型信息样式 */
    .model-info {
        background-color: #f0f7ff;
        padding: 8px 12px;
        border-radius: 5px;
        margin-top: 10px;
        margin-bottom: 15px;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    /* 表格样式优化 */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
    }
    
    .dataframe th {
        background-color: #f1f3f9;
        padding: 8px;
    }
    
    .dataframe td {
        padding: 8px;
        border-bottom: 1px solid #eee;
    }
    
    
    
    /* 调整列宽度 */
    .column-adjust {
        padding: 0 5px !important;
    }
    
    /* 强制展开器内容宽度 */
    .streamlit-expanderContent {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)


def read_docx(file_bytes):
    """读取 Word 文档内容"""
    try:
        doc = Document(io.BytesIO(file_bytes))
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # 只添加非空段落
                full_text.append(paragraph.text)
        return "\n".join(full_text)
    except Exception as e:
        logger.error(f"读取 Word 文档时出错: {str(e)}")
        return None

def main():
    langsmith_api_key = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = "咨询脑暴平台"
    st.set_page_config(page_title="咨询脑暴平台", layout="wide")
    add_custom_css()
    st.markdown("<h1 class='page-title'>咨询脑暴平台</h1>", unsafe_allow_html=True)
    
    if 'prompt_templates' not in st.session_state:
        st.session_state.prompt_templates = PromptTemplates()
    
    tab1, tab2 = st.tabs(["咨询沟通分析助理", "提示词设置"])
    st.markdown(f"<div class='model-info'>🤖 当前使用模型: <b>{st.secrets['OPENROUTER_MODEL']}</b></div>", unsafe_allow_html=True)
    
    with tab1:
        st.title("咨询沟通分析助理")
        
        # 添加文件上传功能
        uploaded_file = st.file_uploader("上传咨询沟通记录文档", type=['docx'])
        
        # 沟通目的输入框
        communication_purpose = st.text_area(
            "请输入本次沟通的目的",
            height=100,
            placeholder="例如：了解学生的学业背景和留学意向，确认是否适合申请英国硕士项目..."
        )
        
        # 处理上传的文件
        if uploaded_file is not None:
            document_content = read_docx(uploaded_file.read())
            if document_content:
                st.success("沟通记录上传成功！")
                with st.expander("查看沟通记录内容", expanded=False):
                    st.write(document_content)
            else:
                st.error("无法读取文档内容，请检查文件格式是否正确。")
        # 添加 agent 选择

        agent_type = st.selectbox(
            "选择分析模式",
            ["常规留学咨询", "三段目标咨询法"],
            help="选择不同的分析模式将使用不同的评估标准"
        )
        if st.button("开始分析", key="start_analysis"):
            if document_content :
                try:
                    # 根据选择创建对应的 agent
                    if agent_type == "常规留学咨询":
                        agent = BrainstormingAgent(
                            api_key=st.secrets["OPENROUTER_API_KEY"],
                            prompt_templates=st.session_state.prompt_templates
                        )
                    else:
                        agent = Agent2(
                            api_key=st.secrets["OPENROUTER_API_KEY"],
                            prompt_templates=st.session_state.prompt_templates
                        )
                    
                    with st.spinner("正在分析沟通记录..."):
                        st.subheader("🤔 分析过程")
                        with st.expander("查看详细分析过程", expanded=True):
                            callback = StreamlitCallbackHandler(st.container())
                            # 将沟通目的添加到处理参数中
                            result = agent.process(
                                document_content, 
                                communication_purpose=communication_purpose,
                                callback=callback
                            )
                            
                            if result["status"] == "success":
                                st.markdown("### 📊 分析结果")
                                st.markdown(result["analysis_result"])
                            else:
                                st.error(f"处理失败: {result.get('message', '未知错误')}")
                            
                except Exception as e:
                    st.error(f"处理过程中出错: {str(e)}")
            else:
                if not document_content:
                    st.warning("请先上传沟通记录文档")
                if not communication_purpose:
                    st.warning("请输入本次沟通的目的")
    
    with tab2:
        st.title("提示词设置")
        
        # 创建两个子标签页
        agent1_tab, agent2_tab = st.tabs(["常规留学咨询设置", "三段目标咨询法设置"])
        
        with agent1_tab:
            st.subheader("常规留学咨询提示词设置")
            consultant_role = st.text_area(
                "角色设定",
                value=st.session_state.prompt_templates.get_template('agent1', 'role'),
                height=200,
                key="consultant_role"
            )
            
            output_format = st.text_area(
                "输出格式",
                value=st.session_state.prompt_templates.get_template('agent1', 'output_format'),
                height=200,
                key="output_format"
            )
            
            consultant_task = st.text_area(
                "任务说明",
                value=st.session_state.prompt_templates.get_template('agent1', 'task'),
                height=200,
                key="consultant_task"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("更新提示词", key="update_agent1"):
                    st.session_state.prompt_templates.update_template('agent1', 'role', consultant_role)
                    st.session_state.prompt_templates.update_template('agent1', 'output_format', output_format)
                    st.session_state.prompt_templates.update_template('agent1', 'task', consultant_task)
                    st.success("✅ Agent1提示词已更新！")
            
            with col2:
                if st.button("重置为默认提示词", key="reset_agent1"):
                    st.session_state.prompt_templates.reset_to_default('agent1')
                    st.rerun()
        
        with agent2_tab:
            st.subheader("三段目标咨询法提示词设置")
            analyzer_role = st.text_area(
                "角色设定",
                value=st.session_state.prompt_templates.get_template('agent2', 'role2'),
                height=200,
                key="analyzer_role"
            )
            
            analyzer_format = st.text_area(
                "输出格式",
                value=st.session_state.prompt_templates.get_template('agent2', 'output_format2'),
                height=200,
                key="analyzer_format"
            )
            
            analyzer_task = st.text_area(
                "任务说明",
                value=st.session_state.prompt_templates.get_template('agent2', 'task2'),
                height=200,
                key="analyzer_task"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("更新提示词", key="update_agent2"):
                    st.session_state.prompt_templates.update_template('agent2', 'role2', analyzer_role)
                    st.session_state.prompt_templates.update_template('agent2', 'output_format2', analyzer_format)
                    st.session_state.prompt_templates.update_template('agent2', 'task2', analyzer_task)
                    st.success("✅ Agent2提示词已更新！")
            
            with col2:
                if st.button("重置为默认提示词", key="reset_agent2"):
                    st.session_state.prompt_templates.reset_to_default('agent2')
                    st.rerun()

if __name__ == "__main__":
    main()
