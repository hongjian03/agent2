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
from demo import DEMO1, DEMO2
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
        # 定义示例数据作为字符串
        self.demo1 = DEMO1
        self.demo2 = DEMO2

        self.default_templates = {
            'consultant_role': """
            你是一位资深的咨询顾问培训师，擅长分析咨询顾问与客户的沟通过程，提供专业的沟通技巧改进建议。
            你需要基于沟通目的，分析沟通过程中的优缺点，并提供具体的改进建议。
            """,
            
            'output_format': """
            请按照以下格式输出分析结果：
            
            """,
            
            'consultant_task': """
            以下是两个优秀案例的分析示例：

            示例1：
            {demo1}

            示例2：
            {demo2}

            请根据以上示例的分析方式，分析下面的案例：
            基于提供的沟通目的：{communication_purpose}
            沟通记录：{document_content}
            
            请分析这份咨询顾问与学生的沟通记录，完成以下分析：
            
            1. 沟通要点分析：
            - 根据沟通目的，列出本次沟通应该关注的关键要点
            - 评估这些要点在实际沟通中是否得到了充分的覆盖
            
            2. 沟通过程细项分析：
            - 开场与关系建立
            - 信息获取的完整性
            - 问题解答的专业性
            - 情绪管理与共情能力
            - 总体节奏把控
            
            3. 改进建议：
            - 沟通文稿的具体优化建议
            - 给咨询顾问的具体提升建议
            - 下次沟通的重点关注事项
            
            请分点陈述，给出具体的例子和建议。
            """
        }
        
        # 初始化 session_state 中的模板
        if 'templates' not in st.session_state:
            st.session_state.templates = self.default_templates.copy()

    def get_template(self, template_name: str) -> str:
        return st.session_state.templates.get(template_name, "")

    def update_template(self, template_name: str, new_content: str) -> None:
        st.session_state.templates[template_name] = new_content

    def reset_to_default(self):
        st.session_state.templates = self.default_templates.copy()

class BrainstormingAgent:
    def __init__(self, api_key: str, prompt_templates: PromptTemplates):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model=st.secrets["OPENROUTER_MODEL"],
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.prompt_templates = prompt_templates
        self.setup_chain()

    def setup_chain(self):
        # 创建咨询分析链
        consultant_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_templates.get_template('consultant_role')),
            ("system", self.prompt_templates.get_template('output_format')),
            ("human", self.prompt_templates.get_template('consultant_task'))
        ]).partial(
            demo1=self.demo1,
            demo2=self.demo2
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
                "task": self.prompt_templates.get_template('consultant_task')
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
        
        if st.button("开始分析", key="start_analysis"):
            if document_content and communication_purpose:
                try:
                    agent = BrainstormingAgent(
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
        
        prompt_templates = st.session_state.prompt_templates
        
        # 咨询顾问设置
        st.subheader("咨询顾问设置")
        consultant_role = st.text_area(
            "角色设定",
            value=prompt_templates.get_template('consultant_role'),
            height=200,
            key="consultant_role"
        )
        
        output_format = st.text_area(
            "输出格式",
            value=prompt_templates.get_template('output_format'),
            height=200,
            key="output_format"
        )
        
        consultant_task = st.text_area(
            "任务说明",
            value=prompt_templates.get_template('consultant_task'),
            height=200,
            key="consultant_task"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("更新提示词", key="update_prompts"):
                prompt_templates.update_template('consultant_role', consultant_role)
                prompt_templates.update_template('output_format', output_format)
                prompt_templates.update_template('consultant_task', consultant_task)
                st.success("✅ 提示词已更新！")
        
        with col2:
            if st.button("重置为默认提示词", key="reset_prompts"):
                prompt_templates.reset_to_default()
                st.rerun()

if __name__ == "__main__":
    main()
