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


        self.default_templates = {
            'consultant_role1': """
            # 角色
            你是资深留学顾问，精通学生背景分析和各国院校招生政策。
            """,
            
            'output_format1': """
            学生背景分析: 
                核心亮点: 亮点1，亮点2，亮点3...
                需要加强的方面: 需要加强的方面1，需要加强的方面2...
            申请策略: 
                国家与专业分析: 对目标国家和专业招生偏好的简要分析
                推荐写作方向: 方向1，方向2...
                核心卖点: 如何突出学生的优势并与专业匹配
            """,
            
            'consultant_task1': """
            根据选校方案先判断是否已选校，如果已选校，则结合选校方案进行后续分析
            分析学生的个人陈述表，提取关键信息与亮点
            根据申请国家和专业确定PS的写作大方向
            评估学生背景与目标专业的匹配度
            制定个性化文书策略，确定核心卖点
            """,
            
            'consultant_role2': """
            # 角色
            你是结构化思维与创意写作专家，擅长内容规划和素材创作。
            """,
            
            'output_format2': """
            文书框架: 
                整体结构: 文书整体结构概述
                段落规划: 
                    段落目的: 这段要达成的目标
                    核心内容: 应包含的关键信息
                    素材建议: 
                    需要补充的内容: 具体需要补充什么类型的素材
                    补充例子: 具体的素材示例
                    与专业关联: 如何将此素材与申请专业关联
                其他段落: 其他段落规划
            """,
            
            'consultant_task2': """
            根据选校方案先判断是否已选校，如果已选校，则结合选校方案进行后续分析
            设计PS的整体框架和段落结构
            为每个段落规划内容要点和与专业的关联
            直接提供具体素材补充建议和实例
            确保补充素材与学生背景一致且符合申请专业需求
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
            base_url="https://openrouter.ai/api/v1",
            streaming=True
        )
        self.prompt_templates = prompt_templates
        self.setup_chains()

    def setup_chains(self):
        # Profile Strategist Chain
        strategist_prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.prompt_templates.get_template('consultant_role1')}\n\n"
                      f"任务:\n{self.prompt_templates.get_template('consultant_task1')}\n\n"
                      f"请按照以下格式输出:\n{self.prompt_templates.get_template('output_format1')}"),
            ("human", "选校方案：\n{school_plan}\n\n"
             "请分析以下学生个人陈述：\n\n"
             "个人陈述：\n{document_content}\n\n")
        ])
        
        self.strategist_chain = LLMChain(
            llm=self.llm,
            prompt=strategist_prompt,
            output_key="strategist_analysis",
            verbose=True
        )

        # Content Creator Chain
        creator_prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.prompt_templates.get_template('consultant_role2')}\n\n"
                      f"任务:\n{self.prompt_templates.get_template('consultant_task2')}\n\n"
                      f"请按照以下格式输出:\n{self.prompt_templates.get_template('output_format2')}"),
            ("human", "选校方案：\n{school_plan}\n\n"
                     "基于第一阶段的分析结果：\n{strategist_analysis}\n\n请创建详细的内容规划。")
        ])
        
        self.creator_chain = LLMChain(
            llm=self.llm,
            prompt=creator_prompt,
            output_key="creator_output",
            verbose=True
        )

    def process(self, document_content: str, school_plan: str, callback=None) -> Dict[str, Any]:
        try:
            
            
            # Run Profile Strategist
            strategist_result = self.strategist_chain(
                {"document_content": document_content, 
                 "school_plan": school_plan
                 },
                callbacks=[callback] if callback else None
            )
            
            # Run Content Creator
            creator_result = self.creator_chain(
                {
                    "strategist_analysis": strategist_result["strategist_analysis"],
                    "school_plan": school_plan
                },
                callbacks=[callback] if callback else None
            )
            
            logger.info("Analysis completed successfully")
            return {
                "status": "success",
                "strategist_analysis": strategist_result["strategist_analysis"],
                "creator_output": creator_result["creator_output"]
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
    """读取 Word 文档内容，包括表格，并去除重复内容"""
    try:
        doc = Document(io.BytesIO(file_bytes))
        content_set = set()  # 用于存储已处理的内容，避免重复
        full_text = []
        
        # 读取普通段落
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text and text not in content_set:  # 只添加非空且未重复的内容
                content_set.add(text)
                full_text.append(text)
        
        # 读取表格内容
        for table in doc.tables:
            table_content = []
            header_row = []
            
            # 获取表头（第一行）
            if table.rows:
                for cell in table.rows[0].cells:
                    header_text = cell.text.strip()
                    if header_text:
                        header_row.append(header_text)
            
            # 处理表格内容（从第二行开始）
            for row_idx, row in enumerate(table.rows[1:], 1):
                row_content = {}
                for col_idx, cell in enumerate(row.cells):
                    if col_idx < len(header_row):  # 确保有对应的表头
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_content[header_row[col_idx]] = cell_text
                
                if row_content:  # 只添加非空行
                    formatted_row = " | ".join([f"{header}: {value}" 
                                              for header, value in row_content.items()])
                    if formatted_row not in content_set:  # 避免重复内容
                        content_set.add(formatted_row)
                        table_content.append(formatted_row)
            
            if table_content:
                full_text.extend(table_content)
        
        # 使用换行符连接所有文本
        result = "\n".join(full_text)
        logger.info(f"成功读取文档内容，包含 {len(doc.tables)} 个表格")
        return result
    except Exception as e:
        logger.error(f"读取 Word 文档时出错: {str(e)}")
        return None

def main():
    langsmith_api_key = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = "初稿脑暴平台"
    st.set_page_config(page_title="初稿脑暴助理平台", layout="wide")
    add_custom_css()
    st.markdown("<h1 class='page-title'>初稿脑暴助理</h1>", unsafe_allow_html=True)
    
    if 'prompt_templates' not in st.session_state:
        st.session_state.prompt_templates = PromptTemplates()
    
    tab1, tab2 = st.tabs(["初稿脑暴助理", "提示词设置"])
    st.markdown(f"<div class='model-info'>🤖 当前使用模型: <b>{st.secrets['OPENROUTER_MODEL']}</b></div>", unsafe_allow_html=True)
    
    with tab1:
        st.title("初稿脑暴助理")
        
        # 添加文件上传功能
        uploaded_file = st.file_uploader("上传初稿文档", type=['docx'])
        
        # 添加选校方案输入框
        school_plan = st.text_area(
            "选校方案",
            value="暂未选校",
            height=100,
            help="请输入已确定的选校方案，包括学校和专业信息"
        )
        
        # 处理上传的文件
        if uploaded_file is not None:
            document_content = read_docx(uploaded_file.read())
            if document_content:
                st.success("个人陈述上传成功！")
                with st.expander("查看个人陈述内容", expanded=False):
                    st.write(document_content)
            else:
                st.error("无法读取文档内容，请检查文件格式是否正确。")
        
        if st.button("开始分析", key="start_analysis"):
            if document_content:
                try:
                    agent = BrainstormingAgent(
                        api_key=st.secrets["OPENROUTER_API_KEY"],
                        prompt_templates=st.session_state.prompt_templates
                    )
                    
                    # 创建两个独立的expander来显示分析过程和结果
                    analysis_expander = st.expander("分析过程", expanded=True)
                    results_expander = st.expander("分析结果", expanded=True)
                    
                    with analysis_expander:
                        st.subheader("🤔 分析过程")
                        callback = StreamlitCallbackHandler(st.container())
                    
                    with results_expander:
                        # 第一阶段：背景分析
                        st.subheader("📊 第一阶段：背景分析")
                        
                        # 创建空容器用于流式输出
                        analysis_container = st.empty()
                        
                        with st.spinner("正在进行背景分析..."):
                            # 创建一个变量来存储累积的输出
                            accumulated_text = ""
                            
                            def streaming_callback(text):
                                nonlocal accumulated_text
                                accumulated_text += text
                                # 使用 markdown 格式更新容器内容
                                analysis_container.markdown(f"```json\n{accumulated_text}\n```")
                            
                            # 创建自定义回调处理器
                            class StreamingCallbackHandler(StreamlitCallbackHandler):
                                def on_llm_new_token(self, token: str, **kwargs) -> None:
                                    streaming_callback(token)
                            
                            strategist_result = agent.strategist_chain(
                                {
                                    "document_content": document_content,
                                    "school_plan": school_plan
                                },
                                callbacks=[StreamingCallbackHandler()]
                            )
                            st.success("✅ 背景分析完成！")
                        
                        # 添加一个分隔线
                        st.markdown("---")
                        
                        # 添加继续按钮
                        if st.button("继续进行内容规划", key="continue_to_planning"):
                            # 第二阶段：内容规划
                            st.subheader("📝 第二阶段：内容规划")
                            
                            # 创建新的空容器用于第二阶段的流式输出
                            planning_container = st.empty()
                            accumulated_text = ""
                            
                            with st.spinner("正在进行内容规划..."):
                                creator_result = agent.creator_chain(
                                    {
                                        "strategist_analysis": strategist_result["strategist_analysis"],
                                        "school_plan": school_plan
                                    },
                                    callbacks=[StreamingCallbackHandler()]
                                )
                                st.success("✅ 内容规划完成！")
                
                except Exception as e:
                    st.error(f"处理过程中出错: {str(e)}")
            else:
                st.warning("请先上传初稿文档")
    
    with tab2:
        st.title("提示词设置")
        
        prompt_templates = st.session_state.prompt_templates
        
        # Agent 1 设置
        st.subheader("Agent 1 - 档案策略师")
        consultant_role1 = st.text_area(
            "角色设定",
            value=prompt_templates.get_template('consultant_role1'),
            height=200,
            key="consultant_role1"
        )
        
        output_format1 = st.text_area(
            "输出格式",
            value=prompt_templates.get_template('output_format1'),
            height=200,
            key="output_format1"
        )
        
        consultant_task1 = st.text_area(
            "任务说明",
            value=prompt_templates.get_template('consultant_task1'),
            height=200,
            key="consultant_task1"
        )

        # Agent 2 设置
        st.subheader("Agent 2 - 内容创作师")
        consultant_role2 = st.text_area(
            "角色设定",
            value=prompt_templates.get_template('consultant_role2'),
            height=200,
            key="consultant_role2"
        )
        
        output_format2 = st.text_area(
            "输出格式",
            value=prompt_templates.get_template('output_format2'),
            height=200,
            key="output_format2"
        )
        
        consultant_task2 = st.text_area(
            "任务说明",
            value=prompt_templates.get_template('consultant_task2'),
            height=200,
            key="consultant_task2"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("更新提示词", key="update_prompts"):
                prompt_templates.update_template('consultant_role1', consultant_role1)
                prompt_templates.update_template('output_format1', output_format1)
                prompt_templates.update_template('consultant_task1', consultant_task1)
                prompt_templates.update_template('consultant_role2', consultant_role2)
                prompt_templates.update_template('output_format2', output_format2)
                prompt_templates.update_template('consultant_task2', consultant_task2)
                st.success("✅ 提示词已更新！")
        
        with col2:
            if st.button("重置为默认提示词", key="reset_prompts"):
                prompt_templates.reset_to_default()
                st.rerun()

if __name__ == "__main__":
    main()
