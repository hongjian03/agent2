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
        # 初始化默认模板
        self.default_templates = {
            'consultant_role': """
            你是一位资深咨询顾问，擅长分析各类文档和材料，提供专业的见解和建议。
            你的主要职责是仔细阅读提供的文档，提取关键信息，分析问题，并提供具体的解决方案和建议。
            """,
            
            'consultant_task': """
            请分析以下文档内容：
            1. 提取文档中的关键信息和重点
            2. 分析存在的主要问题和挑战
            3. 提供具体的解决方案和建议
            4. 给出可行的执行步骤和建议
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
            SystemMessage(content=self.prompt_templates.get_template('consultant_role')),
            HumanMessage(content="{task}\n\n{document_content}")
        ])
        
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=consultant_prompt,
            output_key="analysis_result",
            verbose=True
        )

    def process(self, document_content: str, callback=None) -> Dict[str, Any]:
        try:
            # 准备输入
            chain_input = {
                "document_content": document_content,
                "task": self.prompt_templates.get_template('consultant_task')
            }
            
            # 执行分析
            result = self.analysis_chain(
                chain_input,
                callbacks=[callback] if callback else None
            )
            
            return {
                "status": "success",
                "analysis_result": result["analysis_result"]
            }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

def main():
    st.set_page_config(page_title="咨询脑暴助理", layout="wide")
    
    if 'prompt_templates' not in st.session_state:
        st.session_state.prompt_templates = PromptTemplates()
    
    tab1, tab2 = st.tabs(["咨询脑暴助理", "提示词设置"])
    
    with tab1:
        st.title("咨询脑暴助理")
        
        document_content = st.text_area(
            "请输入需要分析的文档内容",
            height=300,
            placeholder="请输入需要分析的文档内容..."
        )
        
        if st.button("开始分析", key="start_analysis"):
            if document_content:
                try:
                    agent = BrainstormingAgent(
                        api_key=st.secrets["OPENROUTER_API_KEY"],
                        prompt_templates=st.session_state.prompt_templates
                    )
                    
                    with st.spinner("正在分析文档..."):
                        st.subheader("🤔 分析过程")
                        with st.expander("查看详细分析过程", expanded=True):
                            callback = StreamlitCallbackHandler(st.container())
                            result = agent.process(document_content, callback=callback)
                            
                            if result["status"] == "success":
                                # 显示分析结果
                                st.markdown("### 📊 分析结果")
                                st.markdown(result["analysis_result"])
                            else:
                                st.error(f"处理失败: {result.get('message', '未知错误')}")
                            
                except Exception as e:
                    st.error(f"处理过程中出错: {str(e)}")
            else:
                st.warning("请先输入文档内容")
    
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
                prompt_templates.update_template('consultant_task', consultant_task)
                st.success("✅ 提示词已更新！")
        
        with col2:
            if st.button("重置为默认提示词", key="reset_prompts"):
                prompt_templates.reset_to_default()
                st.rerun()

if __name__ == "__main__":
    main()
