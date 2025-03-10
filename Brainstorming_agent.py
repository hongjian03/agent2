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
            'profile_strategist_role': """
            你是一位资深留学顾问ProfileStrategist，精通学生背景分析和各国院校招生政策。
            你的主要职责是分析学生的个人陈述，提取关键信息与亮点，并制定个性化的文书策略。
            """,
            
            'profile_strategist_task': """
            请分析以下学生信息：
            1. 提取关键信息与亮点
            2. 根据申请国家和专业确定PS的写作大方向
            3. 评估学生背景与目标专业的匹配度
            4. 制定个性化文书策略，确定核心卖点
            """,
            
            'content_creator_role': """
            你是一位结构化思维与创意写作专家ContentCreator，擅长内容规划和素材创作。
            你需要基于学生背景分析报告，设计个性化的文书框架和内容规划。
            """,
            
            'content_creator_task': """
            请基于以下背景分析报告，完成：
            1. 设计PS的整体框架和段落结构
            2. 为每个段落规划内容要点和与专业的关联
            3. 提供具体素材补充建议和实例
            4. 确保补充素材与学生背景一致且符合申请专业需求
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
        self.setup_chains()

    def setup_chains(self):
        # 创建 ProfileStrategist 链
        profile_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.prompt_templates.get_template('profile_strategist_role')),
            HumanMessage(content="{task}\n\n{student_info}")
        ])
        
        self.profile_chain = LLMChain(
            llm=self.llm,
            prompt=profile_prompt,
            output_key="profile_analysis",
            verbose=True
        )

        # 创建 ContentCreator 链
        content_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.prompt_templates.get_template('content_creator_role')),
            HumanMessage(content="{task}\n\n{profile_analysis}")
        ])
        
        self.content_chain = LLMChain(
            llm=self.llm,
            prompt=content_prompt,
            output_key="content_plan",
            verbose=True
        )

        # 创建顺序链
        self.brainstorming_chain = SequentialChain(
            chains=[self.profile_chain, self.content_chain],
            input_variables=["student_info", "task"],
            output_variables=["profile_analysis", "content_plan"],
            verbose=True
        )

    def process(self, student_info: str, callback=None) -> Dict[str, Any]:
        try:
            # 准备输入
            chain_input = {
                "student_info": student_info,
                "task": self.prompt_templates.get_template('profile_strategist_task')
            }
            
            # 执行链式处理
            results = self.brainstorming_chain(
                chain_input,
                callbacks=[callback] if callback else None
            )
            
            return {
                "status": "success",
                "profile_analysis": results["profile_analysis"],
                "content_plan": results["content_plan"]
            }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

def main():
    st.set_page_config(page_title="初稿脑暴助理", layout="wide")
    
    if 'prompt_templates' not in st.session_state:
        st.session_state.prompt_templates = PromptTemplates()
    
    tab1, tab2 = st.tabs(["初稿脑暴助理", "提示词设置"])
    
    with tab1:
        st.title("初稿脑暴助理")
        
        student_info = st.text_area(
            "请输入学生信息",
            height=300,
            placeholder="请输入学生的背景信息、申请目标等..."
        )
        
        if st.button("开始分析", key="start_analysis"):
            if student_info:
                try:
                    agent = BrainstormingAgent(
                        api_key=st.secrets["OPENROUTER_API_KEY"],
                        prompt_templates=st.session_state.prompt_templates
                    )
                    
                    with st.spinner("正在分析学生信息..."):
                        st.subheader("🤔 分析过程")
                        with st.expander("查看详细分析过程", expanded=True):
                            callback = StreamlitCallbackHandler(st.container())
                            result = agent.process(student_info, callback=callback)
                            
                            if result["status"] == "success":
                                # 显示背景分析结果
                                st.markdown("### 📊 背景分析结果")
                                st.markdown(result["profile_analysis"])
                                
                                # 显示内容规划结果
                                st.markdown("### 📝 内容规划结果")
                                st.markdown(result["content_plan"])
                            else:
                                st.error(f"处理失败: {result.get('message', '未知错误')}")
                            
                except Exception as e:
                    st.error(f"处理过程中出错: {str(e)}")
            else:
                st.warning("请先输入学生信息")
    
    with tab2:
        st.title("提示词设置")
        
        # 使用session_state中的prompt_templates
        prompt_templates = st.session_state.prompt_templates
        
        # ProfileStrategist设置
        st.subheader("ProfileStrategist设置")
        profile_strategist_role = st.text_area(
            "角色设定",
            value=prompt_templates.get_template('profile_strategist_role'),
            height=200,
            key="profile_strategist_role"
        )
        profile_strategist_task = st.text_area(
            "任务说明",
            value=prompt_templates.get_template('profile_strategist_task'),
            height=200,
            key="profile_strategist_task"
        )
        
        # ContentCreator设置
        st.subheader("ContentCreator设置")
        content_creator_role = st.text_area(
            "角色设定",
            value=prompt_templates.get_template('content_creator_role'),
            height=200,
            key="content_creator_role"
        )
        content_creator_task = st.text_area(
            "任务说明",
            value=prompt_templates.get_template('content_creator_task'),
            height=200,
            key="content_creator_task"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            # 更新按钮
            if st.button("更新提示词", key="update_prompts"):
                prompt_templates.update_template('profile_strategist_role', profile_strategist_role)
                prompt_templates.update_template('profile_strategist_task', profile_strategist_task)
                prompt_templates.update_template('content_creator_role', content_creator_role)
                prompt_templates.update_template('content_creator_task', content_creator_task)
                st.success("✅ 提示词已更新！")
        
        with col2:
            # 添加重置按钮
            if st.button("重置为默认提示词", key="reset_prompts"):
                prompt_templates.reset_to_default()
                st.rerun()  # 重新运行应用以更新显示

if __name__ == "__main__":
    main()
