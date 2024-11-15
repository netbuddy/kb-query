import os
import chainlit as cl
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.openai import OpenAI
import redis
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Redis 配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = None

# Milvus 配置
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530

# OpenAI 配置
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL")

# 设置 LlamaIndex 默认 LLM
Settings.llm = OpenAI(model="gpt-3.5-turbo-0125")

# 初始化 Redis 客户端
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

@cl.step(type="tool", name="初始化存储上下文")
async def get_storage_context(index_name: str):
    """创建存储上下文"""
    doc_store = RedisDocumentStore.from_host_and_port(
        host=REDIS_HOST,
        port=REDIS_PORT,
        namespace=index_name
    )
    
    index_store = RedisIndexStore.from_host_and_port(
        host=REDIS_HOST,
        port=REDIS_PORT,
        namespace=index_name
    )
    
    vector_store = MilvusVectorStore(
        uri=f"tcp://{MILVUS_HOST}:{MILVUS_PORT}",
        collection_name=index_name,
        dim=1536,
        overwrite=False
    )
    
    return StorageContext.from_defaults(
        docstore=doc_store,
        index_store=index_store,
        vector_store=vector_store
    )

@cl.step(type="tool", name="加载索引")
async def load_index(storage_context):
    """加载索引"""
    return load_index_from_storage(storage_context)

@cl.step(type="tool", name="创建查询引擎")
async def create_query_engine(index):
    """创建查询引擎"""
    return index.as_query_engine(
        streaming=True,
        similarity_top_k=3
    )

@cl.step(type="tool", name="检索相关文档")
async def retrieve_documents(query_engine, query_text: str):
    """检索相关文档"""
    step = cl.context.current_step
    step.input = f"用户问题: {query_text}"
    
    # 执行查询获取响应
    response = await cl.make_async(query_engine.query)(query_text)
    
    # 获取源文档信息
    source_info = []
    if hasattr(response, 'source_nodes'):
        for node in response.source_nodes:
            source_info.append({
                'score': round(node.score, 3) if hasattr(node, 'score') else None,
                'text': node.node.text[:200] + "..."  # 只显示前200个字符
            })
    
    # 显示输出
    output_text = "检索到的相关文档:\n"
    for idx, source in enumerate(source_info, 1):
        output_text += f"\n{idx}. 相关度: {source['score']}\n文本: {source['text']}\n"
    
    step.output = output_text
    return response, source_info

@cl.step(type="tool", name="生成回答")
async def generate_answer(response):
    """生成 AI 回答"""
    step = cl.context.current_step
    step.input = "基于检索到的文档生成回答"
    
    # 获取完整响应文本
    response_text = ""
    if hasattr(response, 'response_gen'):
        for text in response.response_gen:
            response_text += text
    else:
        response_text = str(response)
    
    step.output = f"AI回答: {response_text}"
    return response_text

@cl.on_chat_start
async def start():
    """聊天启动时的处理函数"""
    # 从 Redis 获取索引列表
    index_names = redis_client.smembers("llama_index:namespaces")
    
    if not index_names:
        await cl.Message(
            content="没有找到可用的知识库，请确保已经创建了索引。"
        ).send()
        return
    
    # 创建知识库选择动作列表
    actions = [
        cl.Action(
            name=index_name,
            value=index_name,
            label=f"📚 {index_name}"
        ) for index_name in index_names
    ]
    
    # 创建选择界面
    res = await cl.AskActionMessage(
        content="请选择要查询的知识库：",
        actions=actions,
    ).send()
    
    if res:
        selected_index = res.get("value")
        try:
            # 加载选定的索引
            storage_context = await get_storage_context(selected_index)
            index = await load_index(storage_context)
            query_engine = await create_query_engine(index)
            
            # 存储到用户会话
            cl.user_session.set("query_engine", query_engine)
            
            await cl.Message(
                content=f"已选择知识库：{selected_index}，您可以开始提问了。"
            ).send()
        except Exception as e:
            await cl.Message(
                content=f"加载知识库时出错：{str(e)}"
            ).send()
    else:
        await cl.Message(
            content="未选择知识库，请重新开始对话。"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """处理用户消息"""
    query_engine = cl.user_session.get("query_engine")
    if not query_engine:
        await cl.Message(
            content="请先选择知识库。"
        ).send()
        return

    try:
        # # 步骤1: 检索相关文档
        response, source_info = await retrieve_documents(query_engine, message.content)
        
        # 步骤2: 生成回答
        response_text = await generate_answer(response)

        # response_text = "abcdef"
        # source_info = []
        # source_info.append({
        #     'score': 0.9,
        #     'text': "123456"
        # })
        # source_info.append({
        #     'score': 0.8,
        #     'text': "7890"
        # })

        # 构建源文档展示文本
        elements = []
        element_refs = []  # 存储元素引用文本

        for idx, source in enumerate(source_info, 1):
            # 格式化相关度为百分比
            element_name = f"参考文档_{idx}(相关度：{source['score']})"
            
            elements.append(
                cl.Text(
                    name=element_name,
                    content=source['text'],  # 只显示文档内容
                    display="side"
                )
            )
            element_refs.append(element_name)  # 添加元素引用

        # 创建流式消息并逐字符显示
        msg = cl.Message(content="", elements=elements)
        for token in response_text:
            await msg.stream_token(token)

        # 添加元素引用
        await msg.stream_token("\n\n参考文档：")
        for ref in element_refs:
            await msg.stream_token(f"\n{ref}")

        await msg.send()
            
    except Exception as e:
        await cl.Message(
            content=f"查询出错：{str(e)}"
        ).send() 