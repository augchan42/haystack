import os
import logging
import hashlib

from typing import Generator, Optional, Dict, List, Set, Union, Any

from haystack.schema import Document, Answer
from haystack.agents.base import Tool
from haystack.agents.conversational import ConversationalAgent
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode, WebRetriever, PromptTemplate, EmbeddingRetriever
from haystack.nodes.base import BaseComponent
from haystack.pipelines import WebQAPipeline
from haystack.agents.types import Color
from haystack.utils.tb_faiss import build_pipeline
from tb_faiss import setup_document_store, is_reindexing_needed, reindex_document_store

# Configure logging
logging.basicConfig(level=logging.INFO)
haystack_logger = logging.getLogger("haystack")
haystack_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

search_api_key = os.environ.get("SEARCH_API_KEY")
if not search_api_key:
    raise ValueError("Please set the SEARCH_API_KEY environment variable")
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

web_prompt = """
Synthesize a comprehensive answer from the following most relevant paragraphs and the given question.
Provide a clear and concise answer, no longer than 10-20 words.
\n\n Paragraphs: {documents} \n\n Question: {query} \n\n Answer:
"""
class WebQAIndexingPipeline(WebQAPipeline):
    def __init__(self, document_store, docstore_retriever, index_path, config_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.document_store = document_store
        self.index_path = index_path
        self.config_path = config_path
        self.docstore_retriever = docstore_retriever        

    def run(self, query: str, params: Optional[dict] = None, debug: Optional[bool] = None):
        output = super().run(query=query, params=params, debug=debug)

        # Extract the answer from the last line of the PromptNode's output
        answer = output["results"][0].split("\n")[-1]
        output["answers"] = [Answer(answer=answer, type="generative")]

        # Create a new Document object with the question and answer
        document_content = f"Question: {query}\nAnswer: {answer}"
        logger.info("to index document_content: %s", document_content)
        # Create a unique ID based on the hash of the document content
        document_id = hashlib.sha256(document_content.encode()).hexdigest()
        document = Document(id=document_id, content=document_content)

        # Index the question and answer into the document store
        self.document_store.write_documents([document])
        if is_reindexing_needed(document_store):
                reindex_document_store(document_store, embedding_retriever, faiss_index_path, faiss_config_path)            

        return output
    
web_prompt_node = PromptNode(
    "gpt-3.5-turbo", default_prompt_template=PromptTemplate(prompt=web_prompt), api_key=openai_api_key
)

faiss_index_path = "tb_faiss_index.faiss"
faiss_config_path = "tb_faiss_index.json"
embedding_model = "intfloat/e5-base-v2"

try:
    document_store = setup_document_store(
        faiss_index_path, faiss_config_path, embedding_model=embedding_model, use_gpu=True
    )

except Exception as e:
    logger.error("An error occurred while setting up the FAISSDocumentStore: %s", e)
    raise

web_retriever = WebRetriever(api_key=search_api_key, top_search_results=3, mode="snippets")
embedding_retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model=embedding_model, use_gpu=True  # We'll set the document_store later
    )
# pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=web_prompt_node)
pipeline = WebQAIndexingPipeline(retriever=web_retriever, prompt_node=web_prompt_node, 
                                 document_store=document_store,
                                 docstore_retriever=embedding_retriever,
                                 index_path=faiss_index_path,
                                 config_path=faiss_config_path,
                                 )

web_qa_tool = Tool(
    name="GoogleSearch",
    pipeline_or_node=pipeline,
    description="useful for when you need to Google questions if you cannot find answers in the the previous conversation.",
    output_variable="results",
    logging_color=Color.MAGENTA,
)

API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# API_BASE = "http://172.18.176.1:1234/v1"
API_BASE = None
provider = "openai"

faiss_pipeline = build_pipeline(provider, API_KEY, document_store, API_BASE)

faiss_tool = Tool(
    name="Faiss_Query",
    pipeline_or_node=faiss_pipeline,
    description="useful for when you need to ask travel related questions if you cannot find answers in the the previous conversation.",
    output_variable="answers",
    logging_color=Color.CYAN,
)

conversational_agent_prompt_node = PromptNode(
    "gpt-3.5-turbo",
    api_key=openai_api_key,
    max_length=256,
    stop_words=["Observation:"],
    model_kwargs={"temperature": 0.5, "top_p": 0.9},
)
memory = ConversationSummaryMemory(conversational_agent_prompt_node, summary_frequency=2)

tb_custom_prompt = """
    "prompt": "In the following conversation, a human user interacts with an AI Agent. The human user poses questions, and the AI Agent follows a specific process to provide well-informed answers.

    If the AI Agent knows the answer, the response begins with `Final Answer:` on a new line.

    If the AI Agent is uncertain or concerned that the information may be outdated or inaccurate, it must use the available tools in a sequential manner to find the most up-to-date information. The AI has access to these tools:
    {tool_names_with_descriptions}

    The AI Agent must first use 'Tool 1'. If 'Tool 1' does not provide sufficient information, then 'Tool 2' should be used. 

    The following is the previous conversation between a human and an AI:
    {memory}

    AI Agent responses must start with one of the following:
    - Thought: [AI Agent's reasoning process]
    - Tool: [{tool_names}] (on a new line) Tool Input: [input for the selected tool WITHOUT quotation marks and on a new line]
    - Final Answer: [final answer to the human user's question]

    When selecting a tool, the AI Agent must provide both the `Tool:` and `Tool Input:` pair in the same response, but on separate lines. `Observation:` marks the beginning of a tool's result, which the AI Agent trusts.

    The AI Agent should not ask the human user for additional information, clarification, or context.

    If after using both tools and own observations, the AI Agent cannot find a specific answer, it answers with 'Final Answer: inconclusive'.

    Question: {query}
    Thought:
    {transcript}"
    """
# Create a PromptTemplate object with your custom prompt
tb_custom_prompt_template = PromptTemplate(prompt=tb_custom_prompt)


conversational_agent = ConversationalAgent(
    prompt_node=conversational_agent_prompt_node,
    prompt_template=tb_custom_prompt_template,
    tools=[faiss_tool, web_qa_tool], 
    memory=memory
)

test = False
if test:
    questions = [
        "Why was Jamie Foxx recently hospitalized?",
        "Where was he hospitalized?",
        "What movie was he filming at the time?",
        "Who is Jamie's female co-star in the movie he was filing at that time?",
        "Tell me more about her, who is her partner?",
    ]
    for question in questions:
        conversational_agent.run(question)
else:
    while True:
        user_input = input("\nHuman (type 'exit' or 'quit' to quit): ")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            break
        response = conversational_agent.run(user_input)


