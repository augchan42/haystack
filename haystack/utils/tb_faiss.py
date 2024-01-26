import logging
import os

from pathlib import Path
from haystack.nodes import EmbeddingRetriever, JsonConverter
from haystack.schema import Document
from typing import List

logger = logging.getLogger(__name__)


def build_pipeline(provider, API_KEY, document_store, API_BASE):
    # Importing top-level causes a circular import
    from haystack.nodes import AnswerParser, PromptNode, PromptTemplate
    from haystack.pipelines import Pipeline

    provider = provider.lower()
    # A retriever selects the right documents when given a question.
    # retriever = BM25Retriever(document_store=document_store, top_k=5)
    embedding_model = "intfloat/e5-base-v2"  # Example model
    retriever = EmbeddingRetriever(
        top_k=5, document_store=document_store, embedding_model=embedding_model, use_gpu=True
    )

    # Load prompt for doing retrieval augmented generation from https://prompthub.deepset.ai/?prompt=deepset%2Fquestion-answering-with-references
    question_answering_with_references = PromptTemplate(
        prompt="question-answering-with-references", output_parser=AnswerParser(reference_pattern=r"Document\[(\d+)\]")
    )
    # Load the LLM model
    if provider == "anthropic":
        prompt_node = PromptNode(
            model_name_or_path="claude-2", api_key=API_KEY, default_prompt_template=question_answering_with_references
        )
    elif provider == "cohere":
        prompt_node = PromptNode(
            model_name_or_path="command", api_key=API_KEY, default_prompt_template=question_answering_with_references
        )
    elif provider == "huggingface":
        # TODO: swap out for meta-llama/Llama-2-7b-chat-hf or the 40b model once supported in Haystack+HF API free tier
        # The tiiuae/falcon-7b-instruct model cannot handle a complex prompt with references, so we use a very simple one
        simple_QA = PromptTemplate(
            prompt="deepset/question-answering", output_parser=AnswerParser(reference_pattern=r"Document\[(\d+)\]")
        )
        prompt_node = PromptNode(
            model_name_or_path="tiiuae/falcon-7b-instruct", api_key=API_KEY, default_prompt_template=simple_QA
        )
    elif provider == "openai":
        prompt_node = PromptNode(
            model_name_or_path="gpt-3.5-turbo-0301",
            api_key=API_KEY,
            api_base=API_BASE,
            default_prompt_template=question_answering_with_references,
            model_kwargs={"temperature": 0, "stream": True},  # Set your desired temperature here
        )
    else:
        logger.error('Given <provider> unknown. Please use any of "anthropic", "cohere", "huggingface", or "openai"')
    # Compose the query pipeline
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

    return query_pipeline


def add_json_data(document_store, dir):
    # Importing top-level causes a circular import

    dir_path = os.path.abspath(dir)

    files_to_index = [os.path.join(dir_path, f) for f in os.listdir(dir)]
    logger.info("Adding %s number of files from local disk at %s.", len(files_to_index), dir)
    docs = convert_json_to_docs(dir_path=dir)    

    document_store.write_documents(documents=docs)

def convert_json_to_docs(json_file_path: str) -> List[Document]:
    converter = JsonConverter()
    # Assuming the JSON file contains a list of entries that you want to convert to documents
    docs = converter.convert(file_path=Path(json_file_path))
    return docs