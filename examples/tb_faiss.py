# Disable pylint errors for logging basicConfig
# pylint: disable=no-logging-basicconfig
import logging
import os
from typing import Optional
from haystack.document_stores import FAISSDocumentStore  # Import FAISSDocumentStore
from haystack.utils.tb_faiss import build_pipeline, add_json_data
from haystack.utils import print_answers
from haystack.nodes import EmbeddingRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
haystack_logger = logging.getLogger("haystack")
haystack_logger.setLevel(logging.INFO)
# Set the log level for 'numba' to WARNING to suppress debug messages
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def initialize_document_store(faiss_index_path, faiss_config_path, embedding_retriever):
    document_store = None
    if os.path.exists(faiss_index_path):
        logger.info("FAISS index file found. Loading the index.")
        try:
            document_store = FAISSDocumentStore.load(index_path=faiss_index_path, config_path=faiss_config_path)
            # Check if the index is empty or out of sync
            if is_reindexing_needed(document_store):
                reindex_document_store(document_store, embedding_retriever, faiss_index_path, faiss_config_path)
        except ValueError as e:
            logger.info("Reindexing due to: %s", e)
            document_store = create_new_faiss_doc_store()
            reindex_document_store(document_store, embedding_retriever, faiss_index_path, faiss_config_path)
        except Exception as e:
            logger.error("An error occurred while loading the FAISS index: %s", e)
            raise
    else:
        logger.info("FAISS index file does not exist. Creating a new index.")
        document_store = create_new_faiss_doc_store()
        # Add documents to the document store and update embeddings
        reindex_document_store(document_store, embedding_retriever, faiss_index_path, faiss_config_path)

    return document_store


def create_new_faiss_doc_store():
    document_store = FAISSDocumentStore(
        sql_url="sqlite:///tb_faiss_document_store.db",
        faiss_index_factory_str="Flat",
        similarity="cosine",
        validate_index_sync=False,
    )
    return document_store


def reindex_document_store(document_store, embedding_retriever, faiss_index_path, faiss_config_path):
    logger.info("FAISS index is empty or out of sync. Re-indexing.")
    # Add documents to the document store and update embeddings
    add_json_data(document_store, "processed_files")
    document_store.update_embeddings(embedding_retriever)
    # Save the updated index
    document_store.save(index_path=faiss_index_path, config_path=faiss_config_path)


def is_reindexing_needed(document_store):
    # Get the count of documents in the SQL database
    sql_doc_count = document_store.get_document_count()
    # Get the count of indexed documents in the FAISS index
    faiss_doc_count = document_store.get_embedding_count()

    # Reindexing is needed if the counts do not match
    return sql_doc_count != faiss_doc_count


def getting_started(provider, API_KEY, API_BASE: Optional[str] = None):
    """
    This getting_started example shows you how to use LLMs with your data with a technique called Retrieval Augmented Generation - RAG.

    :param provider: We are model agnostic :) Here, you can choose from: "anthropic", "cohere", "huggingface", and "openai".
    :param API_KEY: The API key matching the provider.
    """

    query = "what travel plans do you offer?"
    faiss_index_path = "tb_faiss_index.faiss"
    faiss_config_path = "tb_faiss_index.json"

    # Initialize EmbeddingRetriever
    embedding_retriever = EmbeddingRetriever(
        document_store=None, embedding_model="intfloat/e5-base-v2", use_gpu=True  # We'll set the document_store later
    )

    try:
        document_store = initialize_document_store(faiss_index_path, faiss_config_path, embedding_retriever)
        embedding_retriever.document_store = document_store  # Set the document_store
    except Exception as e:  # Catch any exception that might occur
        logger.error("An error occurred while initializing the FAISSDocumentStore: %s", e)
        raise  # Re-raise the exception to handle it appropriately

    # The rest of the code remains the same
    pipeline = build_pipeline(provider, API_KEY, document_store, API_BASE)

    if is_reindexing_needed(document_store):
        reindex_document_store(document_store, embedding_retriever, faiss_index_path, faiss_config_path)

    result = pipeline.run(query=query, debug=True)
    print(result["_debug"])

    print_answers(result, details="medium")

    return result


if __name__ == "__main__":
    # Get the API key from the environment variable
    API_KEY = os.getenv("OPENAI_API_KEY")
    if API_KEY is None:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    API_BASE = None
    # API_BASE="http://192.168.1.107:1234/v1"
    getting_started(provider="openai", API_KEY=API_KEY, API_BASE=API_BASE)
