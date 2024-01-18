import logging
import sys

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.pipeline_utils import build_rag_pipeline, build_indexing_pipeline
from haystack.pipeline_utils.indexing import download_files

# Set up logging
logging.basicConfig(level=logging.INFO) #set to DEBUG for more info
logger = logging.getLogger(__name__)

#If working on your own fork
sys.path.insert(0, '/home/hosermage/forked-projects/haystack')
print(f"System path: {sys.path}")

# We are model agnostic :) In this getting started you can choose any OpenAI or Huggingface TGI generation model
generation_model = "gpt-3.5-turbo"
API_KEY = "sk-..."  # ADD YOUR KEY HERE
API_BASE_URL = "http://192.168.1.96:1234/v1" #Uncomment if running local model with LM Studio or similar
PROMPT_TEMPLATE = """
                [INST]
                Instruct:
                Given 
                  documents, answer the question.

                Documents:
                {% for doc in documents %}
                    {{ doc.content }}
                {% endfor %}

                Question: {{question}}

                Output:
                [/INST]
                """
SYSTEM_PROMPT = """
                You will be given a list of documents to answer a question.  Please reference ONLY
                these documents when formulating your answer.
                """

# We support many different databases. Here, we load a simple and lightweight in-memory database.
# Use cosine for e5-base-v2 https://discord.com/channels/993534733298450452/1141635410024468491/1141761753772998787
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
logger.info(f"Document store configured: {document_store}")
logger.info("Document store attributes:")
for attr, value in document_store.__dict__.items():
    logger.info(f"  {attr}: {value}")
# document_store = InMemoryDocumentStore()

# Download example files from web
# files = download_files(sources=["http://www.paulgraham.com/superlinear.html"])
sources = ["http://www.paulgraham.com/superlinear.html"]
logger.info(f"Downloading files from sources: {sources}")
files = download_files(sources=sources)
logger.info(f"Downloaded files: {files}")

# Pipelines are our main abstratcion.
# Here we create a pipeline that can index TXT and HTML. You can also use your own private files.
indexing_pipeline = build_indexing_pipeline(
    document_store=document_store,
    embedding_model="intfloat/e5-base-v2",
    supported_mime_types=["text/plain", "text/html"],  # "application/pdf"
)
logger.info(f"Indexing files: {files}")
indexing_pipeline.run(files=files)  # you can also supply files=[path_to_directory], which is searched recursively

# Assuming document_store is an instance of InMemoryDocumentStore
document_count = document_store.count_documents()
# Log the number of documents in the store
logger.info(f"Number of documents in the store: {document_count}")

logger.info(f"Building RAG Pipeline: {files}")
rag_pipeline_kwargs = {
    "document_store": document_store,
    "embedding_model": "intfloat/e5-base-v2",
    "generation_model": generation_model,
    "llm_api_key": API_KEY, 
}
# Include api_base_url only if it's defined
if 'API_BASE_URL' in locals() or 'API_BASE_URL' in globals():
    rag_pipeline_kwargs['api_base_url'] = API_BASE_URL

if 'PROMPT_TEMPLATE' in locals() or 'PROMPT_TEMPLATE' in globals():
    rag_pipeline_kwargs['prompt_template'] = PROMPT_TEMPLATE

if 'SYSTEM_PROMPT' in locals() or 'SYSTEM_PROMPT' in globals():
    rag_pipeline_kwargs['system_prompt'] = SYSTEM_PROMPT

rag_pipeline = build_rag_pipeline(**rag_pipeline_kwargs)

rag_pipeline.log_attributes()

logger.info(f"Running RAG Pipeline: {files}")
# For details, like which documents were used to generate the answer, look into the result object
result = rag_pipeline.run(query="What are superlinear returns and why are they important?")

# logger.info(f"Pipeline returned: {result}")

print(result.data)
