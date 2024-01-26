import logging
import json

from pathlib import Path

from haystack import Document
from haystack.nodes import PreProcessor, PromptNode, PromptTemplate, AnswerParser, EmbeddingRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
haystack_logger = logging.getLogger("haystack")
haystack_logger.setLevel(logging.DEBUG)
# Set the log level for 'numba' to WARNING to suppress debug messages
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Initialize your PreProcessor with desired settings
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="sentence",
    # split_length=200, # for words
    split_length=10,  # for words
    split_respect_sentence_boundary=False,
    split_overlap=2,
    max_chars_check=10_000,
)

# Path to your JSON files
input_directory = Path("crawled_files")
output_directory = Path("processed_files")
output_directory.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

processed_documents = []

# Now you have a list of preprocessed Document objects in processed_documents
# for file_path in input_directory.glob('*.json'):
for file_path in input_directory.glob("https___www.moneyhero.com.hk_en_travel-insurance_psCollapse=true_a889c4.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        document = Document.from_dict(data)
        # Preprocess the document
        processed_docs = preprocessor.process(document)
        # Serialize and write each processed document to the output directory
        for i, doc in enumerate(processed_docs):
            output_file_path = output_directory / f"{file_path.stem}_part_{i}.json"
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                json.dump(doc.to_dict(), output_file, ensure_ascii=False, indent=4)
