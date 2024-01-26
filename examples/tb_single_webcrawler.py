import logging
import re
import hashlib

from pathlib import Path

from haystack.nodes.connector import Crawler
from haystack.nodes import PreProcessor, PromptNode, PromptTemplate, AnswerParser, EmbeddingRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
haystack_logger = logging.getLogger("haystack")
haystack_logger.setLevel(logging.DEBUG)
# Set the log level for 'numba' to WARNING to suppress debug messages
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

crawler = Crawler(output_dir="crawled_files")
# List of URLs to crawl
single_urls = ["https://www.moneyhero.com.hk/en/mortgage/providers"]


def generate_filename(url):
    file_name_link = re.sub("[<>:'/\\|?*\0 ]", "_", url[:129])
    file_name_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    file_name_prefix = f"{file_name_link}_{file_name_hash[-6:]}"
    return f"{file_name_prefix}.json"


def has_been_crawled(url, output_dir):
    # Generate the expected filename from the URL
    expected_filename = generate_filename(url)
    # Check if this file already exists in the output directory
    file_path = Path(output_dir) / expected_filename
    return file_path.exists()


docs = crawler.crawl(urls=single_urls, extract_hidden_text=False, crawler_depth=0)
