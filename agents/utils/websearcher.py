import requests
import uuid
from typing import List, Dict
from cleantext import clean
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from langchain_ollama import OllamaEmbeddings
import fitz # pip install PyMuPDF

class DuckDuckGoSearcher:
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        results = self.ddgs.text(query, max_results=num_results)
        return [{"title": r["title"], "snippet": r["body"], "url": r["href"]} for r in results]

class WebContentRetriever:
    def __init__(self, qdrant_url: str = "http://localhost:6333", qdrant_collection_name: str = "web_content", num_results: int = 3):
        self.searcher = DuckDuckGoSearcher()
        self.num_results = num_results
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = qdrant_collection_name
        self.embeddings = OllamaEmbeddings(model="llama3.1")
        # self._create_qdrant_collection()

    def _create_qdrant_collection(self):
        if self.collection_name not in self.qdrant_client.get_collections().collections:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=4096, distance='Cosine')
            )

    def _scrape_content(self, url: str) -> str:
        if url.lower().endswith(".pdf"):
            return self._extract_pdf_text(url)
        else:
            return self._extract_html_text(url)

    def _extract_html_text(self, url: str) -> str:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return self._clean_text(content)

    def _extract_pdf_text(self, url: str) -> str:
        response = requests.get(url, timeout=15)
        filename = f"./tmp/{uuid.uuid4()}.pdf"
        with open(filename, "wb") as f:
            f.write(response.content)
        try:
            doc = fitz.open(filename)
            text = ""
            for page in doc:
                text += page.get_text()
            return self._clean_text(text)
        except Exception as e:
            print(f"Error parsing PDF {url}: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        content = ' '.join(text.split())
        cleaned_content = clean(
            content,
            fix_unicode=True,
            to_ascii=True,
            lower=True,
            no_line_breaks=False,
            no_urls=False,
            no_emails=False,
            no_phone_numbers=False,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            no_punct=False,
            replace_with_punct="",
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            replace_with_number="<NUMBER>",
            replace_with_digit="0",
            replace_with_currency_symbol="<CUR>",
            lang="en"
        )
        return cleaned_content

    def _get_embeddings(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text=text)

    def store_content_in_qdrant(self, query: str):
        search_results = self.searcher.search(query, num_results=self.num_results)

        for result in search_results:
            url = result["url"]
            title = result["title"]
            print(f"Scraping content from: {url}")

            content = self._scrape_content(url)
            if not content.strip():
                print(f"Skipping empty content for: {url}")
                continue

            embeddings = self._get_embeddings(content)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    {
                        "id": point_id,
                        "vector": embeddings,
                        "payload": {
                            "title": title,
                            "content": content,
                            "url": url
                        }
                    }
                ]
            )
            print(f"Stored content from: {url} in Qdrant")

    def search_in_qdrant(self, query: str) -> List[Dict]:
        query_embeddings = self._get_embeddings(query)
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embeddings,
            with_payload=True,
            limit=3
        ).points

        return [
            {
                "title": result.payload["title"],
                "content": result.payload["content"],
                "score": result.score
            }
            for result in search_results
        ]


###################################################################
#   USAGE GUIDE
###################################################################
# qdrant_url = "http://localhost:6333"
# collection_name = "web_content"

# Create Retriever
# retriever = WebContentRetriever(qdrant_url, collection_name)

# Query For Storage 
# retriever.store_content_in_qdrant("Garfield the Cat")

# Query For Retrieval
# search_results = retriever.search_in_qdrant("Garfield the Cat")
# for result in search_results:
#     print(result)
# result = {'title': "", 'content': "", 'url': "", 'score': float}
