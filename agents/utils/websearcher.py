from duckduckgo_search import DDGS

class DuckDuckGoSearcher:
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, query: str, num_results: int = 5) -> list:
        results = self.ddgs.text(query, max_results=num_results)
        return [{"title": r["title"], "snippet": r["body"], "url": r["href"]} for r in results]
    

# ws = DuckDuckGoSearcher()
# print(ws.search("Garfield the Cat"))