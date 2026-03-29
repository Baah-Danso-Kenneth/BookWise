import json
import logging
from typing import List, Dict, Any

from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq

from app.tools.base import MCPToolDescriptor, MCPToolServer, MCPToolResult, MCPToolStatus
from app.config import config


class BookSearchServer(MCPToolServer):
    """
    MCP-compliant book search tool.
    Fetches Tavily snippets → LLM extracts clean book entries
    → LLM fills any missing authors from its own knowledge.
    """

    def _register(self) -> MCPToolDescriptor:
        return MCPToolDescriptor(
            name="BookSearch",
            version="1.0.0",
            description="Search for books by title, topic, or author using web search",
            input_schema={
                "type": "object",
                "properties": {
                    "query":       {"type": "string"},
                    "search_type": {"type": "string", "enum": ["book_based", "topic_based", "complex"]},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        )

    def execute(self, query: str, search_type: str = "topic_based", max_results: int = 5) -> MCPToolResult:
        logging.info(f"BookSearch: '{query}' (type={search_type})")

        try:
            raw_snippets = self._fetch_tavily(query, search_type, max_results)

            if not raw_snippets:
                return MCPToolResult(
                    tool_name="BookSearch",
                    status=MCPToolStatus.SUCCESS,
                    content={"books": [], "query": query, "total_found": 0},
                    metadata={"search_type": search_type}
                )

            books = self._extract_books_with_llm(raw_snippets, query, max_results)
            books = self._fill_missing_authors(books)

            logging.info(f"BookSearch: extracted {len(books)} books")

            return MCPToolResult(
                tool_name="BookSearch",
                status=MCPToolStatus.SUCCESS,
                content={"books": books, "query": query, "total_found": len(books)},
                metadata={"search_type": search_type, "max_results": max_results}
            )

        except Exception as e:
            logging.error(f"BookSearch failed: {e}")
            return MCPToolResult(
                tool_name="BookSearch",
                status=MCPToolStatus.ERROR,
                content={"books": [], "error": str(e)},
                metadata={"query": query}
            )

    # ── Tavily ───────────────────────────────────────────────────────────────

    def _fetch_tavily(self, query: str, search_type: str, max_results: int) -> str:
        search_query = self._build_search_query(query, search_type)
        tavily = TavilySearch(api_key=config.TAVILY_API_KEY, max_results=max_results)
        response = tavily.invoke(search_query)
        results_list = response.get("results", []) if isinstance(response, dict) else []
        if not results_list:
            return ""
        parts = []
        for r in results_list:
            content = r.get("content", "").strip()
            if content:
                parts.append(f"SOURCE: {r.get('url', '')}\n{content}")
        return "\n\n---\n\n".join(parts)

    def _build_search_query(self, query: str, search_type: str) -> str:
        if search_type == "book_based":
            return f"book titles authors descriptions similar to {query}"
        elif search_type == "topic_based":
            return f"best book titles authors descriptions about {query}"
        return f"book titles authors descriptions {query}"

    # ── Step 1: Extract books from snippets ──────────────────────────────────

    def _extract_books_with_llm(self, raw_snippets: str, query: str, max_results: int) -> List[Dict]:
        llm = ChatGroq(
            model=config.PRIMARY_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=1500
        )

        system_prompt = (
            "You are a book data extractor. Extract real book entries from web search snippets.\n\n"
            "Return ONLY a valid JSON array — no markdown, no backticks, no explanation.\n\n"
            "Each object must have:\n"
            '  "title"       — actual book title (not a webpage title)\n'
            '  "author"      — full author name found in the snippet, or "Unknown" if truly absent\n'
            '  "description" — 2-3 sentences from the snippet describing what the book is about\n'
            '  "score"       — float 0.0-1.0 relevance to the query\n\n'
            "Rules:\n"
            "- NEVER use a webpage/site title as a book title\n"
            "- NEVER invent a description — only use text from the snippets\n"
            "- Extract each book from numbered lists like '#1. Title by Author'\n"
            "- Skip entries with no real description found in the snippets\n"
            f"- Return at most {max_results} books\n"
            "- If no real books found, return []"
        )

        user_prompt = (
            f'User query: "{query}"\n\n'
            f"Search snippets:\n{raw_snippets[:4000]}\n\n"
            "Extract the real book entries."
        )

        try:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            books = json.loads(raw)
            if not isinstance(books, list):
                return []

            cleaned = []
            for b in books:
                if not isinstance(b, dict):
                    continue
                title = b.get("title", "").strip()
                description = b.get("description", "").strip()
                if not title or not description:
                    continue
                cleaned.append({
                    "title":       title,
                    "author":      b.get("author", "Unknown").strip(),
                    "description": description,
                    "score":       float(b.get("score", 0.7)),
                    "reason":      "Found via web search"
                })

            return cleaned[:max_results]

        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"BookSearch extraction failed: {e}")
            return []

    # ── Step 2: Fill missing authors using LLM knowledge ────────────────────

    def _fill_missing_authors(self, books: List[Dict]) -> List[Dict]:
        """
        For any book where author is 'Unknown', ask the LLM directly.
        Uses a single batch call for efficiency.
        """
        unknown_books = [b for b in books if b.get("author", "Unknown") == "Unknown"]

        if not unknown_books:
            return books

        llm = ChatGroq(
            model=config.PRIMARY_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=500
        )

        titles = [b["title"] for b in unknown_books]
        titles_list = "\n".join(f"- {t}" for t in titles)

        prompt = (
            "For each book title below, provide the real author's full name.\n"
            "Return ONLY a JSON object mapping title to author name.\n"
            "If you genuinely don't know a title, use 'Unknown'.\n"
            "No markdown, no backticks.\n\n"
            f"Titles:\n{titles_list}"
        )

        try:
            response = llm.invoke([{"role": "user", "content": prompt}])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            author_map = json.loads(raw)

            for book in books:
                if book.get("author") == "Unknown":
                    found = author_map.get(book["title"])
                    if found and found != "Unknown":
                        book["author"] = found

        except Exception as e:
            logging.warning(f"Author fill failed: {e}")

        return books