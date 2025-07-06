import os
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()  # Make sure your .env file has TAVILY_API_KEY=your_key

TAVILY_API_KEY = "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"

def test_tavily(query: str):
    if not TAVILY_API_KEY:
        print("❌ TAVILY_API_KEY not loaded from environment.")
        return

    tool = TavilySearch(tavily_api_key=TAVILY_API_KEY)

    try:
        print(f"🔍 Querying Tavily with: {query}")
        response = tool.invoke({"query": query})
        print("\n✅ Full Tavily Response:")
        print(response)

        results = response.get("results", [])

        if isinstance(results, list) and results:
            print("\n📄 Top 3 Results Content:")
            for idx, res in enumerate(results[:3]):
                print(f"\nResult {idx+1}:")
                print(res.get("content", "(no content)"))
        else:
            print("⚠️ No results found.")

    except Exception as e:
        print(f"🔥 Exception occurred: {e}")

# Run test
test_tavily("What are some alternatives to LangChain?")
