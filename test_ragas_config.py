import sys
sys.path.insert(0, str(sys.path[0]) + '/backend')

from dotenv import load_dotenv
load_dotenv()

from app.core.config import get_settings
settings = get_settings()

# Test RAGAS configuration
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    print("Langchain wrappers imported successfully")
    
    # Try to create the wrappers
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    
    # Create LLM
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.CHAT_API_KEY,
        base_url=settings.CHAT_API_BASE,
        temperature=0,
        max_tokens=2000
    )
    
    # Try wrapping
    try:
        ragas_llm = LangchainLLMWrapper(llm=llm)
        print("LangchainLLMWrapper created successfully with llm parameter")
    except Exception as e:
        print(f"Failed with llm parameter: {e}")
        
        # Try without parameter name
        try:
            ragas_llm = LangchainLLMWrapper(llm)
            print("LangchainLLMWrapper created successfully without parameter name")
        except Exception as e2:
            print(f"Failed without parameter name: {e2}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
