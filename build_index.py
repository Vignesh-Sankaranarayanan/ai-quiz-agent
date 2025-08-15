import os, json
from openai import OpenAI
try:
    import sys
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

# Seed skills (extend over time)
SKILLS = [
  {"id": "math_g3_multiplication", "subject": "Math", "grades": ["3","4"], "difficulty": 2,
   "text": "Multiply single-digit numbers within 100. Use grouping or skip counting. Common mistakes: mixing add and multiply."},
  {"id": "ela_g3_nouns", "subject": "ELA", "grades": ["2","3","4"], "difficulty": 1,
   "text": "Identify nouns in simple sentences; a noun is a person, place, or thing."},
  {"id": "ela_g5_punctuation", "subject": "ELA", "grades": ["5","6"], "difficulty": 3,
   "text": "Use semicolons to join related independent clauses. Distinguish it's vs its."},
]

# Resolve API key from env or secrets files
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    for rel in [os.path.join(BASE_DIR, ".streamlit", "secrets.toml"), os.path.join(BASE_DIR, "secrets.toml")]:
        try:
            if os.path.exists(rel):
                import tomllib
                with open(rel, "rb") as f:
                    data = tomllib.load(f)
                if isinstance(data, dict):
                    api_key = data.get("OPENAI_API_KEY") or (data.get("openai") or {}).get("api_key") or (data.get("OPENAI") or {}).get("api_key")
                if api_key:
                    break
        except Exception:
            pass
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Set env var or add it to .streamlit/secrets.toml or secrets.toml")

# Optionally configure HTTP client for proxies/TLS
_http_client = None
try:
    import httpx, ssl
    skip_verify = os.getenv("OPENAI_INSECURE_SKIP_VERIFY") == "1"
    verify = False if skip_verify else None
    bundle = os.getenv("OPENAI_CA_BUNDLE") or os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    if not skip_verify and bundle and os.path.exists(bundle):
        verify = bundle
    elif not skip_verify:
        try:
            import truststore  # pip install truststore
            ssl_ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            verify = ssl_ctx
        except Exception:
            verify = None
    use_http2 = os.getenv("OPENAI_DISABLE_HTTP2") != "1"
    _http_client = httpx.Client(http2=use_http2, timeout=30.0, verify=verify)
except Exception:
    _http_client = None

client = OpenAI(api_key=api_key, http_client=_http_client) if _http_client else OpenAI(api_key=api_key)
try:
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
except AttributeError:
    from chromadb.config import Settings
    chroma = chromadb.Client(Settings(persist_directory=CHROMA_PATH))
coll = chroma.get_or_create_collection(name="skills")

# Embed with OpenAI
def embed(texts):
    res = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in res.data]

docs = [s["text"] for s in SKILLS]
metadatas = [{"subject": s["subject"], "grades": ",".join(s["grades"]), "difficulty": s["difficulty"]} for s in SKILLS]
ids = [s["id"] for s in SKILLS]
embs = embed(docs)

coll.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embs)

print("Indexed", len(SKILLS), "skills into Chroma at", CHROMA_PATH)

