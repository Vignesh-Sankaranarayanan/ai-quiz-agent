from openai import OpenAI
import sys
try:
	import pysqlite3  # type: ignore
	sys.modules["sqlite3"] = pysqlite3
except Exception:
	pass
try:
	import chromadb  # initial import
except Exception:
	import importlib
	chromadb = importlib.import_module("chromadb")
import streamlit as st
import json, time, os
import streamlit_authenticator as stauth
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

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
skills = chroma.get_or_create_collection("skills")

def retrieve_skill_context(subject: str, grade: str, difficulty: int):
    # Embed query for vector search; fall back to keyword search on failure
    query = f"{subject} grade {grade} difficulty {difficulty}"
    res = None
    try:
        query_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        ).data[0].embedding
        res = skills.query(query_embeddings=[query_emb], n_results=3, where={"subject": subject})
    except Exception:
        # Fallback: local keyword ranking over all docs to avoid embedding dimension issues
        all_items = skills.get(include=["documents", "metadatas"]) or {}
        all_docs = all_items.get("documents", []) or []
        all_metas = all_items.get("metadatas", []) or []
        # Filter by subject if present
        pairs = [(d, m) for d, m in zip(all_docs, all_metas) if not m or m.get("subject") == subject]
        q_tokens = set(query.lower().split())
        scored = []
        for doc, meta in pairs:
            if not doc:
                continue
            text = str(doc).lower()
            score = sum(1 for t in q_tokens if t in text)
            scored.append((score, doc, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:3]
        res = {
            "documents": [[d for _, d, _ in top]],
            "metadatas": [[m for _, _, m in top]],
        }
    contexts = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        contexts.append({"text": doc, "meta": meta})
    return contexts

def generate_question_llm(subject: str, grade: str, difficulty: int, contexts):
    sys = (
        "You create one multiple-choice question for kids. "
        "Keep it age-appropriate, short, and clear. "
        "The question MUST match the requested Subject exactly and never switch subjects. "
        "Return STRICT JSON only with keys: question (string), options (array of 4 strings), "
        "correct_index (0-3), explanation (string)."
    )
    ctx_text = " ".join([c["text"] for c in contexts])[:1000]
    usr = f"""
Subject: {subject}. Grade: {grade}. Target difficulty (1=easy,3=hard): {difficulty}.
Context to align with skills: {ctx_text}

Constraints:
- 1 question only
- 4 answer options
- exactly one correct
- explanation must be kid-friendly, positive, and concrete

JSON schema:
{{
  "question": "string",
  "options": ["A","B","C","D"],
  "correct_index": 0,
  "explanation": "string"
}}
"""
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                response_format={"type": "json_object"},
                messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            if not (isinstance(data.get("options"), list) and len(data["options"])==4):
                raise ValueError("options not length 4")
            if not (0 <= int(data["correct_index"]) <= 3):
                raise ValueError("bad correct_index")
            return data
        except Exception:
            time.sleep(0.2)
    # Fallback
    return None

def generate_question_offline(subject: str, grade: str, difficulty: int, contexts):
	"""Simple offline fallback generator used when OpenAI is unavailable."""
	try:
		if subject.lower() == "math":
			import random
			random.seed()
			low, high = (2, 9) if difficulty == 1 else (3, 12) if difficulty == 2 else (8, 15)
			a = random.randint(low, high)
			b = random.randint(low, high)
			correct = a * b
			distractors = set()
			while len(distractors) < 3:
				cand = correct + random.choice([-3, -2, -1, 1, 2, 3, 5])
				if cand != correct and cand > 0:
					distractors.add(cand)
			options = [correct] + list(distractors)
			random.shuffle(options)
			return {
				"question": f"What is {a} × {b}?",
				"options": [str(x) for x in options],
				"correct_index": options.index(correct),
				"explanation": f"Multiplication: {a} × {b} = {correct}."
			}
		elif subject.lower() == "ela":
			# Basic noun identification (randomized for variety)
			sentences = [
				("The cat slept on the sofa.", "cat"),
				("Sara read a book in the library.", "library"),
				("The teacher wrote on the board.", "teacher"),
				("Birds fly over the park.", "park"),
				("Tom kicked the ball.", "ball"),
			]
			import random
			random.seed()
			s, noun = random.choice(sentences)
			cands = {"cat", "sofa", "Sara", "book", "library", "teacher", "board", "birds", "park", "Tom", "ball"} - {noun}
			opts = [noun] + random.sample(sorted(cands), 3)
			random.shuffle(opts)
			return {
				"question": f"Which word in this sentence is a noun?\n\n“{s}”",
				"options": opts,
				"correct_index": opts.index(noun),
				"explanation": f"A noun names a person, place, or thing. Here, “{noun}” is a noun."
			}
		else:
			return None
	except Exception:
		return None

st.set_page_config(page_title="AI Quiz Agent", page_icon="🧠")
st.title("🧠 AI Quiz Agent")

# Session state initialization
if "result" not in st.session_state:
    st.session_state.result = None
if "ctxs" not in st.session_state:
    st.session_state.ctxs = []
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False
if "score" not in st.session_state:
    st.session_state.score = 0
if "total" not in st.session_state:
    st.session_state.total = 0
if "history" not in st.session_state:
    st.session_state.history = []
if "accounted" not in st.session_state:
    st.session_state.accounted = False
if "mode" not in st.session_state:
    st.session_state.mode = "offline"
if "last_error" not in st.session_state:
    st.session_state.last_error = None

# Callback helpers

def _do_generate(subject: str, grade: str, difficulty: int):
    with st.spinner("Retrieving context and generating question..."):
        ctxs = retrieve_skill_context(subject, grade, difficulty)
        result = None
        try:
            result = generate_question_llm(subject, grade, difficulty, ctxs)
            st.session_state.mode = "openai"
            st.session_state.last_error = None
        except Exception as e:
            st.session_state.mode = "offline"
            st.session_state.last_error = str(e)
            result = None
        if not result:
            result = generate_question_offline(subject, grade, difficulty, ctxs)
        st.session_state.ctxs = ctxs
        st.session_state.result = result
        st.session_state.show_answer = False
        st.session_state.accounted = False


def _do_check():
    st.session_state.show_answer = True
    # Update score once per question
    if not st.session_state.accounted and st.session_state.result is not None:
        res = st.session_state.result
        correct_idx = int(res["correct_index"])  # ensure int
        correct_opt = res["options"][correct_idx]
        selected = st.session_state.get("selected_option")
        is_correct = selected == correct_opt
        st.session_state.total += 1
        if is_correct:
            st.session_state.score += 1
        st.session_state.accounted = True
        st.session_state.history.append({
            "question": res.get("question", ""),
            "selected": selected,
            "correct": correct_opt,
            "is_correct": is_correct,
        })

with st.sidebar:
	st.header("Settings")
	subject = st.selectbox("Subject", ["Math", "ELA"], index=0, key="subject_select")
	grade = st.selectbox("Grade", ["2", "3", "4", "5", "6"], index=3, key="grade_select")
	difficulty = st.slider("Difficulty (1=easy, 3=hard)", min_value=1, max_value=3, value=2, key="difficulty_slider")
	st.button("Generate Question", on_click=_do_generate, args=(subject, grade, difficulty))
	st.divider()
	st.caption(f"Mode: {st.session_state.mode}")
	if st.session_state.last_error:
		with st.expander("Last error"):
			st.code(st.session_state.last_error)

# Score header
st.caption(f"Score: {st.session_state.score} / {st.session_state.total}")

if not st.session_state.result:
    st.info("Set your subject, grade, and difficulty in the sidebar, then click 'Generate Question'.")
else:
    result = st.session_state.result
    ctxs = st.session_state.ctxs

    st.subheader("Question")
    st.write(result["question"]) 
    choice = st.radio(
        "Choose an answer:",
        options=result["options"],
        index=0,
        key="selected_option",
    )
    st.button("Check Answer", on_click=_do_check)

    if st.session_state.show_answer:
        correct_idx = int(result["correct_index"])  # ensure int
        correct_opt = result["options"][correct_idx]
        if choice == correct_opt:
            st.success("Correct! ✅")
        else:
            st.warning(f"Not quite. The correct answer is: {correct_opt}")
        st.caption("Explanation:")
        st.write(result.get("explanation", ""))

    with st.expander("Context used (top matches)"):
        if not ctxs:
            st.write("No context found.")
        else:
            for i, c in enumerate(ctxs, start=1):
                meta = c.get("meta", {})
                st.markdown(f"**{i}.** {c.get('text','')}")
                st.caption(str(meta))

    if st.session_state.history:
        with st.expander("History"):
            for i, h in enumerate(st.session_state.history, start=1):
                status = "✅" if h.get("is_correct") else "❌"
                st.markdown(f"{i}. {status} {h.get('question','')}")
                st.caption(f"You: {h.get('selected')} | Correct: {h.get('correct')}")
