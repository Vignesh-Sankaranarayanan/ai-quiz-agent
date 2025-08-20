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
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")
SELECT_PROMPT = "— Select an answer —"

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
# Streamlit Cloud secrets fallback
if not api_key:
	try:
		api_key = st.secrets.get("OPENAI_API_KEY")
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
            model=st.session_state.get("embed_model", "text-embedding-3-small"),
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

# Randomize options order and update correct_index accordingly
def _shuffle_mc_options(data: dict) -> dict:
    try:
        import random
        options = data.get("options")
        correct_index = int(data.get("correct_index", 0))
        if not isinstance(options, list) or len(options) != 4:
            return data
        correct_opt = options[correct_index]
        shuffled = options[:]
        random.shuffle(shuffled)
        data["options"] = shuffled
        data["correct_index"] = shuffled.index(correct_opt)
        return data
    except Exception:
        return data

def generate_question_llm(subject: str, grade: str, difficulty: int, contexts, previous_questions=None, question_type: Optional[str] = None):
    sys = (
        "You create one multiple-choice question for kids. "
        "Keep it age-appropriate, short, and clear. "
        "The question MUST match the requested Subject exactly and never switch subjects. "
        "Return STRICT JSON only with keys: question (string), options (array of 4 strings), "
        "correct_index (0-3), explanation (string). "
        "Make the explanation 2-3 short sentences: first explain why the correct option is right, then give a simple tip/strategy."
    )
    ctx_text = " ".join([c["text"] for c in contexts])[:1000]
    prev_list = previous_questions or []
    prev_text = "\n- " + "\n- ".join(prev_list[-10:]) if prev_list else ""
    qtype_text = f"Question type: {question_type}." if question_type else ""
    usr = f"""
Subject: {subject}. Grade: {grade}. Target difficulty (1=easy,3=hard): {difficulty}.
{qtype_text}
Context to align with skills: {ctx_text}

Constraints:
- 1 question only
- 4 answer options
- exactly one correct
- explanation must be kid-friendly, positive, and concrete
- explanation should be 2-3 sentences: why it's correct + a simple strategy/tip
- do not repeat any of these previous questions (avoid same or trivially paraphrased wording):{prev_text}

JSON schema:
{{
  "question": "string",
  "options": ["A","B","C","D"],
  "correct_index": 0,
  "explanation": "string"
}}
"""
    last_err = None
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=st.session_state.get("chat_model", "gpt-4o"),
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
            # Randomize options to avoid position bias
            data = _shuffle_mc_options(data)
            return data
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    # If we reach here, record the last error for the UI
    try:
        if last_err is not None:
            st.session_state.last_error = f"OpenAI error: {last_err}"
    except Exception:
        pass
    # Fallback
    return None

def generate_question_offline(subject: str, grade: str, difficulty: int, contexts, question_type: Optional[str] = None):
	"""Simple offline fallback generator used when OpenAI is unavailable. Diversifies math question types."""
	try:
		import random
		random.seed()
		if subject.lower() == "math":
			# Choose a question type if not provided
			qtypes = ["addition", "subtraction", "multiplication", "division", "word_problem"]
			qtype = (question_type or random.choice(qtypes)).lower()
			low, high = (2, 9) if difficulty == 1 else (3, 12) if difficulty == 2 else (8, 20)
			a = random.randint(low, high)
			b = random.randint(low, high)
			question = ""
			correct = None
			explanation = ""
			if qtype == "addition":
				correct = a + b
				question = f"What is {a} + {b}?"
				explanation = f"Add the numbers: {a} + {b} = {correct}. Tip: add the ones, then the tens."
			elif qtype == "subtraction":
				if a < b:
					a, b = b, a
				correct = a - b
				question = f"What is {a} − {b}?"
				explanation = f"Take away {b} from {a}: {a} − {b} = {correct}. Tip: subtract the ones first."
			elif qtype == "division":
				# Ensure divisible for nice integers
				correct = a
				b = random.randint(low, high)
				c = a * b
				question = f"What is {c} ÷ {b}?"
				explanation = f"Since {correct} × {b} = {c}, {c} ÷ {b} = {correct}. Tip: use the inverse of multiplication."
			elif qtype == "word_problem":
				# Simple multiplicative or additive word problem
				apples_each = a
				bags = b
				correct = apples_each * bags
				question = f"A bag has {apples_each} apples. There are {bags} bags. How many apples are there in total?"
				explanation = f"Multiply groups: {apples_each} apples × {bags} bags = {correct}. Tip: repeated addition also works."
			else:  # multiplication
				correct = a * b
				question = f"What is {a} × {b}?"
				explanation = f"Multiply: {a} × {b} = {correct}. Tip: think of {a} groups of {b}."

			# Build distractors near the correct answer
			distractors = set()
			while len(distractors) < 3:
				delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 4, 6, 8])
				cand = correct + delta
				if cand != correct and cand > 0:
					distractors.add(cand)
			options = [correct] + list(distractors)
			random.shuffle(options)
			return {
				"question": question,
				"options": [str(x) for x in options],
				"correct_index": options.index(correct),
				"explanation": explanation
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
			s, noun = random.choice(sentences)
			cands = {"cat", "sofa", "Sara", "book", "library", "teacher", "board", "birds", "park", "Tom", "ball"} - {noun}
			opts = [noun] + random.sample(sorted(cands), 3)
			random.shuffle(opts)
			# Pick two distractors to mention
			why_not = ", ".join(opts[i] for i in range(4) if opts[i] != noun)[:40]
			return {
				"question": f"Which word in this sentence is a noun?\n\n“{s}”",
				"options": opts,
				"correct_index": opts.index(noun),
				"explanation": f"A noun names a person, place, or thing. Here, “{noun}” is the noun. Tip: words like {why_not} are not the naming word here."
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
if "asked_set" not in st.session_state:
    st.session_state.asked_set = set()
if "auto_next_active" not in st.session_state:
    st.session_state.auto_next_active = False
if "auto_next_deadline" not in st.session_state:
    st.session_state.auto_next_deadline = None
if "pending_generate" not in st.session_state:
    st.session_state.pending_generate = False
if "pending_params" not in st.session_state:
    st.session_state.pending_params = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = "gpt-4o"
if "embed_model" not in st.session_state:
    st.session_state.embed_model = "text-embedding-3-small"
if "test_openai_result" not in st.session_state:
    st.session_state.test_openai_result = None

# Callback helpers

def _do_generate(subject: str, grade: str, difficulty: int):
    with st.spinner("Retrieving context and generating question..."):
        ctxs = retrieve_skill_context(subject, grade, difficulty)
        result = None
        # Build list of previous questions to discourage repeats
        prev_questions = [h.get("question", "") for h in st.session_state.history]
        # For Math, pick a random type for variety
        question_type = None
        try:
            if subject.lower() == "math":
                import random
                question_type = random.choice(["addition", "subtraction", "multiplication", "division", "word_problem"])
        except Exception:
            question_type = None
        try:
            # Try up to 3 times to avoid repeats
            for _ in range(3):
                result = generate_question_llm(subject, grade, difficulty, ctxs, previous_questions=prev_questions, question_type=question_type)
                if result and result.get("question") not in st.session_state.asked_set:
                    break
                result = None
            if result:
                st.session_state.mode = "openai"
                st.session_state.last_error = None
        except Exception as e:
            st.session_state.mode = "offline"
            st.session_state.last_error = str(e)
            result = None
        if not result:
            if st.session_state.last_error is None:
                st.session_state.last_error = "OpenAI did not return a valid question. Falling back to offline mode."
            # offline fallback with duplicate-avoidance (vary type on retries)
            for _ in range(5):
                candidate = generate_question_offline(subject, grade, difficulty, ctxs, question_type=None)
                if candidate and candidate.get("question") not in st.session_state.asked_set:
                    result = candidate
                    break
                result = candidate  # keep last even if duplicate so user still gets something
        # Record state
        st.session_state.ctxs = ctxs
        st.session_state.result = result
        st.session_state.show_answer = False
        st.session_state.accounted = False
        st.session_state.auto_next_active = False
        st.session_state.auto_next_deadline = None
        # Track asked questions to avoid repeats
        if result and result.get("question"):
            st.session_state.asked_set.add(result["question"])

# Process scheduled generation before rendering widgets
if st.session_state.get("pending_generate") and st.session_state.get("pending_params"):
    _subj, _grade, _diff = st.session_state.pending_params
    st.session_state.pending_generate = False
    st.session_state.pending_params = None
    _do_generate(_subj, _grade, _diff)


def _do_check():
    # Require a selection
    selected = st.session_state.get("selected_option")
    if selected is None or selected == SELECT_PROMPT:
        st.warning("Please select an option before checking.")
        return
    st.session_state.show_answer = True
    # Update score once per question
    if not st.session_state.accounted and st.session_state.result is not None:
        res = st.session_state.result
        correct_idx = int(res["correct_index"])  # ensure int
        correct_opt = res["options"][correct_idx]
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
    # Start auto-next timer (10s)
    st.session_state.auto_next_active = True
    st.session_state.auto_next_deadline = time.time() + 10


def _do_test_openai():
    try:
        # Embeddings check
        _ = client.embeddings.create(
            model=st.session_state.embed_model,
            input=["hello"]
        )
        # Chat check
        resp = client.chat.completions.create(
            model=st.session_state.chat_model,
            temperature=0.0,
            messages=[{"role":"user","content":"Reply with: OK"}]
        )
        msg = (resp.choices[0].message.content or "").strip()
        st.session_state.test_openai_result = f"Embeddings OK; Chat OK (model replied: {msg[:50]})"
        st.session_state.last_error = None
        st.session_state.mode = "openai"
    except Exception as e:
        st.session_state.test_openai_result = None
        st.session_state.last_error = f"OpenAI test failed: {e}"
        st.session_state.mode = "offline"

with st.sidebar:
	st.header("Settings")
	subject = st.selectbox("Subject", ["Math", "ELA"], index=0, key="subject_select")
	grade = st.selectbox("Grade", ["2", "3", "4", "5", "6"], index=3, key="grade_select")
	difficulty = st.slider("Difficulty (1=easy, 3=hard)", min_value=1, max_value=3, value=2, key="difficulty_slider")
	st.text_input("Chat model", key="chat_model")
	st.text_input("Embeddings model", key="embed_model")
	st.button("Test OpenAI", on_click=_do_test_openai)
	if st.session_state.test_openai_result:
		st.success(st.session_state.test_openai_result)
	st.button("Generate Question", on_click=_do_generate, args=(subject, grade, difficulty))
	st.divider()
	st.caption(f"Mode: {st.session_state.mode}")
	if st.session_state.last_error:
		with st.expander("Last error"):
			st.code(st.session_state.last_error)

# Score header
st.caption(f"Score: {st.session_state.score} / {st.session_state.total}")

# Notify user whenever we're using offline mode
if st.session_state.mode == "offline":
    if st.session_state.last_error:
        st.warning("OpenAI connection unavailable. Using offline mode. See 'Last error' in the sidebar for details.")
    else:
        st.warning("OpenAI not used or unavailable right now. Using offline mode.")

if not st.session_state.result:
    st.info("Set your subject, grade, and difficulty in the sidebar, then click 'Generate Question'.")
else:
    result = st.session_state.result
    ctxs = st.session_state.ctxs

    st.subheader("Question")
    st.caption(f"Source: {'OpenAI' if st.session_state.mode == 'openai' else 'Offline'}")
    st.write(result["question"])
    choice = st.radio(
        "Choose an answer:",
        options=[SELECT_PROMPT] + result["options"],
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
        # Show and run countdown to next question
        if st.session_state.auto_next_active and st.session_state.auto_next_deadline:
            remaining = max(0, int(st.session_state.auto_next_deadline - time.time()))
            ph = st.empty()
            for sec in range(remaining, 0, -1):
                ph.info(f"Next question in {sec} seconds…")
                time.sleep(1)
            st.session_state.auto_next_active = False
            # Schedule next question safely and rerun
            st.session_state.pending_generate = True
            st.session_state.pending_params = (st.session_state.subject_select, st.session_state.grade_select, st.session_state.difficulty_slider)
            st.rerun()

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
