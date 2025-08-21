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
import tempfile
import sqlite3
import hashlib, base64, secrets
import hmac
from services.generation import (
	retrieve_skill_context as gen_retrieve_skill_context,
	generate_question_llm as gen_generate_question_llm,
	generate_question_offline as gen_generate_question_offline,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")
SELECT_PROMPT = "— Select an answer —"

# Ensure Chroma path is writable (use temp dir on cloud if needed)
def _ensure_writable_dir(path: str) -> str:
	try:
		os.makedirs(path, exist_ok=True)
		# simple write test
		probe = os.path.join(path, ".probe")
		with open(probe, "w", encoding="utf-8") as f:
			f.write("ok")
		os.remove(probe)
		return path
	except Exception:
		fallback = os.path.join(tempfile.gettempdir(), "chroma")
		os.makedirs(fallback, exist_ok=True)
		return fallback

CHROMA_PATH = _ensure_writable_dir(CHROMA_PATH)

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

# Create Chroma client (disable telemetry, persist to writable path, cache the resource)
try:
	from chromadb.config import Settings
except Exception:
	Settings = None  # type: ignore

@st.cache_resource(show_spinner=False)
def _get_chroma_client(persist_dir: str):
	# Prefer Client(Settings(...)) to avoid tenant DB errors on some platforms
	if Settings is not None:
		try:
			# Settings signature differs across versions; try verbose first
			settings = None
			try:
				settings = Settings(
					persist_directory=persist_dir,
					anonymized_telemetry=False,
					is_persistent=True,
				)
			except TypeError:
				settings = Settings(persist_directory=persist_dir)
			return chromadb.Client(settings)
		except Exception:
			# Fallback to in-memory
			return chromadb.Client(Settings(anonymized_telemetry=False)) if Settings is not None else chromadb.Client()
	# Very old versions
	return chromadb.Client()

chroma = _get_chroma_client(CHROMA_PATH)
skills = chroma.get_or_create_collection("skills")

# Utilities

def _dedupe_options(data: dict) -> dict:
	try:
		opts = list(map(lambda x: str(x).strip(), data.get("options", [])))
		if len(opts) != 4:
			return data
		seen_lower = set()
		new_opts = []
		for idx, opt in enumerate(opts):
			key = opt.lower()
			if key in seen_lower:
				opt = f"{opt} (option {idx+1})"
			key = opt.lower()
			seen_lower.add(key)
			new_opts.append(opt)
		ci = int(data.get("correct_index", 0))
		correct_text = opts[ci]
		try:
			new_ci = new_opts.index(correct_text)
		except ValueError:
			lower_map = [o.lower() for o in new_opts]
			new_ci = lower_map.index(correct_text.lower()) if correct_text.lower() in lower_map else 0
		data["options"] = new_opts
		data["correct_index"] = new_ci
		return data
	except Exception:
		return data

# Choose next question type based on weakest performance

def _choose_question_type(subject: str) -> Optional[str]:
	try:
		if subject.lower() != "math":
			return None
		user_id = st.session_state.get("user_id") or "guest"
		cur = _db_conn.cursor()
		cur.execute(
			"""
			SELECT question_type,
				SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) AS correct,
				COUNT(*) AS total
			FROM attempts
			WHERE user_id = ? AND subject = 'Math' AND question_type IS NOT NULL
			GROUP BY question_type
			""",
			(user_id,)
		)
		rows = cur.fetchall()
		qtypes = ["addition", "subtraction", "multiplication", "division", "word_problem", "comparison", "order_ops"]
		if not rows:
			import random
			return random.choice(qtypes)
		stats = {r[0]: (r[1] / r[2] if r[2] else 0.0, r[2]) for r in rows if r[0]}
		candidates = []
		for qt in qtypes:
			acc, tot = stats.get(qt, (0.0, 0))
			candidates.append((acc, tot, qt))
		candidates.sort(key=lambda x: (x[0], x[1]))
		return candidates[0][2]
	except Exception:
		return None

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

# Context retrieval via embeddings with keyword fallback

def retrieve_skill_context(subject: str, grade: str, difficulty: int):
	query = f"{subject} grade {grade} difficulty {difficulty}"
	res = None
	try:
		query_emb = client.embeddings.create(
			model=st.session_state.get("embed_model", "text-embedding-3-small"),
			input=[query]
		).data[0].embedding
		res = skills.query(query_embeddings=[query_emb], n_results=3, where={"subject": subject})
	except Exception:
		all_items = skills.get(include=["documents", "metadatas"]) or {}
		all_docs = all_items.get("documents", []) or []
		all_metas = all_items.get("metadatas", []) or []
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

# LLM question generation

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
			data = _shuffle_mc_options(data)
			data = _dedupe_options(data)
			if question_type:
				data["question_type"] = question_type
			return data
		except Exception as e:
			last_err = e
			time.sleep(0.2)
	try:
		if last_err is not None:
			st.session_state.last_error = f"OpenAI error: {last_err}"
	except Exception:
		pass
	return None

# Offline fallback generator with variety

def generate_question_offline(subject: str, grade: str, difficulty: int, contexts, question_type: Optional[str] = None):
	"""Offline generator when OpenAI is unavailable. Includes varied math types."""
	try:
		import random
		random.seed()
		if subject.lower() == "math":
			qtypes = ["addition", "subtraction", "multiplication", "division", "word_problem"]
			qtype = (question_type or _choose_question_type(subject) or random.choice(qtypes)).lower()
			if qtype not in qtypes + ["comparison", "order_ops"]:
				qtype = random.choice(qtypes)
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
				correct = a
				b = random.randint(low, high)
				c = a * b
				question = f"What is {c} ÷ {b}?"
				explanation = f"Since {correct} × {b} = {c}, {c} ÷ {b} = {correct}. Tip: use the inverse of multiplication."
			elif qtype == "word_problem":
				apples_each = a
				bags = b
				correct = apples_each * bags
				question = f"A bag has {apples_each} apples. There are {bags} bags. How many apples are there in total?"
				explanation = f"Multiply groups: {apples_each} apples × {bags} bags = {correct}. Tip: repeated addition also works."
			elif qtype == "comparison":
				correct = 0 if a > b else 1 if a < b else 2
				question = f"Which is greater? {a} or {b}?"
				options_raw = [f"{a} is greater", f"{b} is greater", "They are equal"]
				options = options_raw[:]
				random.shuffle(options)
				return {
					"question": question,
					"options": options,
					"correct_index": options.index(options_raw[correct]),
					"explanation": f"Compare numbers: {max(a,b)} is greater.",
					"question_type": qtype,
				}
			elif qtype == "order_ops":
				correct = a + b * 2
				question = f"What is {a} + {b} × 2?"
				distractors = set()
				while len(distractors) < 3:
					cand = correct + random.choice([-4,-2,-1,1,2,3,4])
					if cand != correct and cand > 0:
						distractors.add(cand)
				options = [correct] + list(distractors)
				random.shuffle(options)
				return {
					"question": question,
					"options": [str(x) for x in options],
					"correct_index": options.index(correct),
					"explanation": f"Do multiplication first: {b} × 2 = {b*2}, then add {a}: total {correct}.",
					"question_type": qtype,
				}
			else:
				correct = a * b
				question = f"What is {a} × {b}?"
				explanation = f"Multiply: {a} × {b} = {correct}. Tip: think of {a} groups of {b}."
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
				"explanation": explanation,
				"question_type": qtype,
			}
		elif subject.lower() == "ela":
			sentences = [
				("The cat slept on the sofa.", "cat"),
				("Sara read a book in the library.", "library"),
				("The teacher wrote on the board.", "teacher"),
				("Birds fly over the park.", "park"),
				("Tom kicked the ball.", "ball"),
			]
			import random
			s, noun = random.choice(sentences)
			cands = {"cat", "sofa", "Sara", "book", "library", "teacher", "board", "birds", "park", "Tom", "ball"} - {noun}
			opts = [noun] + random.sample(sorted(cands), 3)
			random.shuffle(opts)
			why_not = ", ".join(o for o in opts if o != noun)[:40]
			return {
				"question": f"Which word in this sentence is a noun?\n\n“{s}”",
				"options": opts,
				"correct_index": opts.index(noun),
				"explanation": f"A noun names a person, place, or thing. Here, “{noun}” is the noun. Tip: words like {why_not} are not the naming word here.",
			}
		else:
			return None
	except Exception:
		return None

@st.cache_resource(show_spinner=False)
def _get_db():
	conn = sqlite3.connect(os.path.join(BASE_DIR, "app.db"), check_same_thread=False)
	conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS attempts (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id TEXT,
			ts REAL,
			subject TEXT,
			grade TEXT,
			difficulty INTEGER,
			question_type TEXT,
			is_correct INTEGER
		)
		"""
	)
	conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS users (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			username TEXT UNIQUE,
			password_hash TEXT,
			display_name TEXT,
			created_ts REAL
		)
		"""
	)
	return conn

_db_conn = _get_db()

# Password hashing (PBKDF2-SHA256)

def _hash_password(plain: str) -> str:
	salt = secrets.token_bytes(16)
	dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, 200_000)
	return f"pbkdf2$200000${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"

def _verify_password(plain: str, stored: str) -> bool:
	try:
		algo, iter_s, salt_b64, hash_b64 = stored.split("$")
		iters = int(iter_s)
		salt = base64.b64decode(salt_b64)
		expected = base64.b64decode(hash_b64)
		dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, iters)
		return hmac.compare_digest(dk, expected)
	except Exception:
		return False

# Auth helpers

def _signup_user(username: str, password: str, display_name: Optional[str] = None) -> tuple[bool, str]:
	try:
		username = (username or "").strip().lower()
		password = (password or "").strip()
		if not username or not password:
			return False, "Username and password are required."
		hashv = _hash_password(password)
		_ts = time.time()
		_db_conn.execute(
			"INSERT INTO users (username, password_hash, display_name, created_ts) VALUES (?,?,?,?)",
			(username, hashv, display_name or username, _ts),
		)
		_db_conn.commit()
		return True, "Account created."
	except sqlite3.IntegrityError:
		return False, "Username already exists."
	except Exception as e:
		return False, f"Signup failed: {e}"

def _login_user(username: str, password: str) -> tuple[bool, str]:
	try:
		username = (username or "").strip().lower()
		password = (password or "").strip()
		cur = _db_conn.cursor()
		cur.execute("SELECT password_hash FROM users WHERE lower(username)=?", (username,))
		row = cur.fetchone()
		if not row:
			return False, "User not found."
		if not _verify_password(password, row[0]):
			return False, "Incorrect password."
		st.session_state.auth_user = username
		st.session_state.user_id = username
		return True, "Logged in."
	except Exception as e:
		return False, f"Login failed: {e}"

def _logout_user():
	st.session_state.auth_user = None
	st.session_state.user_id = "guest"

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
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# Callback helpers

def _do_generate(subject: str, grade: str, difficulty: int):
    with st.spinner("Retrieving context and generating question..."):
        ctxs = gen_retrieve_skill_context(client, skills, subject, grade, difficulty, st.session_state.get("embed_model", "text-embedding-3-small"))
        result = None
        prev_questions = [h.get("question", "") for h in st.session_state.history]
        question_type = None
        try:
            if subject.lower() == "math":
                import random
                question_type = random.choice(["addition", "subtraction", "multiplication", "division", "word_problem"])
        except Exception:
            question_type = None
        try:
            for _ in range(3):
                result = gen_generate_question_llm(client, subject, grade, difficulty, ctxs, st.session_state.get("chat_model", "gpt-4o"), previous_questions=prev_questions, question_type=question_type)
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
            for _ in range(5):
                candidate = gen_generate_question_offline(subject, grade, difficulty, ctxs, question_type=None)
                if candidate and candidate.get("question") not in st.session_state.asked_set:
                    result = candidate
                    break
                result = candidate
        st.session_state.ctxs = ctxs
        st.session_state.result = result
        st.session_state.show_answer = False
        st.session_state.accounted = False
        st.session_state.auto_next_active = False
        st.session_state.auto_next_deadline = None
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
    selected_index = st.session_state.get("selected_index")
    if selected_index is None or selected_index == -1:
        st.warning("Please select an option before checking.")
        return
    st.session_state.show_answer = True
    # Update score once per question
    if not st.session_state.accounted and st.session_state.result is not None:
        res = st.session_state.result
        correct_idx = int(res["correct_index"])  # ensure int
        correct_opt = res["options"][correct_idx]
        is_correct = int(selected_index) == correct_idx
        st.session_state.total += 1
        if is_correct:
            st.session_state.score += 1
        st.session_state.accounted = True
        st.session_state.history.append({
            "question": res.get("question", ""),
            "selected": res["options"][selected_index] if isinstance(selected_index, int) and selected_index >= 0 else None,
            "correct": correct_opt,
            "is_correct": is_correct,
        })
        # Persist attempt
        try:
            user_id = st.session_state.get("user_id") or "guest"
            ts = time.time()
            subj = st.session_state.subject_select
            grd = st.session_state.grade_select
            diff = int(st.session_state.difficulty_slider)
            qtype = res.get("question_type")
            _db_conn.execute(
                "INSERT INTO attempts (user_id, ts, subject, grade, difficulty, question_type, is_correct) VALUES (?,?,?,?,?,?,?)",
                (user_id, ts, subj, grd, diff, qtype, 1 if is_correct else 0),
            )
            _db_conn.commit()
        except Exception:
            pass
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
	st.header("Account")
	if not st.session_state.get("auth_user"):
		with st.expander("Login", expanded=True):
			li_user = st.text_input("Username", key="li_user")
			li_pass = st.text_input("Password", type="password", key="li_pass")
			if st.button("Sign in"):
				s_ok, s_msg = _login_user(li_user.strip(), li_pass)
				st.toast(s_msg)
		with st.expander("Sign up"):
			su_user = st.text_input("New username", key="su_user")
			su_pass = st.text_input("New password", type="password", key="su_pass")
			su_name = st.text_input("Display name (optional)", key="su_name")
			if st.button("Create account"):
				ok, msg = _signup_user(su_user.strip(), su_pass, su_name.strip() or None)
				st.toast(msg)
				if ok:
					_login_user(su_user.strip(), su_pass)
	else:
		st.caption(f"Signed in as: {st.session_state.auth_user}")
		st.button("Log out", on_click=_logout_user)

	st.header("Settings")
	subject = st.selectbox("Subject", ["Math", "ELA"], index=0, key="subject_select")
	grade = st.selectbox("Grade", ["2", "3", "4", "5", "6"], index=3, key="grade_select")
	difficulty = st.slider("Difficulty (1=easy, 3=hard)", min_value=1, max_value=3, value=2, key="difficulty_slider")
	# Only show manual user id if not logged in
	if not st.session_state.get("auth_user"):
		st.text_input("User ID", key="user_id", placeholder="e.g., alice")
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
    # Index-based radio with placeholder to avoid wrong matches and duplicates
    choice_index = st.radio(
        "Choose an answer:",
        options=[-1, 0, 1, 2, 3],
        index=0,
        key="selected_index",
        format_func=lambda i: SELECT_PROMPT if i == -1 else result["options"][i],
    )
    st.button("Check Answer", on_click=_do_check)

    if st.session_state.show_answer:
        correct_idx = int(result["correct_index"])  # ensure int
        correct_opt = result["options"][correct_idx]
        if choice_index == correct_idx:
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
    with st.expander("Performance (per user)"):
        try:
            user_id = st.session_state.get("user_id") or "guest"
            cur = _db_conn.cursor()
            cur.execute(
                "SELECT subject, COALESCE(question_type,'(any)'), SUM(is_correct), COUNT(*) FROM attempts WHERE user_id=? GROUP BY subject, question_type",
                (user_id,),
            )
            rows = cur.fetchall()
            if not rows:
                st.caption("No attempts yet.")
            else:
                for (subj, qt, correct, total) in rows:
                    acc = (correct / total * 100.0) if total else 0.0
                    st.write(f"{subj} / {qt}: {correct}/{total} correct ({acc:.0f}%)")
        except Exception as _:
            st.caption("Performance unavailable.")
