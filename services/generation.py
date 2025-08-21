from __future__ import annotations
from typing import Optional, List, Dict, Any
import json
import time

# External deps expected to be provided by caller
# - client: OpenAI client
# - skills: Chroma collection


def _shuffle_mc_options(data: Dict[str, Any]) -> Dict[str, Any]:
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


def _dedupe_options(data: Dict[str, Any]) -> Dict[str, Any]:
	try:
		opts = list(map(lambda x: str(x).strip(), data.get("options", [])))
		if len(opts) != 4:
			return data
		seen_lower = set()
		new_opts: List[str] = []
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


def retrieve_skill_context(client, skills, subject: str, grade: str, difficulty: int, embed_model: str):
	query = f"{subject} grade {grade} difficulty {difficulty}"
	res = None
	try:
		query_emb = client.embeddings.create(
			model=embed_model,
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


def generate_question_llm(client, subject: str, grade: str, difficulty: int, contexts, chat_model: str, previous_questions=None, question_type: Optional[str] = None):
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
				model=chat_model,
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
	return None


def generate_question_offline(subject: str, grade: str, difficulty: int, contexts, question_type: Optional[str] = None):
	"""Offline generator when OpenAI is unavailable. Includes varied math types."""
	try:
		import random
		random.seed()
		if subject.lower() == "math":
			qtypes = ["addition", "subtraction", "multiplication", "division", "word_problem"]
			qtype = (question_type or random.choice(qtypes)).lower()
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
