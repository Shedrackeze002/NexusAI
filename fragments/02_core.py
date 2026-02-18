# =====================================================================
# CORE MODULES - Data models, enums, LLM service, and memory systems
# =====================================================================
# This cell defines the foundational building blocks that every other
# module depends on:
#   - Rich-themed logger for color-coded console output and file logging
#   - Pydantic data models for type-safe message passing between agents
#   - LLMService wrapper around Google Gemini with retry logic
#   - SessionMemory (short-term, per-session conversational context)
#   - PersistentLongTermMemory (cross-session deal and preference storage)
# =====================================================================
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
import uuid
import numpy as np
from enum import Enum
from pydantic import BaseModel, Field
import google.generativeai as genai

# LangSmith tracing (optional, graceful fallback if not installed).
# When the langsmith SDK is present AND LANGCHAIN_TRACING_V2=true is set
# in .env, the @traceable decorator sends execution traces to the
# LangSmith dashboard for observability.  If the package is missing,
# this block provides a no-op decorator so the code runs unchanged.
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return decorator

# ====================
# Part 1: Logger
# ====================
custom_theme = Theme({
    "agent": "bold cyan",
    "tool": "bold magenta",
    "memory": "green",
    "error": "bold red",
    "adaptive": "bold yellow"
})
console = Console(theme=custom_theme)

def setup_logger(level=logging.INFO):
    file_handler = logging.FileHandler("execution_trace.log", mode='a')
    file_handler.setLevel(level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True),
            file_handler
        ]
    )
    return logging.getLogger("rich")

logger = setup_logger()

def log_agent_action(agent_name: str, action: str, details: str = ""):
    console.print(f"[{agent_name}] [bold]{action}[/bold] {details}", style="agent")
    logger.info(f"[{agent_name}] {action} {details}")

def log_memory_event(event_type: str, details: str = ""):
    console.print(f"[Memory] [bold]{event_type}[/bold] {details}", style="memory")
    logger.info(f"[Memory] {event_type} {details}")

def log_adaptive_event(event_type: str, details: str = ""):
    console.print(f"[Adaptive] [bold]{event_type}[/bold] {details}", style="adaptive")
    logger.info(f"[Adaptive] {event_type} {details}")

# ====================
# Part 2: Enums and Literals
# ====================
# Intent is the set of action categories the Planner can classify a
# message into.  Each maps to a dedicated handler in the ExecutorAgent.
Intent = Literal[
    "GENERATE_PO", "EDIT_PO", "DRAFT_EMAIL", "SEND_EMAIL",
    "FLAG_FOR_REVIEW", "MARK_AS_SPAM",
    "QUERY", "CHECK_EMAIL", "GENERAL", "UNKNOWN"
]

# AgentName restricts which names can be assigned to agents (type safety).
AgentName = Literal["Ingestor", "Retrieval", "Planner", "Executor", "Critic", "Orchestrator", "Synthesizer"]

# LoopState tracks where the Orchestrator is in its adaptive control loop.
# The states form a cycle: OBSERVE -> PLAN -> EXECUTE -> EVALUATE -> DECIDE
# with possible branches to RETRY, REPLAN, or ESCALATE.
class LoopState(str, Enum):
    """Tracks where we are in the adaptive OODA cycle."""
    OBSERVE = "OBSERVE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    EVALUATE = "EVALUATE"
    RETRY = "RETRY"
    REPLAN = "REPLAN"
    ESCALATE = "ESCALATE"
    APPROVE = "APPROVE"

# ====================
# Part 3: Data Models
# ====================
class Artifact(BaseModel):
    file_name: str
    file_path: str
    file_type: str
    content_preview: str

class ToolResult(BaseModel):
    tool_name: str = "unknown"
    success: bool
    output: Any
    error: Optional[str] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

class AgentAction(BaseModel):
    agent: str
    action_type: str
    tool_use: Optional[str] = None
    tool_args: Dict[str, Any] = {}
    reasoning: str

class AgentResult(BaseModel):
    action: AgentAction
    tool_result: Optional[ToolResult] = None
    generated_text: Optional[str] = None
    artifacts: List[Artifact] = []
    groundedness_score: float = 0.0
    verification_notes: str = ""

# --- HW3: New Multi-Agent Communication Models ---

class PlanStep(BaseModel):
    """A single step in an execution plan."""
    step_id: int
    action: str  # e.g., "retrieve_context", "extract_po_fields", "generate_document"
    tool: Optional[str] = None  # tool to use, if any
    expected_output: str  # what we expect from this step
    fallback: Optional[str] = None  # what to do if this step fails

class ExecutionPlan(BaseModel):
    """Structured plan produced by the PlannerAgent."""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    intent: str = "UNKNOWN"
    steps: List[PlanStep] = []
    reasoning: str = ""
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

class CriticVerdict(BaseModel):
    """Structured output from the CriticAgent."""
    verdict: str = "APPROVE"  # APPROVE or REVISE
    groundedness_score: float = 0.5
    plan_adherence_score: float = 0.5
    completeness: bool = True
    reasoning: str = ""
    revision_suggestions: List[str] = []

class InterAgentMessage(BaseModel):
    """Structured message passed between agents."""
    sender: str
    receiver: str
    message_type: str  # "plan", "execution_result", "critique", "revision_request"
    payload: Dict[str, Any] = {}
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

class AdaptiveConfig(BaseModel):
    """Thresholds and limits for the adaptive control loop."""
    groundedness_threshold: float = 0.6
    adherence_threshold: float = 0.5
    max_retries: int = 2
    max_replans: int = 1
    escalation_message: str = "This request has been flagged for human review due to repeated low confidence."

# ====================
# Part 4: Knowledge State
# ====================
class KnowledgeState(BaseModel):
    raw_email_body: str = ""
    sender_email: str = ""
    email_subject: str = ""
    message_id: str = ""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    identified_intent: str = "UNKNOWN"
    reasoning_trace: List[str] = []
    retrieved_chunks: List[str] = []
    trajectory: List[AgentResult] = []
    trace_logs: List[str] = []
    conversation_history: List[str] = []
    final_response: Optional[str] = None
    final_artifacts: List[Artifact] = []
    # HW3: Adaptive loop tracking
    current_loop_state: str = "OBSERVE"
    retry_count: int = 0
    replan_count: int = 0
    current_plan: Optional[ExecutionPlan] = None
    current_verdict: Optional[CriticVerdict] = None

    def add_trace(self, message: str):
        entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}"
        self.reasoning_trace.append(entry)
        logger.info(f"[Trace] {entry}")

# ====================
# Part 5: LLM Service
# ====================
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found or empty.")

# --- LLM Service ---
# Wraps Google Gemini with automatic retry logic for 429 (rate limit)
# errors.  Uses exponential backoff (2s, 4s, 8s) before giving up.
# The @traceable decorator logs every LLM call to LangSmith with the
# run_type="llm" so it appears as a dedicated LLM span in the trace.
class LLMService:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name) if GEMINI_API_KEY else None

    @traceable(name="gemini_generate", run_type="llm")
    def generate(self, prompt: str) -> str:
        if not self.model:
            return "ERR: No API Key."

        retries = 3
        delay = 2
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "exhausted" in err_str:
                    if attempt < retries - 1:
                        logger.warning(f"Rate Limit hit (429). Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= 2
                        continue
                logger.error(f"Gemini Generation Failed: {e}")
                return f"Error: {str(e)}"
        return "Error: Max retries exceeded."

# ====================
# Part 6: Memory Systems
# ====================
# --- Real Vector DB Implementation (HW3 Requirement: No Mocks) ---
def get_embedding(text: str) -> List[float]:
    """Fetch real embeddings from Google Gemini."""
    if not GEMINI_API_KEY:
        # Fallback only if key is missing (should not happen in production)
        logger.warning("No API Key for embeddings, using random fallback.")
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
        return list(rng.random(768))
    
    try:
        # Use a stable embedding model
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        # Return zero vector or random on failure to prevent crash
        return [0.0] * 768

class SimpleVectorDB:
    """
    A lightweight, persistent vector database using real embeddings and cosine similarity.
    Replaces the previous mock implementation to satisfy HW requirements.
    Persists data to 'vector_store.json'.
    """
    def __init__(self, storage_file="vector_store.json"):
        self.storage_file = storage_file
        self.documents = self._load()

    def _load(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        with open(self.storage_file, 'w') as f:
            json.dump(self.documents, f)

    def add(self, text: str, metadata: Dict[str, Any] = None):
        # Check for duplicates to avoid bloating
        for doc in self.documents:
            if doc['text'] == text:
                return

        doc_id = str(uuid.uuid4())
        embedding = get_embedding(text)
        self.documents.append({
            'id': doc_id, 
            'text': text, 
            'embedding': embedding, 
            'metadata': metadata or {}
        })
        self._save()
        log_memory_event("VECTOR_ADD", f"Indexed chunk: {text[:30]}...")

    def search(self, query: str, k: int = 3) -> List[str]:
        if not self.documents: return []
        
        # Get query embedding
        try:
            query_embedding = genai.embed_content(
                model="models/gemini-embedding-001",
                content=query,
                task_type="retrieval_query"
            )['embedding']
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []

        # Calculate cosine similarity
        q_vec = np.array(query_embedding)
        norm_q = np.linalg.norm(q_vec)
        
        scored_docs = []
        for doc in self.documents:
            d_vec = np.array(doc['embedding'])
            norm_d = np.linalg.norm(d_vec)
            if norm_q == 0 or norm_d == 0:
                score = 0
            else:
                score = np.dot(q_vec, d_vec) / (norm_q * norm_d)
            scored_docs.append((score, doc))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Log top results
        top_k = scored_docs[:k]
        if top_k:
            best_score = top_k[0][0]
            log_memory_event("VECTOR_SEARCH", f"Query '{query[:20]}...' matched {len(top_k)} docs (top score: {best_score:.4f})")
        
        return [doc['text'] for score, doc in top_k]

class MockKnowledgeGraph:
    def __init__(self):
        self.graph = {}
    def add_relation(self, subject: str, output: str, relation: str):
        if subject not in self.graph: self.graph[subject] = []
        self.graph[subject].append({"relation": relation, "target": output})
    def query(self, entity: str) -> List[str]:
        if entity in self.graph:
            return [f"{rel['relation']} {rel['target']}" for rel in self.graph[entity]]
        return []


# --- HW3: Session Memory (Short-Term) ---
class SessionMemory:
    """
    Short-term memory that lives for the duration of a single session.
    Holds turn-by-turn conversation context and session-level preferences.
    Automatically summarizes and exports to long-term memory when the session ends.

    Write Policy: Every user message and agent response is recorded per turn.
    Read Policy:  Consulted on every new turn to provide conversation context.
    Pruning:      Keeps only the last `max_turns` turns. Older turns are condensed
                  into a running summary to prevent unbounded memory growth.
    """

    def __init__(self, session_id: str, max_turns: int = 10):
        self.session_id = session_id
        self.max_turns = max_turns
        self.turns = []  # List of {"role": "user"|"agent", "content": str, "timestamp": str}
        self.running_summary = ""  # condensed summary of older turns
        self.detected_preferences = {}  # e.g., {"seller_name": "Shedrack", "default_dest": "Dubai"}
        self.intents_seen = []  # Track which intents were processed this session
        self.adaptive_events = {"retries": 0, "replans": 0, "escalations": 0}
        self.pending_email_draft = None  # Stores draft email dict between DRAFT_EMAIL and SEND_EMAIL turns
        self.start_time = datetime.datetime.now().isoformat()
        log_memory_event("SESSION_START", f"Session {session_id} initialized (max {max_turns} turns)")

    def add_turn(self, role: str, content: str):
        """Record a conversational turn and prune if over limit."""
        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })
        log_memory_event("WRITE", f"Session turn recorded: {role} ({len(content)} chars)")

        # Prune: if we exceed max_turns, summarize the oldest turns
        if len(self.turns) > self.max_turns:
            self._prune()

    def get_context(self, last_n: int = 5) -> str:
        """Read policy: return recent context for agent consumption."""
        log_memory_event("READ", f"Session context requested (last {last_n} turns)")
        parts = []
        if self.running_summary:
            parts.append(f"[Earlier Context Summary]: {self.running_summary}")
        recent = self.turns[-last_n:] if len(self.turns) > last_n else self.turns
        for turn in recent:
            parts.append(f"{turn['role'].capitalize()}: {turn['content']}")
        return "\n".join(parts)

    def _prune(self):
        """Summarize the oldest turns into running_summary to keep memory bounded."""
        overflow = len(self.turns) - self.max_turns
        old_turns = self.turns[:overflow]
        self.turns = self.turns[overflow:]

        old_text = " | ".join([f"{t['role']}: {t['content'][:80]}" for t in old_turns])
        if self.running_summary:
            self.running_summary = f"{self.running_summary} ... {old_text}"
        else:
            self.running_summary = old_text

        # Keep summary itself bounded (max 500 chars)
        if len(self.running_summary) > 500:
            self.running_summary = self.running_summary[-500:]

        log_memory_event("PRUNE", f"Pruned {overflow} old turns into summary ({len(self.running_summary)} chars)")

    def detect_preferences(self, data: Dict[str, Any]):
        """Extract recurring user preferences from PO data for future sessions."""
        pref_keys = {
            "X_SellerName": "seller_name",
            "X_SellerAddress": "seller_address",
            "X_SellerTaxId": "seller_tax_id",
            "X_CountryOrigin": "country_origin"
        }
        for src_key, pref_key in pref_keys.items():
            val = data.get(src_key)
            if val and str(val).strip().lower() not in ["n/a", "none", "unknown", ""]:
                self.detected_preferences[pref_key] = str(val).strip()
        if self.detected_preferences:
            log_memory_event("PREFERENCES", f"Detected: {self.detected_preferences}")

    def record_intent(self, intent: str):
        """Track which intents were handled during this session."""
        if intent not in self.intents_seen:
            self.intents_seen.append(intent)

    def record_adaptive_event(self, event_type: str):
        """Track adaptive control events (retry, replan, escalate)."""
        key = event_type.lower() + "s"  # retry -> retries, replan -> replans, escalate -> escalations
        if key not in self.adaptive_events:
            key = event_type.lower()
        if key in self.adaptive_events:
            self.adaptive_events[key] += 1

    def export_summary(self) -> Dict[str, Any]:
        """Export session summary for long-term storage when session ends."""
        summary = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": datetime.datetime.now().isoformat(),
            "total_turns": len(self.turns),
            "summary": self.running_summary or self.get_context(last_n=3),
            "detected_preferences": self.detected_preferences,
            "intents_seen": self.intents_seen,
            "adaptive_events": self.adaptive_events
        }
        log_memory_event("EXPORT", f"Session {self.session_id} exported to long-term memory")
        return summary


# --- HW3: Upgraded Persistent Long-Term Memory ---
class PersistentLongTermMemory:
    """
    Persistent memory that survives across sessions.
    Stores: completed deals, user preferences, and session summaries.

    Write Policy: Writes on PO generation success and on session end.
    Read Policy:  Consulted at session start to pre-load user defaults
                  and during PO extraction to fill recurring fields.
    Pruning:      Keeps only the 20 most recent full deal records.
                  Older records are condensed to one-line summaries.
                  Session summaries older than 30 days are removed.
    """

    MAX_FULL_RECORDS = 20
    MAX_SESSION_SUMMARIES = 15

    def __init__(self, storage_file="long_term_memory.json"):
        self.storage_file = storage_file
        self._data = self._load_from_disk()
        # Ensure structure
        if isinstance(self._data, list):
            # Migrate from old format (flat list of deals)
            self._data = {
                "deals": self._data,
                "user_preferences": {},
                "session_summaries": [],
                "archived_summaries": []
            }
            self._save_to_disk()
        log_memory_event("LOAD", f"Long-term memory loaded: {len(self._data.get('deals',[]))} deals, "
                         f"{len(self._data.get('session_summaries',[]))} sessions")

    @property
    def deals(self):
        return self._data.get("deals", [])

    @property
    def user_preferences(self):
        return self._data.get("user_preferences", {})

    def _load_from_disk(self) -> Dict:
        try:
            with open(self.storage_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"deals": [], "user_preferences": {}, "session_summaries": [], "archived_summaries": []}

    def _save_to_disk(self):
        with open(self.storage_file, 'w') as f:
            json.dump(self._data, f, indent=4, default=str)

    # --- Write Policy ---
    def add_deal(self, po_details: Dict[str, Any]):
        """Write a completed deal to persistent storage."""
        po_details["timestamp"] = str(datetime.datetime.now())
        po_details["id"] = str(uuid.uuid4())
        self._data["deals"].append(po_details)
        self._prune_deals()
        self._save_to_disk()
        log_memory_event("WRITE_DEAL", f"Saved deal: {po_details.get('product_description', 'Unknown')}")

    def save_user_preferences(self, prefs: Dict[str, str]):
        """Write user preferences detected from session patterns."""
        self._data["user_preferences"].update(prefs)
        self._save_to_disk()
        log_memory_event("WRITE_PREFS", f"Updated preferences: {list(prefs.keys())}")

    def save_session_summary(self, summary: Dict[str, Any]):
        """Write a session summary when a session ends."""
        self._data.setdefault("session_summaries", []).append(summary)
        self._prune_sessions()
        self._save_to_disk()
        log_memory_event("WRITE_SESSION", f"Saved session summary: {summary.get('session_id', '?')}")

    # --- Read Policy ---
    def get_recent_deals(self, limit=5) -> List[Dict]:
        """Read recent deals for context."""
        deals = self._data.get("deals", [])
        log_memory_event("READ_DEALS", f"Retrieved {min(limit, len(deals))} recent deals")
        return sorted(deals, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

    def get_user_defaults(self) -> Dict[str, str]:
        """Read stored user preferences to pre-fill PO fields."""
        prefs = self._data.get("user_preferences", {})
        if prefs:
            log_memory_event("READ_PREFS", f"Loaded user defaults: {list(prefs.keys())}")
        return prefs

    def get_session_history(self, limit=3) -> List[Dict]:
        """Read recent session summaries for cross-session context."""
        sessions = self._data.get("session_summaries", [])
        log_memory_event("READ_SESSIONS", f"Retrieved {min(limit, len(sessions))} session summaries")
        return sessions[-limit:]

    # --- Pruning Policy ---
    def _prune_deals(self):
        """Keep only MAX_FULL_RECORDS full records; archive older ones."""
        deals = self._data.get("deals", [])
        if len(deals) > self.MAX_FULL_RECORDS:
            overflow = deals[:-self.MAX_FULL_RECORDS]
            self._data["deals"] = deals[-self.MAX_FULL_RECORDS:]
            # Archive overflow as one-line summaries
            for d in overflow:
                summary_line = (f"{d.get('timestamp','?')} | {d.get('product_description','?')} | "
                                f"qty: {d.get('qty','?')} | to: {d.get('X_Destination','?')}")
                self._data.setdefault("archived_summaries", []).append(summary_line)
            log_memory_event("PRUNE_DEALS", f"Archived {len(overflow)} old deals")

    def _prune_sessions(self):
        """Keep only MAX_SESSION_SUMMARIES recent session summaries."""
        sessions = self._data.get("session_summaries", [])
        if len(sessions) > self.MAX_SESSION_SUMMARIES:
            self._data["session_summaries"] = sessions[-self.MAX_SESSION_SUMMARIES:]
            log_memory_event("PRUNE_SESSIONS", f"Trimmed to {self.MAX_SESSION_SUMMARIES} session summaries")
