"""
Knowledge Nexus v3.0 - Streamlit Demo Frontend
================================================
Presentation-ready chat interface for the Knowledge Nexus multi-agent system.
This app wraps the existing agent pipeline in a polished UI that hides all
source code while demonstrating the system's capabilities.

Run with: streamlit run app.py
"""

import sys
import os
import time
import datetime
import logging
import io

# ---------------------------------------------------------------------------
# Part 1: Path setup - import the agent modules from fragments/
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRAGMENTS_DIR = os.path.join(PROJECT_ROOT, "fragments")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FRAGMENTS_DIR)

# Load environment variables before importing agent modules
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import streamlit as st

# Bridge Streamlit Cloud secrets into os.environ so fragments' os.getenv() works.
try:
    for key in ("GEMINI_API_KEY", "EMAIL_ADDRESS", "EMAIL_PASSWORD",
                "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"):
        if key in st.secrets and key not in os.environ:
            os.environ[key] = st.secrets[key]
except Exception:
    pass
# ---------------------------------------------------------------------------
# Part 2: Page configuration and custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Knowledge Nexus v3.0",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium dark-accented look
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a2332 100%);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #e8eaed !important;
    }

    /* Agent role badges */
    .agent-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        letter-spacing: 0.5px;
    }
    .badge-planner   { background: #1a365d; color: #63b3ed; border: 1px solid #2c5282; }
    .badge-executor  { background: #1a3d23; color: #68d391; border: 1px solid #276749; }
    .badge-critic    { background: #3d1a1a; color: #fc8181; border: 1px solid #742a2a; }
    .badge-orchestrator { background: #3d3a1a; color: #f6e05e; border: 1px solid #744a1a; }
    .badge-retrieval { background: #2d1a3d; color: #b794f4; border: 1px solid #553c7b; }

    /* Intent chip */
    .intent-chip {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        background: linear-gradient(135deg, #2d3748, #4a5568);
        color: #e2e8f0;
        border: 1px solid #718096;
        margin: 4px 0;
    }

    /* Score bar */
    .score-container {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 4px 0;
    }
    .score-label {
        font-size: 0.75rem;
        color: #a0aec0;
        min-width: 100px;
    }
    .score-bar {
        flex: 1;
        height: 8px;
        background: #2d3748;
        border-radius: 4px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    .score-value {
        font-size: 0.75rem;
        color: #e2e8f0;
        font-weight: 600;
        min-width: 40px;
        text-align: right;
    }

    /* Chat messages */
    .stChatMessage { border-radius: 12px !important; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a2332, #243447);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #63b3ed, #b794f4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* Pipeline visualization */
    .pipeline-step {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px;
        transition: all 0.3s ease;
    }
    .pipeline-active {
        background: linear-gradient(135deg, #2b6cb0, #3182ce);
        color: white;
        box-shadow: 0 0 12px rgba(49, 130, 206, 0.4);
    }
    .pipeline-done {
        background: #276749;
        color: #c6f6d5;
    }
    .pipeline-waiting {
        background: #2d3748;
        color: #718096;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Part 2b: Log capture - stores log output for the Logs tab
# ---------------------------------------------------------------------------
class StreamlitLogHandler(logging.Handler):
    """Custom log handler that stores records in session state for display."""
    def emit(self, record):
        try:
            msg = self.format(record)
            if "log_buffer" not in st.session_state:
                st.session_state.log_buffer = []
            st.session_state.log_buffer.append(msg)
            # Cap at 500 lines to prevent memory issues
            if len(st.session_state.log_buffer) > 500:
                st.session_state.log_buffer = st.session_state.log_buffer[-500:]
        except Exception:
            pass


def setup_log_capture():
    """Attach the Streamlit log handler to the root logger."""
    if "log_handler_attached" not in st.session_state:
        handler = StreamlitLogHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S"
        ))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        st.session_state.log_handler_attached = True
        st.session_state.log_buffer = []


# ---------------------------------------------------------------------------
# Part 3: Agent initialization (cached so it runs only once)
# ---------------------------------------------------------------------------
def _load_fragments():
    """
    Execute fragment files sequentially in a shared namespace, exactly
    like Jupyter concatenates notebook cells.  This is necessary because
    02_core.py relies on symbols imported by 01_config.py (e.g. Literal),
    and 03_tools.py relies on symbols from 02_core.py, etc.
    """
    shared_ns = {"__builtins__": __builtins__, "__name__": "__main__"}

    fragment_order = [
        "01_config.py",
        "02_core.py",
        "03_tools.py",
        "04_agents.py",
    ]

    for fname in fragment_order:
        fpath = os.path.join(FRAGMENTS_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, fpath, "exec"), shared_ns)

    return shared_ns


@st.cache_resource(show_spinner="Initializing Knowledge Nexus agents...")
def init_agents():
    """Initialize all agent components. Runs once and is cached."""
    ns = _load_fragments()

    # Core classes
    KnowledgeState = ns["KnowledgeState"]
    SessionMemory = ns["SessionMemory"]
    SimpleVectorDB = ns["SimpleVectorDB"]
    PersistentLongTermMemory = ns["PersistentLongTermMemory"]
    AdaptiveConfig = ns["AdaptiveConfig"]
    LoopState = ns["LoopState"]

    # Agent classes
    IngestorAgent = ns["IngestorAgent"]
    RetrievalAgent = ns["RetrievalAgent"]
    PlannerAgent = ns["PlannerAgent"]
    ExecutorAgent = ns["ExecutorAgent"]
    CriticAgent = ns["CriticAgent"]
    OrchestratorAgent = ns["OrchestratorAgent"]

    # Tool classes
    POGenerator = ns["POGenerator"]
    EmailTool = ns["EmailTool"]
    SlackTool = ns["SlackTool"]

    TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "Templates")

    # Initialize components
    vector_db = SimpleVectorDB()
    long_term_mem = PersistentLongTermMemory()
    state = KnowledgeState()
    session_mem = SessionMemory(session_id=state.session_id, max_turns=10)

    # Mock knowledge graph (required by IngestorAgent)
    class MockKG:
        def add_entity(self, *a, **kw): pass
        def add_relation(self, *a, **kw): pass
    kg = MockKG()

    # Ingest templates into VectorDB
    ingestor = IngestorAgent(state, vector_db, kg, TEMPLATES_DIR)
    ingestor.run()

    # Set up agent pipeline
    retrieval = RetrievalAgent(state, vector_db)
    planner = PlannerAgent(state, session_mem=session_mem, long_term_mem=long_term_mem)
    po_tool = POGenerator(templates_dir=TEMPLATES_DIR)
    email_addr = ns.get("EMAIL_ADDRESS", "")
    email_pass = ns.get("EMAIL_PASSWORD", "")
    email_tool = EmailTool(email_address=email_addr, password=email_pass)
    slack_token = ns.get("SLACK_BOT_TOKEN", "")
    slack_tool = SlackTool(token=slack_token, default_channel="#agenticai-group-9")
    executor = ExecutorAgent(state, po_tool, long_term_mem,
                             session_mem=session_mem, email_tool=email_tool)
    critic = CriticAgent(state)
    adaptive_config = AdaptiveConfig(
        groundedness_threshold=0.6, adherence_threshold=0.5,
        max_retries=2, max_replans=1
    )
    orchestrator = OrchestratorAgent(
        state, planner=planner, executor=executor, critic=critic,
        retrieval=retrieval, adaptive_config=adaptive_config
    )

    return {
        "state": state,
        "session_mem": session_mem,
        "retrieval": retrieval,
        "orchestrator": orchestrator,
        "vector_db": vector_db,
        "long_term_mem": long_term_mem,
        "LoopState": LoopState,
        "TEMPLATES_DIR": TEMPLATES_DIR,
        "ns": ns,
        "email_tool": email_tool,
        "slack_tool": slack_tool,
        "KnowledgeState": KnowledgeState,
        "SessionMemory": SessionMemory,
        "AdaptiveConfig": AdaptiveConfig,
        "RetrievalAgent": RetrievalAgent,
        "PlannerAgent": PlannerAgent,
        "ExecutorAgent": ExecutorAgent,
        "CriticAgent": CriticAgent,
        "OrchestratorAgent": OrchestratorAgent,
        "POGenerator": POGenerator,
        "EmailTool": EmailTool,
        "SlackTool": SlackTool,
    }


# ---------------------------------------------------------------------------
# Part 4: Message processing (the core agent pipeline)
# ---------------------------------------------------------------------------
def process_user_message(message: str, components: dict) -> dict:
    """
    Run a user message through the full agent pipeline and return
    a structured result dictionary for display.
    """
    state = components["state"]
    session_mem = components["session_mem"]
    retrieval = components["retrieval"]
    orchestrator = components["orchestrator"]
    LoopState = components["LoopState"]

    # Record user message in session memory
    if session_mem:
        session_mem.add_turn("user", message[:300])

    # Reset state for new message
    state.raw_email_body = message
    state.sender_email = "demo@presentation.local"
    state.email_subject = "Live Demo"
    state.message_id = ""
    state.identified_intent = "UNKNOWN"
    state.current_plan = None
    state.current_verdict = None
    state.trajectory = []
    state.retrieved_chunks = []
    state.retry_count = 0
    state.replan_count = 0
    state.current_loop_state = LoopState.OBSERVE.value

    # Run retrieval
    retrieval.run()

    # Run the full adaptive pipeline
    result = orchestrator.run()

    # Record agent response
    if session_mem:
        session_mem.add_turn("agent",
            result.generated_text[:500] if result.generated_text else "No response")

    # Track intent
    if session_mem:
        session_mem.record_intent(state.identified_intent)

    # Build result dictionary
    verdict = state.current_verdict

    # Capture trajectory as readable log lines
    trajectory_lines = []
    if state.trajectory:
        for entry in state.trajectory:
            trajectory_lines.append(str(entry))

    return {
        "text": result.generated_text or "No response generated.",
        "intent": state.identified_intent,
        "groundedness": verdict.groundedness_score if verdict else 0.0,
        "adherence": verdict.plan_adherence_score if verdict else 0.0,
        "verdict": verdict.verdict if verdict else "N/A",
        "retries": state.retry_count,
        "replans": state.replan_count,
        "loop_state": state.current_loop_state,
        "artifacts": result.artifacts if result.artifacts else [],
        "plan": state.current_plan,
        "trajectory": trajectory_lines,
    }


def post_result_to_slack(result: dict, source: str, components: dict):
    """
    Post agent response + artifacts to the Slack channel.
    Called automatically after every message processing.
    """
    slack_tool = components.get("slack_tool")
    if not slack_tool or not getattr(slack_tool, 'client', None):
        return  # no live Slack configured

    intent = result.get("intent", "UNKNOWN")
    text = result.get("text", "No response")
    groundedness = result.get("groundedness", 0)
    adherence = result.get("adherence", 0)
    verdict = result.get("verdict", "N/A")
    retries = result.get("retries", 0)
    replans = result.get("replans", 0)

    header = f"[{intent}] Agent Response (via {source})"

    loop_info = ""
    if retries > 0 or replans > 0:
        loop_info = f"\n[Adaptive: {retries} retries, {replans} replans]"

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": header[:150], "emoji": False}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Source:*\n{source}"},
            {"type": "mrkdwn", "text": f"*Intent:*\n{intent}"}
        ]},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": text[:2900] + loop_info}},
        {"type": "context", "elements": [{"type": "mrkdwn",
            "text": f"*Scores:* G:{groundedness:.2f} | A:{adherence:.2f} | Verdict: {verdict}"}]}
    ]

    try:
        slack_tool.post_message(text=header, blocks=blocks, channel="#agenticai-group-9")
    except Exception as e:
        logging.warning(f"Slack post failed: {e}")

    # Upload any generated artifacts
    artifacts = result.get("artifacts", [])
    for art in artifacts:
        file_path = art.file_path if hasattr(art, 'file_path') else str(art)
        file_name = art.file_name if hasattr(art, 'file_name') else os.path.basename(str(art))
        if os.path.exists(file_path):
            try:
                slack_tool.upload_file(file_path, f"Generated: {file_name}", "#agenticai-group-9")
            except Exception as e:
                logging.warning(f"Slack file upload failed: {e}")


# ---------------------------------------------------------------------------
# Part 5: Helper rendering functions
# ---------------------------------------------------------------------------
def render_score_bar(label: str, score: float, color: str = "#63b3ed"):
    """Render a horizontal score bar with label and value."""
    pct = max(0, min(100, int(score * 100)))
    if score >= 0.7:
        bar_color = "#48bb78"
    elif score >= 0.4:
        bar_color = "#ecc94b"
    else:
        bar_color = "#fc8181"

    st.markdown(f"""
    <div class="score-container">
        <span class="score-label">{label}</span>
        <div class="score-bar">
            <div class="score-fill" style="width:{pct}%; background:{bar_color};"></div>
        </div>
        <span class="score-value">{score:.2f}</span>
    </div>
    """, unsafe_allow_html=True)


def render_intent_chip(intent: str):
    """Render an intent classification chip."""
    st.markdown(f'<span class="intent-chip">{intent}</span>', unsafe_allow_html=True)


def render_pipeline_status(loop_state: str, retries: int, replans: int):
    """Render the adaptive pipeline status."""
    states = ["OBSERVE", "PLAN", "EXECUTE", "EVALUATE", "DECIDE"]
    html = ""
    for s in states:
        if s == loop_state:
            cls = "pipeline-active"
        else:
            cls = "pipeline-done"
        html += f'<span class="pipeline-step {cls}">{s}</span> '

    if retries > 0:
        html += f'<span class="pipeline-step" style="background:#744a1a;color:#fbd38d;">RETRY x{retries}</span> '
    if replans > 0:
        html += f'<span class="pipeline-step" style="background:#742a2a;color:#fc8181;">REPLAN x{replans}</span> '

    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(label: str, value: str):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Part 6: Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(components: dict):
    """Render the sidebar with architecture info and system status."""
    with st.sidebar:
        st.markdown("## Knowledge Nexus v3.0")
        st.caption("Multi-Agent Supply Chain System")

        st.divider()

        # System status metrics
        st.markdown("### System Status")
        vdb = components["vector_db"]
        ltm = components["long_term_mem"]
        doc_count = len(vdb.documents) if hasattr(vdb, 'documents') else 0
        deal_count = len(ltm._data.get("deals", [])) if hasattr(ltm, '_data') else 0

        col1, col2 = st.columns(2)
        col1.metric("VectorDB Docs", doc_count)
        col2.metric("Stored Deals", deal_count)

        st.divider()

        # Agent pipeline diagram
        st.markdown("### Agent Pipeline")
        st.markdown("""
        <div style="font-size:0.85rem; line-height:1.8;">
        <span class="agent-badge badge-orchestrator">Orchestrator</span><br/>
        <span style="color:#718096; margin-left:20px;">Adaptive OODA Loop</span><br/><br/>
        <span class="agent-badge badge-planner">Planner</span>
        <span style="color:#718096;">Intent + Plan</span><br/>
        <span class="agent-badge badge-executor">Executor</span>
        <span style="color:#718096;">Tool Use</span><br/>
        <span class="agent-badge badge-critic">Critic</span>
        <span style="color:#718096;">Verify + Score</span><br/>
        <span class="agent-badge badge-retrieval">Retrieval</span>
        <span style="color:#718096;">VectorDB Search</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Architecture summary
        st.markdown("### Adaptive Control")
        st.markdown("""
        ```
        OBSERVE -> PLAN -> EXECUTE
             -> EVALUATE -> DECIDE
        Retry if G < 0.6
        Replan if A < 0.5
        Escalate if max attempts
        ```
        """)

        st.divider()

        # Memory policies
        st.markdown("### Memory Policies")
        st.markdown("""
        **Short-term**: 10-turn window with auto-summarization

        **Long-term**: Persistent deals, preferences, and session summaries
        """)

        st.divider()

        # Live email monitor (auto-refresh fragment)
        st.markdown("### Email Monitor")
        _email_poll_fragment(components)


@st.fragment(run_every="30s")
def _email_poll_fragment(components: dict):
    """
    Auto-polling fragment that checks for new unread emails every 30 seconds.
    Runs independently of the main app, so it does not interrupt the chat.
    Uses st.toast() for floating notifications visible from any tab.
    """
    email_tool = components["email_tool"]

    # Initialize tracking state
    if "poll_email_count" not in st.session_state:
        st.session_state.poll_email_count = 0
    if "poll_email_subjects" not in st.session_state:
        st.session_state.poll_email_subjects = []
    if "poll_last_check" not in st.session_state:
        st.session_state.poll_last_check = "Never"
    if "poll_seen_ids" not in st.session_state:
        st.session_state.poll_seen_ids = set()

    try:
        result = email_tool.fetch_unread()
        now = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.poll_last_check = now

        if result.success and result.output:
            new_count = len(result.output)
            current_ids = set()
            new_subjects = []

            for e in result.output:
                eid = e.get("message_id", e.get("subject", ""))
                current_ids.add(eid)
                if eid not in st.session_state.poll_seen_ids:
                    new_subjects.append(e.get("subject", "(no subject)"))

            # Detect genuinely new emails (not seen before)
            if new_subjects:
                st.session_state.poll_new_alert = True
                # Fire a toast notification visible from any tab
                for subj in new_subjects[:3]:
                    st.toast(f"New email: {subj[:60]}", icon="")
            else:
                st.session_state.poll_new_alert = False

            st.session_state.poll_seen_ids = current_ids
            st.session_state.poll_email_count = new_count
            st.session_state.poll_email_subjects = [
                e.get("subject", "(no subject)") for e in result.output
            ]

            # Push into the Email Inbox tab state so they appear immediately
            st.session_state.fetched_emails = result.output
        else:
            st.session_state.poll_email_count = 0
            st.session_state.poll_email_subjects = []
            st.session_state.poll_new_alert = False
    except Exception:
        pass

    # Display in sidebar
    count = st.session_state.poll_email_count
    last = st.session_state.poll_last_check
    is_new = st.session_state.get("poll_new_alert", False)

    if is_new:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #744a1a, #92400e);
                    border: 1px solid #f6ad55; border-radius: 10px;
                    padding: 10px; text-align: center;
                    animation: pulse 1.5s ease-in-out infinite;">
            <span style="color:#fbd38d; font-weight:700; font-size:1.1rem;">
                NEW MAIL ({count})
            </span>
            <br/>
            <span style="color:#fefcbf; font-size:0.75rem;">
                Check the Email Inbox tab
            </span>
        </div>
        <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        </style>
        """, unsafe_allow_html=True)
    elif count > 0:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a3d23, #276749);
                    border: 1px solid #68d391; border-radius: 10px;
                    padding: 10px; text-align: center;">
            <span style="color:#c6f6d5; font-weight:600; font-size:0.95rem;">
                {count} unread email(s)
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("No unread emails")

    st.caption(f"Last checked: {last}")


# ---------------------------------------------------------------------------
# Part 7: Chat tab
# ---------------------------------------------------------------------------
def render_chat_tab(components: dict):
    """Render the main chat interface."""

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

                # Show agent metadata if available
                meta = msg.get("meta", {})
                if meta:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        render_intent_chip(meta.get("intent", "N/A"))
                    with col2:
                        st.caption(f"Verdict: {meta.get('verdict', 'N/A')}")
                    with col3:
                        retries = meta.get("retries", 0)
                        replans = meta.get("replans", 0)
                        if retries or replans:
                            st.caption(f"Retries: {retries} | Replans: {replans}")

                    render_score_bar("Groundedness", meta.get("groundedness", 0))
                    render_score_bar("Adherence", meta.get("adherence", 0))

                    # Show pipeline status
                    render_pipeline_status(
                        meta.get("loop_state", "APPROVE"),
                        meta.get("retries", 0),
                        meta.get("replans", 0)
                    )

                    # Show artifacts if any
                    artifacts = meta.get("artifacts", [])
                    if artifacts:
                        st.divider()
                        st.markdown("**Generated Artifacts:**")
                        for art in artifacts:
                            file_path = art.file_path if hasattr(art, 'file_path') else str(art)
                            file_name = art.file_name if hasattr(art, 'file_name') else os.path.basename(str(art))
                            if os.path.exists(file_path):
                                with open(file_path, "rb") as f:
                                    st.download_button(
                                        f"Download {file_name}",
                                        f.read(),
                                        file_name=file_name,
                                        key=f"dl_{file_name}_{time.time()}"
                                    )

    # Chat input
    if prompt := st.chat_input("Type a message to the agent..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Process through agent pipeline
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                result = process_user_message(prompt, components)

            st.write(result["text"])

            # Show metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                render_intent_chip(result["intent"])
            with col2:
                st.caption(f"Verdict: {result['verdict']}")
            with col3:
                if result["retries"] or result["replans"]:
                    st.caption(f"Retries: {result['retries']} | Replans: {result['replans']}")

            render_score_bar("Groundedness", result["groundedness"])
            render_score_bar("Adherence", result["adherence"])
            render_pipeline_status(result["loop_state"], result["retries"], result["replans"])

            # Show artifacts
            if result["artifacts"]:
                st.divider()
                st.markdown("**Generated Artifacts:**")
                for art in result["artifacts"]:
                    file_path = art.file_path if hasattr(art, 'file_path') else str(art)
                    file_name = art.file_name if hasattr(art, 'file_name') else os.path.basename(str(art))
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            st.download_button(
                                f"Download {file_name}",
                                f.read(),
                                file_name=file_name,
                                key=f"dl_{file_name}_{time.time()}"
                            )

            # Show trajectory (agent reasoning log)
            if result.get("trajectory"):
                with st.expander("Agent Reasoning Trace", expanded=False):
                    for line in result["trajectory"]:
                        st.text(line)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["text"],
            "meta": {
                "intent": result["intent"],
                "groundedness": result["groundedness"],
                "adherence": result["adherence"],
                "verdict": result["verdict"],
                "retries": result["retries"],
                "replans": result["replans"],
                "loop_state": result["loop_state"],
                "artifacts": result["artifacts"],
            }
        })

        # Auto-post to Slack for cross-channel visibility
        post_result_to_slack(result, "Streamlit Chat", components)


# ---------------------------------------------------------------------------
# Part 8: Evaluation tab
# ---------------------------------------------------------------------------
def render_eval_tab(components: dict):
    """Render the evaluation dashboard."""
    st.markdown("### Automated Evaluation Harness")
    st.markdown("""
    Runs 5 structured test cases through the full **Planner -> Executor -> Critic**
    pipeline and computes three metrics: **Groundedness**, **Task Completion**, and
    **Plan Adherence**.
    """)

    test_descriptions = [
        ("TC-001", "Full PO request with all fields", "GENERATE_PO"),
        ("TC-002", "Email check request", "CHECK_EMAIL"),
        ("TC-003", "General knowledge query (HS codes)", "QUERY"),
        ("TC-004", "Ambiguous/vague request (failure case)", "FLAG_FOR_REVIEW"),
        ("TC-005", "Multi-turn PO with incremental details", "GENERATE_PO"),
    ]

    st.markdown("**Test Cases:**")
    for tid, desc, intent in test_descriptions:
        st.markdown(f"- `{tid}`: {desc} (expected: `{intent}`)")

    st.divider()

    if st.button("Run Evaluation", type="primary", use_container_width=True):
        # Get class references from components
        KnowledgeState = components["KnowledgeState"]
        SessionMemory = components["SessionMemory"]
        AdaptiveConfig = components["AdaptiveConfig"]
        RetrievalAgent = components["RetrievalAgent"]
        PlannerAgent = components["PlannerAgent"]
        ExecutorAgent = components["ExecutorAgent"]
        CriticAgent = components["CriticAgent"]
        OrchestratorAgent = components["OrchestratorAgent"]
        POGenerator = components["POGenerator"]
        EmailTool = components["EmailTool"]

        vector_db = components["vector_db"]
        long_term_mem = components["long_term_mem"]
        TEMPLATES_DIR = components["TEMPLATES_DIR"]

        # We need to import the EvaluationHarness from the notebook
        # Since it is defined in the notebook, we rebuild it here
        # using the same test cases and logic
        progress = st.progress(0, text="Initializing evaluation...")

        test_cases = [
            {
                "test_id": "TC-001",
                "description": "Full PO request with all fields",
                "input_message": (
                    "I need a Purchase Order for 500kg of Arabica Coffee (Emirundi) from Shedrack Eze, "
                    "Kwa Nayinzira Kigali, Rwanda to Dubai UAE. Lot 12, production date Jan 2026, "
                    "350 bags, gross weight 1200kg, net weight 1100kg."
                ),
                "expected_intent": "GENERATE_PO",
                "expected_keywords": ["Generated", "PO"],
                "should_succeed": True,
                "is_multi_turn": False,
                "follow_ups": [],
            },
            {
                "test_id": "TC-002",
                "description": "Email check request",
                "input_message": "Check my emails please",
                "expected_intent": "CHECK_EMAIL",
                "expected_keywords": ["email", "Email"],
                "should_succeed": True,
                "is_multi_turn": False,
                "follow_ups": [],
            },
            {
                "test_id": "TC-003",
                "description": "General knowledge query about HS codes",
                "input_message": "What HS code applies to green coffee beans?",
                "expected_intent": "QUERY",
                "expected_keywords": ["coffee", "0901"],
                "should_succeed": True,
                "is_multi_turn": False,
                "follow_ups": [],
            },
            {
                "test_id": "TC-004",
                "description": "Ambiguous/vague request (FAILURE CASE)",
                "input_message": "Send something to somewhere soon",
                "expected_intent": "FLAG_FOR_REVIEW",
                "expected_keywords": ["review", "ambiguous", "flag"],
                "should_succeed": False,
                "is_multi_turn": False,
                "follow_ups": [],
            },
            {
                "test_id": "TC-005",
                "description": "Multi-turn PO with incremental details",
                "input_message": "I want to place an order for coffee to Dubai",
                "expected_intent": "GENERATE_PO",
                "expected_keywords": ["details", "need"],
                "should_succeed": True,
                "is_multi_turn": True,
                "follow_ups": [
                    "Seller is Shedrack Eze at Kwa Nayinzira Kigali Rwanda",
                    "500kg Arabica, 350 bags, lot 12, gross 1200, net 1100, produced Jan 2026"
                ],
            },
        ]

        results = []
        for idx, tc in enumerate(test_cases):
            progress.progress((idx) / len(test_cases),
                              text=f"Running {tc['test_id']}: {tc['description']}...")

            # Fresh state for each test
            test_state = KnowledgeState()
            test_session = SessionMemory(session_id=test_state.session_id, max_turns=10)
            test_retrieval = RetrievalAgent(test_state, vector_db)
            test_planner = PlannerAgent(test_state, session_mem=test_session, long_term_mem=long_term_mem)
            test_po = POGenerator(templates_dir=TEMPLATES_DIR)
            test_email = EmailTool(email_address="", password="")
            test_executor = ExecutorAgent(test_state, test_po, long_term_mem,
                                          session_mem=test_session, email_tool=test_email)
            test_critic = CriticAgent(test_state)
            test_config = AdaptiveConfig(groundedness_threshold=0.6, adherence_threshold=0.5,
                                         max_retries=2, max_replans=1)
            test_orch = OrchestratorAgent(test_state, planner=test_planner, executor=test_executor,
                                          critic=test_critic, retrieval=test_retrieval,
                                          adaptive_config=test_config)

            try:
                test_state.sender_email = "test@evaluator.com"
                test_state.email_subject = "Evaluation Test"
                test_state.raw_email_body = tc["input_message"]
                test_session.add_turn("user", tc["input_message"])
                test_retrieval.run()
                final_result = test_orch.run()

                # Multi-turn handling
                if tc["is_multi_turn"] and tc["follow_ups"]:
                    for follow_up in tc["follow_ups"]:
                        if final_result.generated_text:
                            test_session.add_turn("agent", final_result.generated_text)
                        test_session.add_turn("user", follow_up)
                        test_state.raw_email_body = follow_up
                        test_state.trajectory = []
                        test_retrieval.run()
                        final_result = test_orch.run()

                verdict = test_state.current_verdict
                output_text = final_result.generated_text or ""

                task_completed = True
                if tc["expected_keywords"]:
                    found = any(kw.lower() in output_text.lower() for kw in tc["expected_keywords"])
                    if not found and tc["should_succeed"]:
                        task_completed = False

                results.append({
                    "test_id": tc["test_id"],
                    "description": tc["description"],
                    "expected_intent": tc["expected_intent"],
                    "actual_intent": test_state.identified_intent,
                    "intent_match": test_state.identified_intent == tc["expected_intent"],
                    "groundedness": verdict.groundedness_score if verdict else 0.0,
                    "adherence": verdict.plan_adherence_score if verdict else 0.0,
                    "completed": task_completed,
                    "retries": test_state.retry_count,
                    "replans": test_state.replan_count,
                    "output": output_text[:150],
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "test_id": tc["test_id"],
                    "description": tc["description"],
                    "expected_intent": tc["expected_intent"],
                    "actual_intent": "ERROR",
                    "intent_match": False,
                    "groundedness": 0.0,
                    "adherence": 0.0,
                    "completed": False,
                    "retries": 0,
                    "replans": 0,
                    "output": str(e)[:150],
                    "error": str(e),
                })

        progress.progress(1.0, text="Evaluation complete!")

        # Display results
        st.divider()
        st.markdown("### Results Summary")

        # Aggregate metrics
        n = len(results)
        avg_g = sum(r["groundedness"] for r in results) / n if n > 0 else 0
        avg_a = sum(r["adherence"] for r in results) / n if n > 0 else 0
        completed = sum(1 for r in results if r["completed"])
        intent_matches = sum(1 for r in results if r["intent_match"])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Avg Groundedness", f"{avg_g:.2f}")
        with col2:
            render_metric_card("Task Completion", f"{completed}/{n}")
        with col3:
            render_metric_card("Avg Adherence", f"{avg_a:.2f}")
        with col4:
            render_metric_card("Intent Accuracy", f"{intent_matches}/{n}")

        st.divider()

        # Results table
        import pandas as pd
        df = pd.DataFrame([
            {
                "Test ID": r["test_id"],
                "Description": r["description"][:35],
                "Intent Match": "YES" if r["intent_match"] else "NO",
                "Groundedness": f"{r['groundedness']:.2f}",
                "Adherence": f"{r['adherence']:.2f}",
                "Completed": "YES" if r["completed"] else "NO",
                "Retries": r["retries"],
            }
            for r in results
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Failure analysis
        failures = [r for r in results if not r["completed"] or r["error"] or not r["intent_match"]]
        if failures:
            st.divider()
            st.markdown("### Failure Case Analysis")
            fc = failures[0]
            st.markdown(f"""
            **Test**: {fc["test_id"]} - {fc["description"]}

            | Metric | Value |
            |--------|-------|
            | Expected Intent | `{fc["expected_intent"]}` |
            | Actual Intent | `{fc["actual_intent"]}` |
            | Groundedness | {fc["groundedness"]:.2f} |
            | Adherence | {fc["adherence"]:.2f} |
            | Retries | {fc["retries"]} |
            | Replans | {fc["replans"]} |

            **Output**: {fc["output"]}
            """)


# ---------------------------------------------------------------------------
# Part 9: Email Inbox tab
# ---------------------------------------------------------------------------
def render_email_tab(components: dict):
    """Render the email inbox with fetch, display, and agent processing."""
    st.markdown("### Email Inbox")
    st.caption("Fetch unread emails from your live Gmail account and process them through the agent.")

    # Initialize email state
    if "fetched_emails" not in st.session_state:
        st.session_state.fetched_emails = []
    if "email_responses" not in st.session_state:
        st.session_state.email_responses = {}

    email_tool = components["email_tool"]

    # Fetch button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Fetch Unread Emails", type="primary", use_container_width=True):
            with st.spinner("Connecting to Gmail..."):
                result = email_tool.fetch_unread()
            if result.success and result.output:
                st.session_state.fetched_emails = result.output
                st.session_state.email_responses = {}
                st.rerun()
            elif result.success:
                st.session_state.fetched_emails = []
                st.info("No unread emails found.")
            else:
                st.error(f"Failed to fetch emails: {result.error}")
    with col2:
        count = len(st.session_state.fetched_emails)
        if count > 0:
            st.success(f"{count} unread email(s) loaded")

    st.divider()

    # Display emails
    emails = st.session_state.fetched_emails
    if not emails:
        st.info("Click 'Fetch Unread Emails' to load your inbox.")
        return

    for idx, em in enumerate(emails):
        sender = em.get("sender", "Unknown")
        subject = em.get("subject", "(no subject)")
        body = em.get("body", "")
        msg_id = em.get("message_id", f"email_{idx}")

        with st.container():
            # Email header row
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a2332, #243447);
                        border: 1px solid #2d3748; border-radius: 12px;
                        padding: 16px; margin-bottom: 8px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:#63b3ed; font-weight:600; font-size:0.9rem;">{sender}</span>
                        <br/>
                        <span style="color:#e2e8f0; font-size:1rem; font-weight:500;">{subject}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Body preview in expander
            with st.expander("View Email Body", expanded=False):
                st.text(body[:1000] if body else "(empty body)")

            # Process button
            btn_key = f"process_email_{idx}"
            if st.button(f"Process with Agent", key=btn_key, use_container_width=True):
                with st.spinner("Running agent pipeline on this email..."):
                    response = process_user_message(
                        body if body else subject,
                        components
                    )
                st.session_state.email_responses[idx] = response
                st.rerun()

            # Show agent response if processed
            if idx in st.session_state.email_responses:
                resp = st.session_state.email_responses[idx]
                st.markdown("---")
                st.markdown("**Agent Response:**")
                st.write(resp["text"])

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    render_intent_chip(resp["intent"])
                with col_b:
                    st.caption(f"Verdict: {resp['verdict']}")
                with col_c:
                    if resp["retries"] or resp["replans"]:
                        st.caption(f"Retries: {resp['retries']} | Replans: {resp['replans']}")

                render_score_bar("Groundedness", resp["groundedness"])
                render_score_bar("Adherence", resp["adherence"])

                # Artifacts
                if resp.get("artifacts"):
                    for art in resp["artifacts"]:
                        file_path = art.file_path if hasattr(art, 'file_path') else str(art)
                        file_name = art.file_name if hasattr(art, 'file_name') else os.path.basename(str(art))
                        if os.path.exists(file_path):
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    f"Download {file_name}",
                                    f.read(),
                                    file_name=file_name,
                                    key=f"dl_email_{idx}_{file_name}"
                                )

                # Trajectory
                if resp.get("trajectory"):
                    with st.expander("Agent Reasoning Trace"):
                        for line in resp["trajectory"]:
                            st.text(line)

            st.divider()


# ---------------------------------------------------------------------------
# Part 10: Logs tab
# ---------------------------------------------------------------------------
def render_logs_tab():
    """Render the live agent logs."""
    st.markdown("### Agent Execution Logs")
    st.caption("Real-time log output from the agent pipeline. Logs update after each interaction.")

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Clear Logs", use_container_width=True):
            st.session_state.log_buffer = []
            st.rerun()

    logs = st.session_state.get("log_buffer", [])
    if logs:
        # Show logs in a scrollable code block
        log_text = "\n".join(logs)
        st.code(log_text, language="log", line_numbers=False)
        st.caption(f"Showing {len(logs)} log entries")
    else:
        st.info("No logs yet. Send a message in the Chat tab to see agent activity here.")


# ---------------------------------------------------------------------------
# Part 13: Slack Messages tab
# ---------------------------------------------------------------------------
def render_slack_tab(components: dict):
    """Render the Slack channel interface with polling and agent processing."""
    st.markdown("### Slack Channel")
    st.caption("View and respond to messages from #agenticai-group-9")

    slack_tool = components["slack_tool"]
    has_live = getattr(slack_tool, 'client', None) is not None

    # Initialize Slack state
    if "slack_messages" not in st.session_state:
        st.session_state.slack_messages = []
    if "slack_responses" not in st.session_state:
        st.session_state.slack_responses = {}
    if "slack_last_ts" not in st.session_state:
        import time as _time
        st.session_state.slack_last_ts = str(_time.time() - 3600)

    if not has_live:
        st.warning(
            "Slack is in mock mode. Add SLACK_BOT_TOKEN to your .env file "
            "or Streamlit Cloud secrets to enable live Slack integration."
        )
        return

    # Controls row
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("Fetch Messages", type="primary", use_container_width=True):
            with st.spinner("Polling Slack channel..."):
                result = slack_tool.fetch_messages(
                    channel="#agenticai-group-9",
                    oldest=st.session_state.slack_last_ts
                )
            if result.success and result.output:
                # Filter out bot messages
                human_msgs = [
                    m for m in result.output
                    if not m.get("bot_id") and m.get("text", "").strip()
                ]
                st.session_state.slack_messages = human_msgs
                st.session_state.slack_responses = {}
                st.rerun()
            elif result.success:
                st.session_state.slack_messages = []
                st.info("No new messages in the channel.")
            else:
                st.error(f"Slack fetch failed: {result.error}")
    with col2:
        if st.button("Send to Slack", use_container_width=True):
            st.session_state.show_slack_compose = True

    # Compose new Slack message
    if st.session_state.get("show_slack_compose", False):
        with st.form("slack_compose"):
            slack_msg = st.text_area("Message to post to #agenticai-group-9:")
            submitted = st.form_submit_button("Post Message")
            if submitted and slack_msg.strip():
                result = slack_tool.post_message(
                    text=slack_msg,
                    channel="#agenticai-group-9"
                )
                if result.success:
                    st.success("Message posted to Slack!")
                    st.session_state.show_slack_compose = False
                else:
                    st.error(f"Failed: {result.error}")

    st.divider()

    # Display messages
    msgs = st.session_state.slack_messages
    if not msgs:
        st.info("Click 'Fetch Messages' to load recent channel activity.")
        return

    st.success(f"{len(msgs)} message(s) from Slack")

    for idx, msg in enumerate(msgs):
        user = msg.get("user", "unknown")
        text = msg.get("text", "")
        ts = msg.get("ts", "")

        # Format timestamp
        try:
            msg_time = datetime.datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
        except (ValueError, TypeError):
            msg_time = "--:--:--"

        with st.container():
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e1e3f, #2d2b55);
                        border: 1px solid #4c3d99; border-radius: 12px;
                        padding: 16px; margin-bottom: 8px;">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#b794f4; font-weight:600;">@{user}</span>
                    <span style="color:#718096; font-size:0.8rem;">{msg_time}</span>
                </div>
                <div style="color:#e2e8f0; margin-top:8px;">{text[:500]}</div>
            </div>
            """, unsafe_allow_html=True)

            # Process button
            if st.button(f"Process with Agent", key=f"slack_process_{idx}",
                         use_container_width=True):
                with st.spinner("Running agent pipeline..."):
                    response = process_user_message(text, components)
                st.session_state.slack_responses[idx] = response
                # Auto-post result back to Slack
                post_result_to_slack(response, f"Slack/@{user}", components)
                st.rerun()

            # Show agent response
            if idx in st.session_state.slack_responses:
                resp = st.session_state.slack_responses[idx]
                st.markdown("---")
                st.markdown("**Agent Response** (also posted to Slack):")
                st.write(resp["text"])

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    render_intent_chip(resp["intent"])
                with col_b:
                    st.caption(f"Verdict: {resp['verdict']}")
                with col_c:
                    if resp["retries"] or resp["replans"]:
                        st.caption(f"Retries: {resp['retries']} | Replans: {resp['replans']}")

                render_score_bar("Groundedness", resp["groundedness"])
                render_score_bar("Adherence", resp["adherence"])

                # Artifacts
                if resp.get("artifacts"):
                    for art in resp["artifacts"]:
                        file_path = art.file_path if hasattr(art, 'file_path') else str(art)
                        file_name = art.file_name if hasattr(art, 'file_name') else os.path.basename(str(art))
                        if os.path.exists(file_path):
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    f"Download {file_name}",
                                    f.read(),
                                    file_name=file_name,
                                    key=f"dl_slack_{idx}_{file_name}"
                                )

            st.divider()


# ---------------------------------------------------------------------------
# Part 14: Main app entry point
# ---------------------------------------------------------------------------
def main():
    """Main application entry point."""
    # Set up log capture before anything else
    setup_log_capture()

    # Initialize agents
    components = init_agents()

    # Render sidebar
    render_sidebar(components)

    # Main area with tabs
    st.markdown("# Knowledge Nexus v3.0")
    st.caption("Multi-Agent Supply Chain System | CMU Agentic AI - Spring 2026")

    chat_tab, email_tab, slack_tab, eval_tab, logs_tab = st.tabs(
        ["Chat with Agent", "Email Inbox", "Slack Channel",
         "Evaluation Dashboard", "Agent Logs"]
    )

    with chat_tab:
        render_chat_tab(components)

    with email_tab:
        render_email_tab(components)

    with slack_tab:
        render_slack_tab(components)

    with eval_tab:
        render_eval_tab(components)

    with logs_tab:
        render_logs_tab()


if __name__ == "__main__":
    main()
