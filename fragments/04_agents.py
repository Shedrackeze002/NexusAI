# =====================================================================
# AGENTS - Role-based multi-agent architecture
# =====================================================================
# This cell defines 6 specialized agents that collaborate through a
# shared KnowledgeState (blackboard pattern):
#
#   1. IngestorAgent   - Pre-loads template documents into VectorDB/KG
#   2. RetrievalAgent  - Semantic search over ingested knowledge
#   3. PlannerAgent    - Classifies intent + generates a step-by-step plan
#   4. ExecutorAgent   - Runs the plan (PO generation, email drafting, etc.)
#   5. CriticAgent     - Evaluates output quality (groundedness, adherence)
#   6. OrchestratorAgent - Adaptive control loop that ties it all together
#
# Inter-agent communication uses InterAgentMessage objects, and each
# agent logs its actions via the shared logger and execution trace.
# =====================================================================
import glob
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str, state: KnowledgeState):
        self.name = name
        self.state = state

    @abstractmethod
    def run(self) -> AgentResult:
        pass

    def log(self, action: str, details: str = ""):
        log_agent_action(self.name, action, details)

# ====================
# 1. Ingestor Agent
# ====================
# Scans the Templates directory for .docx files, extracts text from each
# paragraph, and stores it in both the VectorDB (for semantic retrieval)
# and the KnowledgeGraph (for entity relationships).  Runs once at
# startup to "prime" the system's knowledge base.
class IngestorAgent(BaseAgent):
    name: AgentName = "Ingestor"

    def __init__(self, state: KnowledgeState, vector_db: SimpleVectorDB, kg: MockKnowledgeGraph, templates_dir: str):
        super().__init__(self.name, state)
        self.vector_db = vector_db
        self.kg = kg
        self.templates_dir = templates_dir

    @traceable(name="ingestor_run", run_type="chain")
    def run(self) -> AgentResult:
        self.log("START", f"Scanning {self.templates_dir} for knowledge...")
        count = 0
        for root, dirs, files in os.walk(self.templates_dir):
            for file in files:
                if file.endswith(".docx") and not file.startswith("~$"):
                    file_path = os.path.join(root, file)
                    text = self._read_docx(file_path)
                    chunks = self._sliding_window_chunk(text, window_size=500, overlap=100)
                    for i, chunk in enumerate(chunks):
                        metadata = {"source": file, "path": file_path, "chunk_id": i}
                        self.vector_db.add(chunk, metadata)
                        count += 1
        self.log("COMPLETE", f"Ingested {count} chunks from Project Templates.")
        return AgentResult(
            action=AgentAction(agent=self.name, action_type="ingest", reasoning="Initial knowledge load complete."),
            tool_result=None, verification_notes=f"Loaded {count} chunks."
        )

    def _read_docx(self, path: str) -> str:
        try:
            doc = Document(path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip(): full_text.append(para.text)
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            if para.text.strip(): full_text.append(para.text)
            return "\n".join(full_text)
        except Exception as e:
            self.log("ERROR", f"Failed to read {path}: {e}")
            return ""

    def _sliding_window_chunk(self, text: str, window_size: int, overlap: int) -> list[str]:
        if not text: return []
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + window_size
            chunks.append(text[start:end])
            if end >= text_len: break
            start += (window_size - overlap)
        return chunks

# ====================
# 2. Retrieval Agent
# ====================
# Performs semantic search over the VectorDB using the user's raw message
# as the query.  Returns the top-k most relevant document chunks and
# stores them in state.retrieved_chunks for downstream agents to use.
class RetrievalAgent(BaseAgent):
    name: AgentName = "Retrieval"

    def __init__(self, state: KnowledgeState, vector_db: SimpleVectorDB):
        super().__init__(self.name, state)
        self.vector_db = vector_db

    @traceable(name="retrieval_run", run_type="chain")
    def run(self) -> AgentResult:
        query = self.state.raw_email_body
        self.log("SEARCHING", f"Querying VectorDB for: {query[:30]}...")
        results = self.vector_db.search(query, k=3)
        self.state.retrieved_chunks = results
        count = len(results)
        self.log("FOUND", f"{count} relevant chunks.")
        return AgentResult(
            action=AgentAction(agent=self.name, action_type="search", reasoning=f"Found {count} docs"),
            tool_result=None,
            verification_notes=f"Top Match: {results[0][:50]}..." if results else "No matches"
        )

# ====================
# 3. Planner Agent (HW3 - New)
# ====================
# The "brain" of intent classification.  Uses an LLM prompt that includes:
#   - The raw user message
#   - Session context (recent conversation turns from SessionMemory)
#   - User defaults and recent deals from LongTermMemory
# The LLM returns a structured JSON with the classified intent, reasoning,
# and a step-by-step execution plan.  Nine intents are supported:
# GENERATE_PO, EDIT_PO, DRAFT_EMAIL, SEND_EMAIL, CHECK_EMAIL,
# FLAG_FOR_REVIEW, MARK_AS_SPAM, QUERY, GENERAL.
class PlannerAgent(BaseAgent):
    """
    Planning Node: Analyzes incoming messages and produces a structured
    ExecutionPlan. This replaces the old Orchestrator's intent-only classification
    with a richer planning step that lays out the steps the Executor should follow.
    """
    name: AgentName = "Planner"

    def __init__(self, state: KnowledgeState, session_mem: SessionMemory = None, long_term_mem: PersistentLongTermMemory = None):
        super().__init__(self.name, state)
        self.llm = LLMService()
        self.session_mem = session_mem
        self.long_term_mem = long_term_mem

    @traceable(name="planner_run", run_type="chain")
    def run(self) -> AgentResult:
        self.log("THINKING", "Analyzing message and generating execution plan...")
        self.state.add_trace("Planner: Starting intent classification and plan generation")

        # Gather context from memory systems
        session_context = ""
        if self.session_mem:
            session_context = self.session_mem.get_context(last_n=5)

        user_defaults = {}
        recent_deals_str = ""
        if self.long_term_mem:
            user_defaults = self.long_term_mem.get_user_defaults()
            recent_deals = self.long_term_mem.get_recent_deals(limit=3)
            if recent_deals:
                recent_deals_str = "\n".join([
                    f"  - {d.get('product_description','?')} | qty: {d.get('qty','?')} | to: {d.get('X_Destination','?')}"
                    for d in recent_deals
                ])

        prompt = f"""
        You are the Planner for an Autonomous Supply Chain Agent called "Knowledge Nexus".
        Your job is to (1) classify the intent and (2) produce a step-by-step execution plan.

        Incoming Message from: {self.state.sender_email}
        Subject/Source: {self.state.email_subject}
        Content: {self.state.raw_email_body}
        RAG Context: {self.state.retrieved_chunks[:2] if self.state.retrieved_chunks else "No prior context."}
        Session History: {session_context if session_context else "No session history."}
        User Defaults: {json.dumps(user_defaults) if user_defaults else "None stored."}
        Recent Deals: {recent_deals_str if recent_deals_str else "None."}

        Available Intents:
        1. GENERATE_PO: Concrete order details or explicit PO request.
        2. EDIT_PO: User wants to revise, update, or correct a previously generated PO.
        3. DRAFT_EMAIL: User wants to compose, draft, write, or send an email. Also if context suggests an email follow-up (e.g., "send this PO to the buyer").
        4. SEND_EMAIL: User is confirming or approving a previously drafted email. Short replies like "send it", "looks good", "approved" when a pending draft exists.
        5. CHECK_EMAIL: Explicit request to check inbox or unread messages.
        6. FLAG_FOR_REVIEW: Ambiguous, missing specs, high-stakes negotiation.
        7. MARK_AS_SPAM: Junk/Cold sales.
        8. QUERY: Status check or info request (General knowledge).
        9. GENERAL: Greetings, banter, identity, small talk, or anything that does not fit other categories but requires a reply.

        CRITICAL RULE: If the session history shows the Agent just asked for missing details
        (e.g. "I need net weight"), and the user replies with a number or short phrase
        (e.g. "1000kg"), CLASSIFY AS "GENERATE_PO" and plan to merge with prior context.

        EDIT RULE: If the session history shows a PO was just generated and the user says
        "edit", "change", "update", "revise", or "correct" the PO, CLASSIFY AS "EDIT_PO".

        DRAFT/SEND RULE: If session history shows the Agent just presented an email draft
        and the user says "send", "approve", "yes", "looks good", or provides edits to the
        draft, CLASSIFY AS "SEND_EMAIL". If user wants to compose a NEW email, use "DRAFT_EMAIL".

        Respond in this exact JSON format:
        {{
            "intent": "ONE_OF_THE_9_INTENTS",
            "reasoning": "Brief explanation of why this intent was chosen",
            "steps": [
                {{"step_id": 1, "action": "description", "tool": "tool_name_or_null", "expected_output": "what this step produces", "fallback": "what to do if this fails or null"}}
            ]
        }}
        """
        response_text = self.llm.generate(prompt)

        # Parse the structured plan
        intent = "UNKNOWN"
        plan_steps = []
        reasoning = response_text

        try:
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            intent = parsed.get("intent", "UNKNOWN")
            reasoning = parsed.get("reasoning", "")
            raw_steps = parsed.get("steps", [])
            for s in raw_steps:
                plan_steps.append(PlanStep(
                    step_id=s.get("step_id", 0),
                    action=s.get("action", ""),
                    tool=s.get("tool"),
                    expected_output=s.get("expected_output", ""),
                    fallback=s.get("fallback")
                ))
        except (json.JSONDecodeError, Exception) as e:
            self.log("PARSE_FALLBACK", f"Could not parse structured plan: {e}. Using regex fallback.")
            # Fallback: try to extract intent from raw text
            for keyword in ["GENERATE_PO", "EDIT_PO", "DRAFT_EMAIL", "SEND_EMAIL", "CHECK_EMAIL", "FLAG_FOR_REVIEW", "MARK_AS_SPAM", "QUERY", "GENERAL"]:
                if keyword in response_text.upper():
                    intent = keyword
                    break

        # Build the execution plan
        plan = ExecutionPlan(intent=intent, steps=plan_steps, reasoning=reasoning)
        self.state.identified_intent = intent
        self.state.current_plan = plan
        self.state.add_trace(f"Planner decided: {intent} with {len(plan_steps)} steps")

        # Send structured message to Orchestrator
        msg = InterAgentMessage(
            sender="Planner", receiver="Orchestrator",
            message_type="plan",
            payload={"intent": intent, "plan_id": plan.plan_id, "num_steps": len(plan_steps)}
        )
        self.log("DECIDED", f"Intent: {intent} | Plan: {len(plan_steps)} steps | ID: {plan.plan_id}")

        return AgentResult(
            action=AgentAction(agent=self.name, action_type="plan", reasoning=reasoning),
            tool_result=None,
            generated_text=json.dumps({"intent": intent, "steps": [s.action for s in plan_steps]})
        )


# ====================
# 4. Executor Agent (HW3 - Enhanced)
# ====================
# The "hands" of the system.  Dispatches to a dedicated handler based on
# the Planner's classified intent.  Each handler uses LLM prompts for
# data extraction, the PO Generator tool for document creation, and
# the EmailTool for sending.  Handlers:
#   _handle_generate_po  - Extract PO fields from text, fill template
#   _handle_edit_po      - Select a previous PO, apply revisions
#   _handle_draft_email  - Compose an email draft for approval
#   _handle_send_email   - Confirm/edit and send a pending draft
#   _handle_check_email  - Fetch unread inbox messages
#   _handle_query        - Answer supply-chain questions
#   _handle_general      - Handle greetings and small talk
class ExecutorAgent(BaseAgent):
    """
    Execution Node: Takes the plan from PlannerAgent and executes it.
    Handles PO generation, email checks, queries, and general responses.
    This is the refactored version of the old SynthesizerAgent.
    """
    name: AgentName = "Executor"

    def __init__(self, state: KnowledgeState, po_tool: POGenerator,
                 long_term_mem: PersistentLongTermMemory,
                 session_mem: SessionMemory = None,
                 email_tool: Any = None):
        super().__init__(self.name, state)
        self.llm = LLMService()
        self.po_tool = po_tool
        self.long_term_mem = long_term_mem
        self.session_mem = session_mem
        self.email_tool = email_tool

    @traceable(name="executor_run", run_type="chain")
    def run(self) -> AgentResult:
        intent = self.state.identified_intent
        plan = self.state.current_plan
        self.log("EXECUTING", f"Running plan for intent: {intent}")
        self.state.add_trace(f"Executor: Starting execution for {intent}")

        generated_artifacts = []
        response_text = ""
        steps_completed = 0

        if intent == "GENERATE_PO":
            response_text, generated_artifacts, steps_completed = self._handle_generate_po()
        elif intent == "EDIT_PO":
            response_text, generated_artifacts, steps_completed = self._handle_edit_po()
        elif intent == "DRAFT_EMAIL":
            response_text, generated_artifacts, steps_completed = self._handle_draft_email()
        elif intent == "SEND_EMAIL":
            response_text, generated_artifacts, steps_completed = self._handle_send_email()
        elif intent == "CHECK_EMAIL":
            response_text, steps_completed = self._handle_check_email()
        elif intent == "FLAG_FOR_REVIEW":
            response_text = f"This request from {self.state.sender_email} has been flagged for manual review due to ambiguity or missing specifications."
            steps_completed = 1
        elif intent == "MARK_AS_SPAM":
            response_text = (
                "Email marked as SPAM. No action taken.\n\n"
                "[SAFETY WARNING] This email has been identified as spam or a phishing attempt. "
                "Please do NOT click any links, download attachments, or reply to the sender. "
                "If unsure, verify the sender's identity through a separate, trusted channel before taking any action."
            )
            steps_completed = 1
        elif intent == "QUERY":
            response_text, steps_completed = self._handle_query()
        elif intent == "GENERAL":
            response_text, steps_completed = self._handle_general()
        else:
            response_text = "Unknown intent. Unable to process."
            steps_completed = 0

        total_steps = len(plan.steps) if plan else 1
        self.state.add_trace(f"Executor: Completed {steps_completed}/{total_steps} steps")
        self.log("COMPLETE", f"Result: {response_text[:80]}...")

        return AgentResult(
            action=AgentAction(agent=self.name, action_type="execute", reasoning=f"Handled {intent}: {steps_completed}/{total_steps} steps"),
            tool_result=None,
            generated_text=response_text,
            artifacts=generated_artifacts
        )

    def _handle_generate_po(self):
        current_msg = self.state.raw_email_body
        # Build context from session memory
        session_context = ""
        if self.session_mem:
            session_context = self.session_mem.get_context(last_n=5)

        # Load user defaults from long-term memory
        user_defaults = self.long_term_mem.get_user_defaults()
        defaults_str = json.dumps(user_defaults) if user_defaults else "None"

        full_context = f"{session_context}\n\nCurrent Message: {current_msg}" if session_context else current_msg

        prompt = f"""
        Extract PO details into JSON.
        Context: {full_context}
        User Defaults (from prior sessions): {defaults_str}

        INSTRUCTIONS:
        - Prioritize the LATEST message. If user says "Start over" or "New Order", ignore conflicting old history.
        - If User Defaults are available, use them to pre-fill fields the user has not explicitly provided.
        - INTELLIGENT MAPPING: Map user labels directly to schema.
        - Map 'qty' (e.g. "300 bags") to BOTH 'qty' (raw string) AND 'X_Total_Number_Of_Bags' (numeric "300") AND 'X_Standard_Packing' ("Bags").
        - Map Weights to NUMBERS ONLY (e.g. "1000", not "1000 kg").
        - Infer 'X_Destination' from "Destination".
        - Infer 'X_SellerName' and address from context.

        Keys:
        {{
            "X_SellerName": "...", "X_SellerAddress": "...", "X_SellerTaxId": "...", "X_CountryOrigin": "...",
            "X_Destination": "...", "X_CargoDescription": "...", "X_Lot_Numbers": "...", "X_ReferenceNumber": "...",
            "X_hsCode": "...", "X_ProductionDate": "...", "X_Total_Number_Of_Bags": "...", "X_Standard_Packing": "...",
            "X_Total_Gross_Weight": "...", "X_Total_Net_Weight": "...", "X_Container_Numbers": "...",
            "supplier": "...", "product_description": "...", "qty": "..."
        }}
        Use "N/A" if completely missing.
        """
        extraction_text = self.llm.generate(prompt)
        clean_json = extraction_text.replace("```json", "").replace("```", "").strip()
        generated_artifacts = []

        try:
            data = json.loads(clean_json)

            def clean(val):
                if not val: return None
                s = str(val).strip()
                if s.lower() in ["n/a", "none", "unknown", "null", ""]: return None
                return s

            # Hardcode constants
            data['X_hsCode'] = "0901.11 (Coffee)"
            data['X_SellerTaxId'] = "123455789"

            # Smart defaults
            import random
            if not clean(data.get('X_ReferenceNumber')):
                data['X_ReferenceNumber'] = f"REF-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(100,999)}"
            if not clean(data.get('X_Standard_Packing')):
                data['X_Standard_Packing'] = "60kg Jute Bags"
            if not clean(data.get('X_Container_Numbers')):
                data['X_Container_Numbers'] = "PENDING NOMINATION"

            # Fill from user defaults if fields are still missing
            default_map = {
                "seller_name": "X_SellerName",
                "seller_address": "X_SellerAddress",
                "country_origin": "X_CountryOrigin"
            }
            for pref_key, data_key in default_map.items():
                if not clean(data.get(data_key)) and user_defaults.get(pref_key):
                    data[data_key] = user_defaults[pref_key]
                    self.state.add_trace(f"Executor: Filled {data_key} from long-term memory: {user_defaults[pref_key]}")

            # Numeric cleaning
            for w_key in ['X_Total_Gross_Weight', 'X_Total_Net_Weight', 'X_Total_Number_Of_Bags']:
                val = clean(data.get(w_key))
                if val:
                    numeric = "".join(c for c in val if c.isdigit() or c == '.')
                    data[w_key] = numeric if numeric else None
                else:
                    data[w_key] = None

            # Strict validation
            critical_fields = [
                'X_SellerName', 'X_SellerAddress', 'X_CountryOrigin', 'X_Destination',
                'X_CargoDescription', 'X_Total_Gross_Weight', 'X_Total_Net_Weight',
                'qty', 'X_Lot_Numbers', 'X_ProductionDate', 'X_Total_Number_Of_Bags'
            ]
            missing_fields = []
            for f in critical_fields:
                if not clean(data.get(f)):
                    readable_name = f.replace("X_", "").replace("_", " ")
                    missing_fields.append(readable_name)

            if missing_fields:
                response_text = f"I need a few more details to generate the Packing List:\n- " + "\n- ".join(missing_fields)
                return response_text, generated_artifacts, 1
            else:
                template_name = "CCD-EMI-PL.docx"
                tool_res = self.po_tool.execute(template_name, data)
                if tool_res.success:
                    file_path = tool_res.output['file_path']
                    response_text = f"Generated PO: {file_path}"
                    generated_artifacts.append(Artifact(
                        file_name=tool_res.output['file_name'],
                        file_path=file_path, file_type="docx",
                        content_preview=str(data)
                    ))
                    self.long_term_mem.add_deal(data)
                    # Detect and save user preferences for future sessions
                    if self.session_mem:
                        self.session_mem.detect_preferences(data)
                        self.long_term_mem.save_user_preferences(self.session_mem.detected_preferences)
                    return response_text, generated_artifacts, 3
                else:
                    response_text = f"Failed to generate PO: {tool_res.error}"
                    return response_text, generated_artifacts, 2
        except Exception as e:
            return f"Failed to extract order details: {e}", generated_artifacts, 1

    def _handle_edit_po(self):
        """Handle EDIT_PO: identify which PO to edit, then apply revisions."""
        current_msg = self.state.raw_email_body
        session_context = ""
        if self.session_mem:
            session_context = self.session_mem.get_context(last_n=5)

        # Retrieve recent deals for selection
        recent_deals = self.long_term_mem.get_recent_deals(limit=5)
        if not recent_deals:
            return "No previous POs found to edit. Please generate a PO first.", [], 0

        # Build a concise summary of each PO for LLM matching
        po_summaries = []
        for i, deal in enumerate(recent_deals):
            ref = deal.get("X_ReferenceNumber", "N/A")
            product = deal.get("product_description", deal.get("X_CargoDescription", "Unknown"))
            dest = deal.get("X_Destination", "N/A")
            qty = deal.get("qty", "N/A")
            ts = deal.get("timestamp", "N/A")
            po_summaries.append(
                f"  PO #{i+1}: Ref={ref} | Product={product} | Dest={dest} | Qty={qty} | Date={ts}"
            )
        po_list_str = "\n".join(po_summaries)

        self.state.add_trace(f"Executor: EDIT_PO - Found {len(recent_deals)} recent POs for selection")

        # Step 1: Ask LLM to identify which PO the user wants to edit
        identify_prompt = f"""
        The user wants to EDIT a previously generated Purchase Order.
        There are {len(recent_deals)} recent POs in the system:

{po_list_str}

        SESSION CONTEXT (recent conversation):
        {session_context}

        USER'S MESSAGE:
        {current_msg}

        TASK: Determine WHICH PO the user wants to edit.
        Look for clues: reference numbers, product names, destinations, quantities,
        or if the session context shows a specific PO was just generated.

        Respond in this exact JSON format:
        {{
            "matched_po_index": null,
            "confidence": "high/medium/low",
            "reasoning": "why this PO was matched or why no match was found"
        }}

        RULES:
        - matched_po_index: 0-based index into the PO list, or null if no clear match.
        - If session history shows a PO was JUST generated in this conversation, and
          the user does not specify a different one, match that PO (likely index 0).
        - If the user mentions a specific ref number, product, or destination, match it.
        - If the message is generic (e.g., "edit a PO" with no context), set to null.
        - confidence "high" = certain match, "medium" = likely match, "low" = guessing.
        """
        identify_text = self.llm.generate(identify_prompt)
        clean_id = identify_text.replace("```json", "").replace("```", "").strip()

        selected_deal = None
        try:
            id_parsed = json.loads(clean_id)
            po_idx = id_parsed.get("matched_po_index")
            confidence = id_parsed.get("confidence", "low")

            if po_idx is not None and 0 <= po_idx < len(recent_deals) and confidence in ("high", "medium"):
                selected_deal = recent_deals[po_idx]
                self.state.add_trace(
                    f"Executor: EDIT_PO matched PO #{po_idx+1} "
                    f"(Ref={selected_deal.get('X_ReferenceNumber','?')}) with {confidence} confidence"
                )
        except (json.JSONDecodeError, Exception):
            pass

        # If no match, list POs and ask the user to choose
        if selected_deal is None:
            lines = ["Here are your recent POs. Which one would you like to edit?\n"]
            for i, deal in enumerate(recent_deals):
                ref = deal.get("X_ReferenceNumber", "N/A")
                product = deal.get("product_description", deal.get("X_CargoDescription", "Unknown"))
                dest = deal.get("X_Destination", "N/A")
                qty = deal.get("qty", "N/A")
                lines.append(f"{i+1}. Ref: {ref} | {product} | To: {dest} | Qty: {qty}")
            lines.append("\nReply with the PO number (1-5) or reference number.")
            return "\n".join(lines), [], 1

        # Step 2: Apply edits to the selected PO
        return self._apply_po_edits(selected_deal, current_msg, session_context)

    def _apply_po_edits(self, original_data, edit_request, session_context):
        """Apply field-level edits to a selected PO and regenerate the document."""
        prompt = f"""
        The user wants to EDIT a previously generated Purchase Order.

        ORIGINAL PO DATA:
        {json.dumps(original_data, indent=2, default=str)}

        SESSION CONTEXT:
        {session_context}

        USER'S EDIT REQUEST:
        {edit_request}

        TASK: Return a JSON object containing ONLY the fields that should be changed,
        with their NEW values. Use the exact same keys as the original PO.
        If the user's request is vague (e.g., "change the destination"), ask for the
        specific new value instead of guessing.

        RULES:
        - Map weights to NUMBERS ONLY (e.g., "1000", not "1000 kg").
        - Map bag counts to numbers only.
        - If user provides a completely new value, use it directly.
        - If user says "increase by 100", compute the new total.
        - Return ONLY changed fields, not the full PO.

        Respond in this exact JSON format:
        {{
            "changes": {{}},
            "needs_clarification": false,
            "clarification_message": ""
        }}
        """
        extraction_text = self.llm.generate(prompt)
        clean_json = extraction_text.replace("```json", "").replace("```", "").strip()
        generated_artifacts = []

        try:
            parsed = json.loads(clean_json)

            # If clarification is needed, ask the user
            if parsed.get("needs_clarification", False):
                return parsed.get("clarification_message", "Could you specify which fields to change and their new values?"), [], 1

            changes = parsed.get("changes", {})
            if not changes:
                return "I could not determine what changes to make. Please specify which fields to update and their new values.", [], 1

            # Merge changes onto the original data
            updated_data = dict(original_data)
            for key, value in changes.items():
                old_val = updated_data.get(key, "N/A")
                updated_data[key] = value
                self.state.add_trace(f"Executor: EDIT_PO field '{key}': '{old_val}' -> '{value}'")

            # Numeric cleaning (same as generate)
            for w_key in ['X_Total_Gross_Weight', 'X_Total_Net_Weight', 'X_Total_Number_Of_Bags']:
                val = updated_data.get(w_key)
                if val:
                    numeric = "".join(c for c in str(val) if c.isdigit() or c == '.')
                    updated_data[w_key] = numeric if numeric else None

            # Remove internal metadata fields before filling template
            for meta_key in ['timestamp', 'id']:
                updated_data.pop(meta_key, None)

            # Generate a new unique reference number for the edited PO
            # so it is stored as a distinct entry from the original.
            import random
            today_str = datetime.datetime.now().strftime("%Y%m%d")
            new_ref = f"REF-{today_str}-{random.randint(100, 999)}"
            old_ref = updated_data.get('X_ReferenceNumber', '?')
            updated_data['X_ReferenceNumber'] = new_ref
            self.state.add_trace(
                f"Executor: EDIT_PO assigned new ref {new_ref} (was {old_ref})"
            )

            # Regenerate the document
            template_name = "CCD-EMI-PL.docx"
            tool_res = self.po_tool.execute(template_name, updated_data)
            if tool_res.success:
                file_path = tool_res.output['file_path']
                ref = updated_data.get('X_ReferenceNumber', '?')
                response_text = f"Revised PO (Ref: {ref}) generated with updates: {list(changes.keys())}. File: {file_path}"
                generated_artifacts.append(Artifact(
                    file_name=tool_res.output['file_name'],
                    file_path=file_path, file_type="docx",
                    content_preview=f"Edited fields: {list(changes.keys())}"
                ))
                # Save the edited PO as a new deal in long-term memory
                self.long_term_mem.add_deal(updated_data)
                return response_text, generated_artifacts, 3
            else:
                return f"Failed to regenerate PO: {tool_res.error}", generated_artifacts, 2

        except (json.JSONDecodeError, Exception) as e:
            return f"Failed to process edit request: {e}", [], 1

    def _handle_check_email(self):
        if self.email_tool:
            res = self.email_tool.execute("fetch_unread")
            if res.success and res.output:
                emails = res.output
                details = []
                for i, e in enumerate(emails, 1):
                    details.append(f"{i}. From: {e['sender']} | Subj: {e['subject']}")
                return f"Unread Emails ({len(emails)}):\n" + "\n".join(details), 2
            else:
                return "No unread emails found.", 1
        else:
            return "Email Tool not available.", 0

    def _handle_query(self):
        # Include deal data from long-term memory so the LLM can
        # answer PO-related questions (counts, references, etc.).
        deals = self.long_term_mem.get_recent_deals(limit=20)
        deal_summary = "No POs in the system."
        if deals:
            lines = [f"Total POs in system: {len(deals)}"]
            for i, d in enumerate(deals):
                ref = d.get('X_ReferenceNumber', 'N/A')
                product = d.get('product_description', d.get('X_CargoDescription', 'Unknown'))
                dest = d.get('X_Destination', 'N/A')
                qty = d.get('qty', 'N/A')
                lines.append(f"  {i+1}. Ref={ref} | {product} | To: {dest} | Qty: {qty}")
            deal_summary = "\n".join(lines)

        prompt = (
            f"Answer the user's supply-chain query using the documents AND "
            f"the deal/PO data below.\n\n"
            f"Query: {self.state.raw_email_body}\n\n"
            f"Documents:\n{self.state.retrieved_chunks}\n\n"
            f"PO / Deal Data:\n{deal_summary}"
        )
        response_text = self.llm.generate(prompt)
        return response_text, 2

    def _handle_general(self):
        prompt = f"""
        You are the "Knowledge Nexus Supply Chain Agent".
        Your capabilities: Generating Purchase Orders, Tracking Logistics, and answering Supply Chain queries.
        User Message: {self.state.raw_email_body}

        Task: Reply helpfully but STAY IN CHARACTER.
        - Do NOT offer to write poems, stories, or translate languages.
        - Briefly mention you can help with "generating POs" or "checking status".
        - Keep it professional and concise.
        """
        response_text = self.llm.generate(prompt).strip()
        return response_text, 1

    # ------------------------------------------------------------------
    # DRAFT_EMAIL Handler
    # ------------------------------------------------------------------
    def _handle_draft_email(self):
        """Compose an email draft based on user request and session context."""
        current_msg = self.state.raw_email_body
        session_context = ""
        if self.session_mem:
            session_context = self.session_mem.get_context(last_n=5)

        # Gather recent deal info for context-aware drafting
        recent_deal_str = ""
        if self.long_term_mem:
            recent_deals = self.long_term_mem.get_recent_deals(limit=1)
            if recent_deals:
                d = recent_deals[0]
                recent_deal_str = json.dumps({
                    k: v for k, v in d.items()
                    if k not in ("id", "timestamp")
                }, indent=2, default=str)

        # Find the right PO file to attach based on conversation context.
        # Instead of blindly picking the most recent file, we list all
        # available POs and let the LLM match based on the user's request
        # and recent conversation.  This prevents sending the wrong PO.
        attachment_path = None
        output_dir = os.path.join(os.getcwd(), "outputs")
        available_files = []
        if os.path.exists(output_dir):
            docx_files = sorted(
                [f for f in os.listdir(output_dir) if f.endswith(".docx")],
                key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
                reverse=True
            )
            for f in docx_files[:10]:  # cap at 10 most recent files
                mtime = datetime.datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(output_dir, f))
                ).strftime("%Y-%m-%d %H:%M")
                available_files.append({"filename": f, "modified": mtime})

        # Use the LLM to match the right PO file to the conversation context
        if available_files:
            files_list = "\n".join(
                [f"  {i+1}. {af['filename']} (last modified: {af['modified']})"
                 for i, af in enumerate(available_files)]
            )
            match_prompt = f"""Given the conversation context and available PO files, which file should be attached to this email?

CONVERSATION CONTEXT:
{session_context}

USER REQUEST:
{current_msg}

AVAILABLE FILES:
{files_list}

RULES:
- Pick the file that best matches the product, buyer, or PO being discussed.
- If you are confident about the match, respond with ONLY the exact filename.
- If you cannot confidently determine which file to attach, respond with "NONE".
- Do NOT guess. It is critical to attach the correct document.

YOUR ANSWER (filename or NONE):"""
            match_answer = self.llm.generate(match_prompt).strip().strip('"').strip("'")

            if match_answer.upper() != "NONE":
                # Verify the LLM's answer is actually a valid file
                for af in available_files:
                    if af["filename"] in match_answer or match_answer in af["filename"]:
                        attachment_path = os.path.join(output_dir, af["filename"])
                        break


        # Extract real email addresses from session context and state so
        # the LLM can pick the right recipient instead of defaulting to
        # a placeholder.  We scan both session turns and the current
        # state for anything that looks like an email address.
        import re
        known_emails = set()
        email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

        # Scan session turns for email addresses
        if self.session_mem and hasattr(self.session_mem, 'turns'):
            for turn in self.session_mem.turns:
                content = turn.get('content', '') if isinstance(turn, dict) else str(turn)
                found = email_pattern.findall(content)
                known_emails.update(found)

        # Scan current message and state
        known_emails.update(email_pattern.findall(current_msg))
        if self.state.sender_email and '@' in str(self.state.sender_email):
            clean_sender = email_pattern.findall(str(self.state.sender_email))
            known_emails.update(clean_sender)

        # Remove our own email and system addresses from the list
        own_email = os.getenv("EMAIL_ADDRESS", "")
        known_emails.discard(own_email)
        known_emails.discard("recipient@example.com")
        known_emails.discard("mailer-daemon@googlemail.com")

        # Build the known contacts section for the prompt
        contacts_section = "No email addresses found in conversation."
        if known_emails:
            contacts_section = "The following email addresses were found in this conversation:\n"
            contacts_section += "\n".join([f"  - {e}" for e in known_emails])

        prompt = f"""
        You are the email-drafting assistant for the "Knowledge Nexus Supply Chain Agent".

        SESSION CONTEXT (recent conversation):
        {session_context}

        MOST RECENT PO DATA (if available):
        {recent_deal_str if recent_deal_str else "No recent PO found."}

        KNOWN CONTACTS FROM THIS CONVERSATION:
        {contacts_section}

        USER'S REQUEST:
        {current_msg}

        TASK: Draft a professional email based on the user's request and available context.
        - If the user specifies a recipient, use that address.
        - If no recipient is specified, use the most relevant email from KNOWN CONTACTS above
          (e.g., the original sender, buyer, or supplier from the conversation).
        - NEVER use "recipient@example.com". Always pick a real address from known contacts
          or ask the user to provide one.
        - If a PO was recently generated, reference its details (product, quantity, destination,
          reference number) in the email body as appropriate.
        - Keep the tone professional and concise.
        - If the user requests a specific type of email (e.g., shipping confirmation, invoice
          request, PO follow-up), tailor the content accordingly.

        FORMATTING RULES:
        - Write the body as clean PLAIN TEXT only.
        - Do NOT use markdown symbols like *, **, #, or bullet markers.
        - Use simple dashes (-) for list items if needed.
        - Use blank lines to separate paragraphs.
        - The email will be sent as HTML automatically, so just write natural text.

        Respond in this exact JSON format:
        {{
            "to": "actual_email@domain.com",
            "subject": "Email subject line",
            "body": "Full email body text"
        }}
        """
        raw = self.llm.generate(prompt)
        clean_json = raw.replace("```json", "").replace("```", "").strip()

        try:
            draft = json.loads(clean_json)
            to_addr = draft.get("to", "recipient@example.com")
            subject = draft.get("subject", "No Subject")
            body = draft.get("body", "")

            # Store draft in session memory for the SEND_EMAIL step,
            # including the path to the PO file to attach
            if self.session_mem:
                self.session_mem.pending_email_draft = {
                    "to": to_addr,
                    "subject": subject,
                    "body": body,
                    "attachment_path": attachment_path
                }

            self.state.add_trace(f"Executor: DRAFT_EMAIL composed for {to_addr}")

            # Format the preview for the user, with clear attachment info
            attachment_note = ""
            if attachment_path:
                attachment_note = f"\nAttachment: {os.path.basename(attachment_path)}"
            elif available_files:
                # LLM could not confidently match - list options for the user
                file_list = "\n".join(
                    [f"  {i+1}. {af['filename']} ({af['modified']})"
                     for i, af in enumerate(available_files[:5])]
                )
                attachment_note = f"\nNo attachment (could not determine which PO to attach).\nAvailable files:\n{file_list}\nTo attach one, reply: 'attach file 1' or specify a filename."

            preview = (
                f"Here is your draft email:\n\n"
                f"To: {to_addr}\n"
                f"Subject: {subject}{attachment_note}\n"
                f"{'=' * 40}\n"
                f"{body}\n"
                f"{'=' * 40}\n\n"
                f"[PLEASE VERIFY] Recipient: {to_addr}\n"
                f"Reply 'SEND' to confirm, or provide edits.\n"
                f"Examples: 'change recipient to buyer@company.com', 'change subject to ...'"
            )
            return preview, [], 2

        except (json.JSONDecodeError, Exception) as e:
            self.state.add_trace(f"Executor: DRAFT_EMAIL failed to parse LLM output: {e}")
            return "I could not compose the email. Could you provide more details about the recipient, subject, and what you want to say?", [], 1

    # ------------------------------------------------------------------
    # SEND_EMAIL Handler
    # ------------------------------------------------------------------
    def _handle_send_email(self):
        """Confirm, optionally edit, and send a previously drafted email."""
        current_msg = self.state.raw_email_body

        # Check for a pending draft
        if not self.session_mem or not self.session_mem.pending_email_draft:
            return "No pending email draft found. Please ask me to draft an email first (e.g., 'Draft an email to buyer@example.com about the latest PO').", [], 1

        draft = self.session_mem.pending_email_draft

        # Determine if the user wants edits or just straight confirmation
        confirm_keywords = ["send", "send it", "approve", "approved", "yes", "looks good", "confirm", "lgtm", "go ahead"]
        is_pure_confirm = current_msg.strip().lower() in confirm_keywords

        if not is_pure_confirm:
            # User provided edits - apply them via LLM
            edit_prompt = f"""
            The user wants to modify an email draft before sending.

            CURRENT DRAFT:
            To: {draft['to']}
            Subject: {draft['subject']}
            Body: {draft['body']}

            USER'S EDITS:
            {current_msg}

            TASK: Apply the user's requested changes to the draft and return the updated email.
            Only change what the user asks for - keep everything else the same.

            Respond in this exact JSON format:
            {{
                "to": "updated_or_same@example.com",
                "subject": "Updated or same subject",
                "body": "Updated or same body"
            }}
            """
            raw = self.llm.generate(edit_prompt)
            clean_json = raw.replace("```json", "").replace("```", "").strip()
            try:
                updated = json.loads(clean_json)
                draft["to"] = updated.get("to", draft["to"])
                draft["subject"] = updated.get("subject", draft["subject"])
                draft["body"] = updated.get("body", draft["body"])
                self.state.add_trace(f"Executor: SEND_EMAIL applied edits to draft")
            except (json.JSONDecodeError, Exception) as e:
                self.state.add_trace(f"Executor: SEND_EMAIL could not parse edits: {e}")
                return f"I could not apply those edits. Please try again with clearer instructions, or reply 'SEND' to send the original draft.", [], 1

        # Send the email with optional attachment
        if self.email_tool:
            result = self.email_tool.send_email(
                to_email=draft["to"],
                subject=draft["subject"],
                body=draft["body"],
                attachment_path=draft.get("attachment_path")
            )
            # Clear the pending draft
            self.session_mem.pending_email_draft = None

            if result.success:
                attach_note = ""
                if draft.get("attachment_path"):
                    attach_note = f" with attachment '{os.path.basename(draft['attachment_path'])}'"
                self.state.add_trace(f"Executor: SEND_EMAIL sent to {draft['to']}")
                return f"Email sent successfully to {draft['to']} with subject '{draft['subject']}'{attach_note}.", [], 3
            else:
                self.state.add_trace(f"Executor: SEND_EMAIL failed: {result.error}")
                return f"Failed to send email: {result.error}", [], 1
        else:
            # No email tool available - mock response
            self.session_mem.pending_email_draft = None
            self.state.add_trace(f"Executor: SEND_EMAIL (mock) to {draft['to']}")
            return f"[Mock] Email would be sent to {draft['to']} with subject '{draft['subject']}'.", [], 2


# ====================
# 5. Critic Agent (HW3 - New, replaces Evaluator)
# ====================
# The Critic acts as a quality gate.  It scores the Executor's output on
# two dimensions: groundedness (is the output supported by retrieved
# context?) and plan_adherence (did the Executor follow the Planner's
# steps?).  If either score falls below the Orchestrator's thresholds,
# the adaptive loop triggers a RETRY or REPLAN.
class CriticAgent(BaseAgent):
    """
    Critique/Evaluation Node: Reviews the Executor's output against the original
    plan and retrieved context. Produces a CriticVerdict with scores and a
    verdict of APPROVE or REVISE. This drives the adaptive control loop.
    """
    name: AgentName = "Critic"

    def __init__(self, state: KnowledgeState):
        super().__init__(self.name, state)
        self.llm = LLMService()

    @traceable(name="critic_run", run_type="chain")
    def run(self) -> AgentResult:
        if not self.state.trajectory or not self.state.trajectory[-1].generated_text:
            verdict = CriticVerdict(verdict="APPROVE", reasoning="Nothing to evaluate.")
            self.state.current_verdict = verdict
            return AgentResult(
                action=AgentAction(agent=self.name, action_type="skip", reasoning="Nothing to evaluate."),
                tool_result=None
            )

        last_result = self.state.trajectory[-1]
        plan = self.state.current_plan
        self.log("EVALUATING", "Computing groundedness and plan adherence scores...")
        self.state.add_trace("Critic: Starting evaluation of Executor output")

        # Build plan summary for adherence check
        plan_summary = "No plan available."
        if plan and plan.steps:
            plan_summary = " -> ".join([s.action for s in plan.steps])

        prompt = f"""
        You are the Critic for an agentic supply chain system. Evaluate the Executor's output.

        Original Plan: {plan_summary}
        Plan Intent: {plan.intent if plan else 'UNKNOWN'}
        Retrieved Context: {self.state.retrieved_chunks[:2] if self.state.retrieved_chunks else 'None'}
        Executor Output: {last_result.generated_text}

        Evaluate on three dimensions:
        1. Groundedness (0.0-1.0): Is the output factually grounded in the retrieved context?
           - 1.0 = fully grounded, 0.0 = completely hallucinated
        2. Plan Adherence (0.0-1.0): Did the Executor follow the planned steps?
           - 1.0 = all steps completed, 0.0 = plan was ignored
        3. Completeness: Did the output fully address the user's request? (true/false)

        If groundedness < 0.6 OR plan_adherence < 0.5, set verdict to "REVISE" and suggest improvements.
        Otherwise, set verdict to "APPROVE".

        Respond in this exact JSON format:
        {{
            "verdict": "APPROVE or REVISE",
            "groundedness_score": float,
            "plan_adherence_score": float,
            "completeness": true_or_false,
            "reasoning": "string",
            "revision_suggestions": ["suggestion1", "suggestion2"]
        }}
        """
        raw_verdict = self.llm.generate(prompt)

        # Parse verdict
        verdict = CriticVerdict()
        try:
            clean_json = raw_verdict.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            verdict = CriticVerdict(
                verdict=parsed.get("verdict", "APPROVE"),
                groundedness_score=float(parsed.get("groundedness_score", 0.5)),
                plan_adherence_score=float(parsed.get("plan_adherence_score", 0.5)),
                completeness=parsed.get("completeness", True),
                reasoning=parsed.get("reasoning", ""),
                revision_suggestions=parsed.get("revision_suggestions", [])
            )
        except (json.JSONDecodeError, Exception) as e:
            self.log("PARSE_FALLBACK", f"Could not parse verdict JSON: {e}")
            verdict = CriticVerdict(verdict="APPROVE", groundedness_score=0.5, reasoning=raw_verdict[:200])

        # Update state
        self.state.current_verdict = verdict
        last_result.groundedness_score = verdict.groundedness_score
        last_result.verification_notes = verdict.reasoning

        self.state.add_trace(
            f"Critic verdict: {verdict.verdict} | "
            f"Groundedness: {verdict.groundedness_score:.2f} | "
            f"Adherence: {verdict.plan_adherence_score:.2f} | "
            f"Complete: {verdict.completeness}"
        )

        # Send structured message
        msg = InterAgentMessage(
            sender="Critic", receiver="Orchestrator",
            message_type="critique",
            payload={
                "verdict": verdict.verdict,
                "groundedness": verdict.groundedness_score,
                "adherence": verdict.plan_adherence_score
            }
        )
        self.log("VERDICT",
                 f"{verdict.verdict} | G:{verdict.groundedness_score:.2f} "
                 f"A:{verdict.plan_adherence_score:.2f} | {verdict.reasoning[:60]}...")

        return AgentResult(
            action=AgentAction(agent=self.name, action_type="evaluate", reasoning=verdict.reasoning),
            tool_result=None,
            groundedness_score=verdict.groundedness_score,
            verification_notes=verdict.reasoning
        )


# ====================
# 6. Orchestrator (Coordinator Role)
# ====================
# The Orchestrator runs the full adaptive control loop for each message:
#   OBSERVE  -> Load state and context
#   PLAN     -> Call PlannerAgent for intent classification + plan
#   EXECUTE  -> Call ExecutorAgent to run the plan
#   EVALUATE -> Call CriticAgent to score the output
#   DECIDE   -> APPROVE (done), RETRY (re-retrieve + re-execute), or
#               REPLAN (generate a new plan from scratch)
# This loop is bounded by AdaptiveConfig (max_retries, max_replans).
# If all retries and replans are exhausted, the loop ESCALATES.
class OrchestratorAgent(BaseAgent):
    """
    Orchestrator coordinates the Planner -> Executor -> Critic pipeline
    and implements the adaptive control loop (retry, replan, escalate).
    """
    name: AgentName = "Orchestrator"

    def __init__(self, state: KnowledgeState, planner: PlannerAgent,
                 executor: ExecutorAgent, critic: CriticAgent,
                 retrieval: RetrievalAgent,
                 adaptive_config: AdaptiveConfig = None,
                 session_mem: SessionMemory = None):
        super().__init__(self.name, state)
        self.planner = planner
        self.executor = executor
        self.critic = critic
        self.retrieval = retrieval
        self.config = adaptive_config or AdaptiveConfig()
        self.session_mem = session_mem

    @traceable(name="orchestrator_run", run_type="chain")
    def run(self) -> AgentResult:
        """
        Adaptive Control Loop:
        OBSERVE -> PLAN -> EXECUTE -> EVALUATE -> DECIDE
            if groundedness < threshold -> RETRY (re-retrieve + re-execute)
            if adherence < threshold  -> REPLAN (generate new plan + re-execute)
            if max retries exceeded   -> ESCALATE (flag for human review)
            else                     -> APPROVE (deliver response)
        """
        self.state.retry_count = 0
        self.state.replan_count = 0

        # --- OBSERVE ---
        self.state.current_loop_state = LoopState.OBSERVE.value
        self.log("OBSERVE", f"Received message from {self.state.sender_email}")
        self.state.add_trace(f"Orchestrator: OBSERVE - New message received")

        # --- PLAN ---
        self.state.current_loop_state = LoopState.PLAN.value
        log_adaptive_event("PLAN", "Generating execution plan...")
        plan_result = self.planner.run()
        self.state.trajectory.append(plan_result)

        # Track intent for session summary
        if self.session_mem:
            self.session_mem.record_intent(self.state.identified_intent)

        # --- EXECUTE + EVALUATE Loop ---
        final_result = self._execute_evaluate_loop()

        return final_result

    def _execute_evaluate_loop(self) -> AgentResult:
        """The core adaptive loop that retries and replans as needed."""

        while True:
            # --- EXECUTE ---
            self.state.current_loop_state = LoopState.EXECUTE.value
            log_adaptive_event("EXECUTE", f"Attempt {self.state.retry_count + 1}")
            exec_result = self.executor.run()
            self.state.trajectory.append(exec_result)

            # --- EVALUATE ---
            self.state.current_loop_state = LoopState.EVALUATE.value
            log_adaptive_event("EVALUATE", "Critic reviewing output...")
            critic_result = self.critic.run()
            self.state.trajectory.append(critic_result)

            verdict = self.state.current_verdict
            if not verdict:
                # Safety fallback
                self.state.current_loop_state = LoopState.APPROVE.value
                return exec_result

            # --- DECIDE ---
            # Check 1: Groundedness below threshold -> RETRY
            if verdict.groundedness_score < self.config.groundedness_threshold:
                if self.state.retry_count < self.config.max_retries:
                    self.state.retry_count += 1
                    self.state.current_loop_state = LoopState.RETRY.value
                    log_adaptive_event("RETRY",
                        f"Groundedness {verdict.groundedness_score:.2f} < {self.config.groundedness_threshold}. "
                        f"Re-retrieving context (attempt {self.state.retry_count}/{self.config.max_retries})...")
                    self.state.add_trace(
                        f"Orchestrator: RETRY triggered - Groundedness {verdict.groundedness_score:.2f} "
                        f"below threshold {self.config.groundedness_threshold}"
                    )
                    # Re-retrieve with more context
                    self.retrieval.run()
                    if self.session_mem:
                        self.session_mem.record_adaptive_event("retry")
                    continue  # loop back to EXECUTE

            # Check 2: Plan adherence below threshold -> REPLAN
            if verdict.plan_adherence_score < self.config.adherence_threshold:
                if self.state.replan_count < self.config.max_replans:
                    self.state.replan_count += 1
                    self.state.current_loop_state = LoopState.REPLAN.value
                    log_adaptive_event("REPLAN",
                        f"Adherence {verdict.plan_adherence_score:.2f} < {self.config.adherence_threshold}. "
                        f"Re-planning (attempt {self.state.replan_count}/{self.config.max_replans})...")
                    self.state.add_trace(
                        f"Orchestrator: REPLAN triggered - Adherence {verdict.plan_adherence_score:.2f} "
                        f"below threshold {self.config.adherence_threshold}"
                    )
                    # Generate a new plan
                    self.planner.run()
                    if self.session_mem:
                        self.session_mem.record_adaptive_event("replan")
                    continue  # loop back to EXECUTE

            # Check 3: Both checks failed and we are out of retries -> ESCALATE
            if (verdict.verdict == "REVISE" and
                self.state.retry_count >= self.config.max_retries and
                self.state.replan_count >= self.config.max_replans):
                self.state.current_loop_state = LoopState.ESCALATE.value
                log_adaptive_event("ESCALATE",
                    f"Max retries ({self.config.max_retries}) and replans ({self.config.max_replans}) exhausted. "
                    f"Escalating to human review.")
                self.state.add_trace("Orchestrator: ESCALATE - All retry/replan attempts exhausted")
                if self.session_mem:
                    self.session_mem.record_adaptive_event("escalation")
                # Modify the response to include escalation notice
                exec_result.generated_text = (
                    f"{self.config.escalation_message}\n\n"
                    f"Original response: {exec_result.generated_text}"
                )
                return exec_result

            # All checks passed -> APPROVE
            self.state.current_loop_state = LoopState.APPROVE.value
            log_adaptive_event("APPROVE",
                f"Output approved. G:{verdict.groundedness_score:.2f} A:{verdict.plan_adherence_score:.2f}")
            self.state.add_trace(
                f"Orchestrator: APPROVE - Output accepted "
                f"(G:{verdict.groundedness_score:.2f}, A:{verdict.plan_adherence_score:.2f})"
            )
            return exec_result
