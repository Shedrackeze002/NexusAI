# =====================================================================
# MAIN LOOP - Orchestrates the entire Knowledge Nexus agent lifecycle
# =====================================================================
# This module is the "entry point" that ties everything together.
# It initializes all components (memory, tools, agents), sets up a
# health-check dashboard, defines the message-processing pipeline,
# and runs an infinite polling loop that listens for new emails and
# Slack messages.  A "Stop Agent" button (via ipywidgets) or a
# KeyboardInterrupt triggers a graceful shutdown that persists session
# data to long-term memory and generates an annotated execution trace.
# =====================================================================
import threading  # for thread-safe stop signaling between the button callback and the main loop

def main_loop():
    logger.info("Initializing Knowledge Nexus System (HW3)...")

    # Track the most recent Slack timestamp so we only fetch NEW messages
    last_slack_ts = str(time.time() - 60)

    # =============================================
    # Part 1: Initialize State & Memory Systems
    # =============================================
    # KnowledgeState is the shared blackboard that every agent reads and
    # writes to during a single message-processing cycle.
    state = KnowledgeState()

    # SimpleVectorDB provides semantic search using real Gemini embeddings.
    # MockKnowledgeGraph stores entity relationships extracted on ingestion.
    vector_db = SimpleVectorDB()
    kg = MockKnowledgeGraph()

    # PersistentLongTermMemory survives across sessions (backed by JSON).
    # SessionMemory is short-term, scoped to this single run, and tracks
    # conversation turns, detected preferences, and pending email drafts.
    long_term_mem = PersistentLongTermMemory()
    session_mem = SessionMemory(session_id=state.session_id)

    # Pre-load any user defaults from long-term memory so returning users
    # get auto-filled PO fields (e.g., seller name, address).
    user_defaults = long_term_mem.get_user_defaults()
    if user_defaults:
        log_memory_event("SESSION_INIT", f"Pre-loaded user defaults from prior sessions: {list(user_defaults.keys())}")
        session_mem.detected_preferences = user_defaults.copy()

    # Load summaries from the last few sessions for cross-session context.
    # This lets the Planner and Executor reference previous interactions.
    prior_sessions = long_term_mem.get_session_history(limit=3)
    if prior_sessions:
        log_memory_event("SESSION_INIT", f"Found {len(prior_sessions)} prior session(s) for context")
        for ps in prior_sessions:
            state.add_trace(f"Prior session {ps.get('session_id','?')[:8]}: {ps.get('summary','')[:100]}")

    # =============================================
    # Part 2: Initialize Tools
    # =============================================
    # EmailTool - IMAP (read unread emails) + SMTP (send drafted emails)
    email_tool = EmailTool(email_address=EMAIL_ADDRESS, password=EMAIL_PASSWORD)

    # SlackTool - post responses and files to the team Slack channel
    slack_tool = SlackTool(token=SLACK_BOT_TOKEN)

    # POGenerator - fills .docx templates with PO data to produce documents
    po_gen_tool = POGenerator(templates_dir=TEMPLATES_DIR)

    # =============================================
    # Part 3: Initialize Agents (Role-Based Architecture)
    # =============================================
    # Each agent has a single responsibility:
    #   Ingestor  - Pre-loads template knowledge into VectorDB and KG
    #   Retrieval - Fetches relevant document chunks for a given query
    #   Planner   - Classifies intent and generates an execution plan
    #   Executor  - Runs the plan (PO generation, email drafting, etc.)
    #   Critic    - Evaluates the Executor's output for quality
    ingestor = IngestorAgent(state, vector_db, kg, TEMPLATES_DIR)
    retrieval = RetrievalAgent(state, vector_db)
    planner = PlannerAgent(state, session_mem=session_mem, long_term_mem=long_term_mem)
    executor = ExecutorAgent(state, po_gen_tool, long_term_mem, session_mem=session_mem, email_tool=email_tool)
    critic = CriticAgent(state)

    # Adaptive control parameters for the Orchestrator's retry/replan loop.
    # If the Critic scores fall below these thresholds, the Orchestrator
    # will retry (re-retrieve + re-execute) or replan (generate a new plan).
    adaptive_config = AdaptiveConfig(
        groundedness_threshold=0.6,
        adherence_threshold=0.5,
        max_retries=2,
        max_replans=1
    )

    # The Orchestrator coordinates the full OBSERVE -> PLAN -> EXECUTE ->
    # EVALUATE -> DECIDE loop, including adaptive retries and replans.
    orchestrator = OrchestratorAgent(
        state, planner=planner, executor=executor,
        critic=critic, retrieval=retrieval,
        adaptive_config=adaptive_config,
        session_mem=session_mem
    )

    # =============================================
    # Part 4: Pre-Load Knowledge (Ingestion)
    # =============================================
    # Scan the Templates directory and ingest any .docx templates into the
    # VectorDB (for semantic retrieval) and KnowledgeGraph (for relations).
    ingestor.run()

    logger.info("System Ready. Listening for emails and Slack messages...")

    # =============================================
    # Health Check Dashboard
    # =============================================
    # Print a summary of which API keys and directories are available so
    # the user can spot configuration issues at a glance.
    print("\n")
    print("="*50)
    print("     KNOWLEDGE NEXUS v3.0 - STATUS CHECK     ")
    print("="*50)

    import os
    keys = ["GEMINI_API_KEY", "SLACK_BOT_TOKEN", "EMAIL_PASSWORD"]
    for k in keys:
        if os.getenv(k):
            print(f"  [OK]  {k:<20} : CONNECTED")
        else:
            print(f"  [--]  {k:<20} : MISSING (Mock Mode)")

    dirs = ["Templates", "outputs"]
    for d in dirs:
        if os.path.exists(d):
            print(f"  [OK]  {d:<20} : FOUND")
        else:
            print(f"  [!!]  {d:<20} : NOT FOUND")

    print("-"*50)
    print(f"  Session ID : {state.session_id}")
    print(f"  Architecture: Planner -> Executor -> Critic (Adaptive Loop)")
    print(f"  Memory     : Session({session_mem.max_turns} turns) + LongTerm({len(long_term_mem.deals)} deals)")
    print(f"  Adaptive   : G>{adaptive_config.groundedness_threshold} A>{adaptive_config.adherence_threshold} "
          f"Retry:{adaptive_config.max_retries} Replan:{adaptive_config.max_replans}")
    print("="*50)
    print("\n")

    # =============================================
    # Part 5: Message Processing (Adaptive Loop)
    # =============================================
    # This inner function is called once per incoming message (from email
    # or Slack).  It resets the shared state, records the message in
    # session memory, runs the full Orchestrator pipeline, and posts
    # the result back to Slack with rich Block Kit formatting.
    @traceable(name="process_message", run_type="chain")
    def process_message(sender, subject, body, message_id, origin="email"):
        logger.info(f"Processing Message from {sender} ({origin}): {subject}")

        # 1. Record the user message in session memory for context
        if session_mem:
            session_mem.add_turn("user", f"[{origin}] {body[:300]}")

        # 2. Reset the shared state for this new message
        state.raw_email_body = body
        state.sender_email = sender
        state.email_subject = subject
        state.message_id = message_id
        state.identified_intent = "UNKNOWN"
        state.current_plan = None
        state.current_verdict = None
        state.trajectory = []
        state.retrieved_chunks = []
        state.retry_count = 0
        state.replan_count = 0
        state.current_loop_state = LoopState.OBSERVE.value

        # 3. Run the full adaptive pipeline:
        #    OBSERVE -> PLAN -> EXECUTE -> EVALUATE -> DECIDE (retry/replan/approve)
        result = orchestrator.run()

        # 4. Record the agent response in session memory so subsequent
        #    turns have conversational context
        if session_mem:
            session_mem.add_turn("agent", result.generated_text[:500] if result.generated_text else "No response")

        # 5. Track which intents were processed (for session summary export)
        intent = state.identified_intent
        if session_mem:
            session_mem.record_intent(intent)

        # 6. Track adaptive events (retries, replans, escalations)
        if session_mem:
            for _ in range(state.retry_count):
                session_mem.record_adaptive_event("retry")
            for _ in range(state.replan_count):
                session_mem.record_adaptive_event("replan")
            if state.current_loop_state == LoopState.ESCALATE.value:
                session_mem.record_adaptive_event("escalate")

        # 7. Build the Slack response with rich Block Kit formatting
        verdict = state.current_verdict
        body_text = result.generated_text or "No response generated."

        if result.artifacts:
            for art in result.artifacts:
                body_text += f"\n> Artifact: `{art.file_name}` ({art.file_type})"
                # Upload generated files (e.g., PO documents) to Slack
                try:
                    slack_tool.upload_file(art.file_path, f"Generated: {art.file_name}", "#agenticai-group-9")
                except Exception as e:
                    logger.warning(f"Could not upload artifact: {e}")

        # Set the header text based on the identified intent
        header_text = f"[{intent}] Response to {sender}"
        if intent == "MARK_AS_SPAM":
            header_text = "[Marked as SPAM/Ignore]"

        # Include adaptive loop metrics if retries or replans occurred
        loop_info = ""
        if state.retry_count > 0 or state.replan_count > 0:
            loop_info = f"\n[Adaptive: {state.retry_count} retries, {state.replan_count} replans]"

        # Build a Gmail deep-link so the user can jump to the original email
        context_link = ""
        if origin == "email" and state.message_id:
            clean_id = str(state.message_id).strip("<>")
            encoded_id = urllib.parse.quote(clean_id)
            context_link = f"https://mail.google.com/mail/u/0/#search/rfc822msgid%3A{encoded_id}"

        preview_text = f">{state.raw_email_body[:200]}..." if state.raw_email_body else "No content"

        # Assemble the Slack Block Kit message
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": header_text, "emoji": False}},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Source:*\n{sender}"},
                {"type": "mrkdwn", "text": f"*Intent:*\n{intent}"}
            ]},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Preview:*\n{preview_text}"}},
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn", "text": body_text + loop_info}}
        ]
        if context_link and origin == "email":
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"<{context_link}|View Original Email>"}})

        if verdict:
            score_text = f"*Scores:* G:{verdict.groundedness_score:.2f} | A:{verdict.plan_adherence_score:.2f} | Verdict: {verdict.verdict}"
            blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": score_text}]})

        slack_tool.post_message(text=header_text, blocks=blocks, channel="#agenticai-group-9")

    # =============================================
    # Part 6: Polling Loop (with Stop button)
    # =============================================

    # Graceful shutdown helper - called by both the Stop button and
    # KeyboardInterrupt.  Exports the session summary (including intents,
    # adaptive events, and detected preferences) into long-term memory,
    # then runs clean_trace.py to produce the annotated execution trace.
    def _shutdown():
        logger.info("Shutting down... Saving session summary to long-term memory.")
        session_summary = session_mem.export_summary()
        long_term_mem.save_session_summary(session_summary)
        if session_mem.detected_preferences:
            long_term_mem.save_user_preferences(session_mem.detected_preferences)
        logger.info("Session saved. Goodbye.")

        # Auto-generate annotated trace file from the raw execution log
        try:
            import subprocess, sys
            # clean_trace.py is now in the same directory (fragments/)
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_trace.py")
            if os.path.exists(script_path):
                subprocess.run([sys.executable, script_path], check=True)
                logger.info("Annotated trace written to execution_trace_annotated.log")
        except Exception as e:
            logger.warning(f"Could not generate annotated trace: {e}")

    # Thread-safe stop signal.  The button callback sets this event from
    # the main thread, and the background polling thread checks it.
    _stop_event = threading.Event()

    # The polling loop runs in a BACKGROUND THREAD.  It takes the
    # stop_event as an explicit parameter to avoid Python closure
    # rebinding issues when starting/stopping multiple times.
    def _polling_loop(stop_event):
        nonlocal last_slack_ts
        try:
            while not stop_event.is_set():
                # -- Email polling --
                try:
                    unread_res = email_tool.execute("fetch_unread")
                    if unread_res.success and unread_res.output:
                        for email_data in unread_res.output:
                            if stop_event.is_set():
                                break
                            process_message(
                                email_data['sender'], email_data['subject'],
                                email_data['body'], email_data.get('message_id', ''), "email"
                            )
                except Exception as e:
                    logger.warning(f"Email polling error: {e}")

                if stop_event.is_set():
                    break

                # -- Slack polling --
                try:
                    slack_res = slack_tool.fetch_messages(channel="#agenticai-group-9", oldest=last_slack_ts)
                    if slack_res.success and slack_res.output:
                        max_ts = last_slack_ts
                        for msg in slack_res.output:
                            ts = msg.get('ts')
                            if ts and (max_ts is None or float(ts) > float(max_ts)): max_ts = ts
                            if msg.get("subtype") == "bot_message" or "bot_id" in msg: continue
                            if stop_event.is_set():
                                break
                            process_message(f"SlackUser:{msg.get('user')}", f"Slack Chat", msg.get("text", ""), ts, "slack")
                        last_slack_ts = max_ts
                except Exception as e:
                    logger.warning(f"Slack polling error: {e}")

                # Wait ~10 seconds between polls, but wake up instantly
                # if the stop event is set.
                stop_event.wait(timeout=10)

            # Loop exited - run graceful shutdown
            _shutdown()
        except Exception as e:
            logger.error(f"Polling loop crashed: {e}")
            import traceback
            traceback.print_exc()

    # =============================================
    # Part 7: Start / Stop Buttons
    # =============================================
    # Mutable state container so button callbacks can share state
    # without nonlocal issues across nested scopes.
    global _agent_state
    _agent_state = {
        "stop_event": None,
        "thread": None,
    }

    try:
        from ipywidgets import Button, Output, HBox
        from IPython.display import display

        start_btn = Button(description="Start Agent", button_style="success", icon="play")
        stop_btn = Button(description="Stop Agent", button_style="danger", icon="stop")
        stop_btn.disabled = True
        btn_output = Output()

        def _on_start_click(btn):
            try:
                # Create a fresh stop event for this run
                ev = threading.Event()
                _agent_state["stop_event"] = ev

                t = threading.Thread(target=_polling_loop, args=(ev,), daemon=True)
                _agent_state["thread"] = t
                t.start()

                start_btn.disabled = True
                stop_btn.disabled = False
                stop_btn.description = "Stop Agent"
                with btn_output:
                    btn_output.clear_output()
                    print("Agent started. Polling email and Slack in background.")
            except Exception as e:
                with btn_output:
                    btn_output.clear_output()
                    print(f"ERROR starting agent: {e}")
                    import traceback
                    traceback.print_exc()

        def _on_stop_click(btn):
            try:
                ev = _agent_state.get("stop_event")
                if ev:
                    ev.set()
                stop_btn.description = "Stopping..."
                stop_btn.disabled = True
                with btn_output:
                    btn_output.clear_output()
                    print("Stop requested. Agent will shut down after the current cycle.")

                # Re-enable Start button once the thread finishes
                def _wait_and_enable():
                    t = _agent_state.get("thread")
                    if t:
                        t.join(timeout=15)
                    start_btn.disabled = False
                    with btn_output:
                        print(" Agent stopped. Click 'Start Agent' to restart.")
                threading.Thread(target=_wait_and_enable, daemon=True).start()
            except Exception as e:
                with btn_output:
                    btn_output.clear_output()
                    print(f"ERROR stopping agent: {e}")
                    import traceback
                    traceback.print_exc()

        start_btn.on_click(_on_start_click)
        stop_btn.on_click(_on_stop_click)
        display(HBox([start_btn, stop_btn]), btn_output)

        # Auto-start the agent immediately
        _on_start_click(start_btn)

    except ImportError:
        # No ipywidgets - run directly (for terminal / non-Jupyter use)
        ev = threading.Event()
        t = threading.Thread(target=_polling_loop, args=(ev,), daemon=True)
        t.start()
        logger.info("Agent started. Press Ctrl+C to stop.")
        try:
            while t.is_alive():
                t.join(timeout=1)
        except KeyboardInterrupt:
            ev.set()
            t.join(timeout=5)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("Stopped by User")


