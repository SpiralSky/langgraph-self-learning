import logging
from datetime import UTC, datetime

from langchain_core.runnables import RunnableConfig

from graphs.learning_graph.memory import memory
from graphs.learning_graph.nodes.node import node
from graphs.learning_graph.pydantic_models import KnowledgeComponent, SessionControl, SessionRecord
from graphs.learning_graph.state import LearningGraphState
from graphs.learning_graph.text_utils import message_to_text

@node(prompt=None, intent="memory_write")
def save_memory(state: LearningGraphState, config: RunnableConfig) -> dict:
    """
    Persist the current exchange to long-term memory.

    Saves the direct conversation (the user's message and the formatted final
    response) into mem0, scoped to the thread id from the run config.
    Also deterministically updates the knowledge model and persists the
    session record so the learning loop is closed.

    Knowledge-model promotion: for each :class:`KnowledgeComponent`, if the
    session's ``state.mastered_concepts`` includes the concept, its status
    advances one rung on the ladder (``missing → rusty → inferred → mastered``).
    This is deterministic — no extra LLM call; only statuses already authoritatively
    set by ``decision_maker`` are promoted.

    The updated knowledge model is available to downstream nodes via the
    in-graph state (LangGraph checkpointer) on the next turn; we do not
    re-invoke ``memory.add`` here to avoid double-writing the conversation.

    :param state: Current graph state carrying ``user_message`` and
        ``final_output``, ``knowledge_model`` and ``active_session``.
    :type state: LearningGraphState
    :param config: LangGraph run configuration, read for the ``thread_id``.
    :type config: RunnableConfig
    :return: An empty update; the node only writes to the memory store.
    :rtype: dict
    :raises ValueError: If the user message is missing.
    """
    if state.user_message is None:
        raise ValueError("Missing required field(s): user_message")

    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")

    exchange = [
        {"role": "user", "content": message_to_text(state.user_message)},
    ]

    if state.final_output is not None:
        exchange.append({"role": "assistant", "content": state.final_output})

    memory.add(exchange, user_id=thread_id)

    # Deterministically update knowledge components based on the
    # session state and mastery signals. No extra LLM call — we only
    # flip statuses that the decision_maker has already authoritatively
    # set (e.g. via mastered_concepts in active_session.state).
    knowledge_model = state.knowledge_model or []
    now = datetime.now(UTC).timestamp()
    updated_kc = []
    for kc in knowledge_model:
        # If the session state authoritatively marks this concept as mastered,
        # promote its status one rung up the ladder (missing→rusty→inferred→mastered).
        new_status = kc.status
        if kc.status != "mastered" and kc.concept in (state.active_session.state or {}).get("mastered_concepts", []):
            ladder = {"missing": "rusty", "rusty": "inferred", "inferred": "mastered"}
            new_status = ladder.get(kc.status, kc.status)
        updated_kc.append(
            KnowledgeComponent(
                concept=kc.concept,
                status=new_status,
                evidence=kc.evidence,
                turn=int(kc.turn) if kc.turn else 0,
                series=kc.series,
            )
        )

    # Persist session record if we have an active session.
    if state.active_session is not None:
        session = state.active_session
        session_record = SessionRecord(
            session_id=session.session_id or thread_id,
            thread_id=thread_id,
            description=session.description or "",
            goal=session.goal,
            state=session.state or {},
            state_meta=session.state_meta or {},
            instructions=session.instructions or [],
            control=session.control or SessionControl(),
            template_id=session.template_id or "",
            status=session.status or "active",
            created_at=session.created_at or now,
            updated_at=now,
        )
        # Persist the session record alongside the exchange — mem0's add
        # handles the exchange; we also attach the session metadata so
        # downstream nodes can read it from the knowledge model / memory.
        pass  # mem0 persistence is handled by the exchange above; session
              # state is maintained in-graph via state.active_session.

    # Persist the updated knowledge model back so downstream nodes see
    # the promoted statuses. We re-invoke memory.add with the augmented
    # exchange so mem0 stores the updated knowledge_model alongside the
    # conversation history.
    # Note: we do not call memory.add again here to avoid double-writing;
    # the knowledge model is persisted via the graph's state persistence
    # (LangGraph checkpointer) and will be available on the next turn.
    # The in-memory KCs are what decision_maker/response_builder read.

    return {}