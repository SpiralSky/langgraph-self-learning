from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.pregel.protocol import RunnableConfig
import pytest

from graphs.learning_graph.learning_graph import graph
from graphs.learning_graph.pydantic_models import InputAnalysisResult, TeachingStrategy, SessionContext, ResponseBuilderOutput, SessionControl, PrerequisiteGap
from graphs.learning_graph.state import LearningGraphState
from graphs.learning_graph.state import LearningGraphState


@pytest.fixture
def mock_mem0():
    with patch("graphs.learning_graph.nodes.retrieve_memory.memory") as m1, \
         patch("graphs.learning_graph.nodes.save_memory.memory") as m2:
        m1.search.return_value = {"results": []}
        m1.add.return_value = {"message": "ok"}
        m2.add.return_value = {"message": "ok"}
        yield m1, m2


@pytest.fixture
def mock_llm():
    with patch("graphs.learning_graph.nodes.response_builder.get_chat_model") as p1, \
         patch("graphs.learning_graph.nodes.decision_maker.get_chat_model") as p2, \
         patch("graphs.learning_graph.nodes.input_analyzer.get_chat_model") as p3, \
         patch("graphs.learning_graph.nodes.format_output.get_chat_model") as p4:
        yield p1, p2, p3, p4


def _chain(return_value):
    chain = MagicMock()
    chain.with_structured_output.return_value = chain
    chain.invoke.return_value = return_value
    return chain


def _content_chain(content_text: str):
    msg = MagicMock()
    msg.content = content_text
    chain = MagicMock()
    chain.with_structured_output.return_value = chain
    chain.invoke.return_value = msg
    return chain


@pytest.mark.usefixtures("mock_llm", "mock_mem0")
class TestGraphInvocation:
    def test_graph_invocation_returns_state(self, mock_llm):
        p1, p2, p3, p4 = mock_llm
        strategy = TeachingStrategy(
            pedagogical_move="explain_concept",
            focus_concept="conditional probability",
            tone="socratic",
            execution_plan=["Define conditional probability", "Give an example"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="conditional probability",
                status="active",
                control=SessionControl(awaiting_user_input=False),
            ),
        )
        response = ResponseBuilderOutput(
            draft_response="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)",
            tone="socratic",
            sources_used=[],
        )
        p2.return_value = _chain(strategy)
        p1.return_value = _chain(response)
        p3.return_value = _chain(
            InputAnalysisResult(
                is_clear=True,
                intent="conceptual",
                key_points=["conditional probability"],
                prerequisite_gaps=[],
                comments="Clear conceptual question.",
            )
        )
        p4.return_value = _content_chain(response.draft_response)

        result = graph.invoke({
            "messages": [HumanMessage("hello")],
            "user_message": HumanMessage("hello"),
        })

        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) >= 2
        assert isinstance(result["messages"][-1], AIMessage)

    def test_graph_with_thread_id(self, mock_llm):
        p1, p2, p3, p4 = mock_llm
        strategy = TeachingStrategy(
            pedagogical_move="explain_concept",
            focus_concept="conditional probability",
            tone="socratic",
            execution_plan=["Define conditional probability", "Give an example"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="conditional probability",
                status="active",
                control=SessionControl(awaiting_user_input=False),
            ),
        )
        response = ResponseBuilderOutput(
            draft_response="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)",
            tone="socratic",
            sources_used=[],
        )
        p2.return_value = _chain(strategy)
        p1.return_value = _chain(response)
        p3.return_value = _chain(
            InputAnalysisResult(
                is_clear=True,
                intent="conceptual",
                key_points=["conditional probability"],
                prerequisite_gaps=[],
                comments="Clear conceptual question.",
            )
        )
        p4.return_value = _content_chain(response.draft_response)

        config: RunnableConfig = RunnableConfig(configurable={"thread_id": "test-thread"})
        result = graph.invoke({
            "messages": [HumanMessage("hello")],
            "user_message": HumanMessage("hello"),
        }, config=config)

        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) >= 2

    def test_multi_turn_state_persists(self, mock_llm):
        p1, p2, p3, p4 = mock_llm
        strategy = TeachingStrategy(
            pedagogical_move="explain_concept",
            focus_concept="conditional probability",
            tone="socratic",
            execution_plan=["Define conditional probability", "Give an example"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="conditional probability",
                status="active",
                control=SessionControl(awaiting_user_input=False),
            ),
        )
        response = ResponseBuilderOutput(
            draft_response="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)",
            tone="socratic",
            sources_used=[],
        )
        p2.return_value = _chain(strategy)
        p1.return_value = _chain(response)
        p3.return_value = _chain(
            InputAnalysisResult(
                is_clear=True,
                intent="conceptual",
                key_points=["conditional probability"],
                prerequisite_gaps=[],
                comments="Clear conceptual question.",
            )
        )
        p4.return_value = _content_chain(response.draft_response)

        config: RunnableConfig = RunnableConfig(configurable={"thread_id": "multi-turn-1"})

        result1 = graph.invoke({
            "messages": [HumanMessage("I want to learn probability")],
            "user_message": HumanMessage("I want to learn probability"),
        }, config=config)

        assert len(result1["messages"]) >= 2
        assert result1["teaching_strategy"] is not None
        assert result1["teaching_strategy"].pedagogical_move == "explain_concept"

        strategy2 = TeachingStrategy(
            pedagogical_move="ask_guiding_question",
            focus_concept="conditional probability",
            tone="socratic",
            execution_plan=["Ask about P(A∩B)"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="conditional probability",
                status="active",
                control=SessionControl(awaiting_user_input=True, pending_item_id="task_1"),
            ),
        )
        response2 = ResponseBuilderOutput(
            draft_response="## Follow-up\n\nCan you tell me what P(A∩B) means?",
            tone="socratic",
            sources_used=[],
        )
        p2.return_value = _chain(strategy2)
        p1.return_value = _chain(response2)

        result2 = graph.invoke({
            "messages": [HumanMessage("What is conditional probability?"), AIMessage("# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)"), HumanMessage("Can you explain more?")],
            "user_message": HumanMessage("Can you explain more?"),
        }, config=config)

        assert len(result2["messages"]) >= 3
        assert result2["teaching_strategy"].pedagogical_move == "ask_guiding_question"

    def test_multi_turn_session_persists(self, mock_llm, mock_mem0):
        p1, p2, p3, p4 = mock_llm
        m1, m2 = mock_mem0
        strategy = TeachingStrategy(
            pedagogical_move="propose_session",
            focus_concept="probability",
            tone="encouraging",
            execution_plan=["Propose a learning session on probability"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="probability",
                status="active",
                control=SessionControl(awaiting_user_input=False),
            ),
        )
        response = ResponseBuilderOutput(
            draft_response="## Session Proposal\n\nLet's start a session on probability.",
            tone="encouraging",
            sources_used=[],
        )
        p2.return_value = _chain(strategy)
        p1.return_value = _chain(response)
        p3.return_value = _chain(
            InputAnalysisResult(
                is_clear=True,
                intent="conceptual",
                key_points=["probability"],
                prerequisite_gaps=[],
                comments="Clear conceptual question.",
            )
        )
        p4.return_value = _content_chain(response.draft_response)

        config: RunnableConfig = RunnableConfig(configurable={"thread_id": "session-thread"})

        result1 = graph.invoke({
            "messages": [HumanMessage("I want to learn probability")],
            "user_message": HumanMessage("I want to learn probability")},
            config=config,
        )

        assert isinstance(result1, dict)
        assert "messages" in result1

        strategy2 = TeachingStrategy(
            pedagogical_move="explain_concept",
            focus_concept="conditional probability",
            tone="socratic",
            execution_plan=["Define conditional probability"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="conditional probability",
                status="active",
                control=SessionControl(awaiting_user_input=False),
            ),
        )
        response2 = ResponseBuilderOutput(
            draft_response="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)",
            tone="socratic",
            sources_used=[],
        )
        p2.return_value = _chain(strategy2)
        p1.return_value = _chain(response2)

        result2 = graph.invoke({
            "messages": [HumanMessage("I want to learn probability"), AIMessage("## Session Proposal\n\nLet's start a session on probability."), HumanMessage("Can you teach me conditional probability?")],
            "user_message": HumanMessage("Can you teach me conditional probability?"),
        }, config=config)

        assert isinstance(result2, dict)
        assert len(result2["messages"]) >= 3

    def test_full_turn_populates_all_state_fields(self, mock_llm, mock_mem0):
        p1, p2, p3, p4 = mock_llm
        m1, m2 = mock_mem0
        strategy = TeachingStrategy(
            pedagogical_move="explain_concept",
            focus_concept="conditional probability",
            tone="socratic",
            execution_plan=["Define conditional probability", "Give an example"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="conditional probability",
                status="active",
                control=SessionControl(awaiting_user_input=False),
            ),
        )
        response = ResponseBuilderOutput(
            draft_response="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)",
            tone="socratic",
            sources_used=[],
        )
        p2.return_value = _chain(strategy)
        p1.return_value = _chain(response)
        p3.return_value = _chain(
            InputAnalysisResult(
                is_clear=True,
                intent="conceptual",
                key_points=["conditional probability"],
                prerequisite_gaps=[],
                comments="Clear conceptual question.",
            )
        )
        p4.return_value = _content_chain(response.draft_response)

        result = graph.invoke({
            "messages": [HumanMessage("What is conditional probability?")],
            "user_message": HumanMessage("What is conditional probability?"),
        })

        assert isinstance(result, dict)
        assert result["user_message"] is not None
        assert result["memory_results"] is not None
        assert result["analysis_results"] is not None
        assert result["teaching_strategy"] is not None
        assert result["draft_response"] is not None
        assert result["final_output"] is not None
        assert result["teaching_strategy"].pedagogical_move == "explain_concept"
        assert result["draft_response"].draft_response != ""
        assert result["final_output"] != ""

    def test_memory_search_called(self, mock_llm, mock_mem0):
        p1, p2, p3, p4 = mock_llm
        m1, m2 = mock_mem0
        strategy = TeachingStrategy(
            pedagogical_move="explain_concept",
            focus_concept="conditional probability",
            tone="socratic",
            execution_plan=["Define conditional probability"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="conditional probability",
                status="active",
                control=SessionControl(awaiting_user_input=False),
            ),
        )
        response = ResponseBuilderOutput(
            draft_response="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)",
            tone="socratic",
            sources_used=[],
        )
        p2.return_value = _chain(strategy)
        p1.return_value = _chain(response)
        p3.return_value = _chain(
            InputAnalysisResult(
                is_clear=True,
                intent="conceptual",
                key_points=["conditional probability"],
                prerequisite_gaps=[],
                comments="Clear conceptual question.",
            )
        )
        p4.return_value = _content_chain(response.draft_response)

        user_msg = "What is conditional probability?"
        graph.invoke({
            "messages": [HumanMessage(user_msg)],
            "user_message": HumanMessage(user_msg),
        })

        m1.search.assert_called_once()
        call_args = m1.search.call_args
        assert user_msg in call_args[0][0] or user_msg in str(call_args)

    def test_state_threads_between_nodes(self, mock_llm, mock_mem0):
        p1, p2, p3, p4 = mock_llm
        m1, m2 = mock_mem0
        strategy = TeachingStrategy(
            pedagogical_move="explain_concept",
            focus_concept="conditional probability",
            tone="socratic",
            execution_plan=["Define conditional probability"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="conditional probability",
                status="active",
                control=SessionControl(awaiting_user_input=False),
            ),
        )
        response = ResponseBuilderOutput(
            draft_response="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)",
            tone="socratic",
            sources_used=[],
        )
        p2.return_value = _chain(strategy)
        p1.return_value = _chain(response)
        p3.return_value = _chain(
            InputAnalysisResult(
                is_clear=True,
                intent="conceptual",
                key_points=["conditional probability"],
                prerequisite_gaps=[],
                comments="Clear conceptual question.",
            )
        )
        p4.return_value = _content_chain(response.draft_response)

        user_msg = "What is conditional probability?"
        result = graph.invoke({
            "messages": [HumanMessage(user_msg)],
            "user_message": HumanMessage(user_msg),
        })

        assert result["user_message"] is not None
        assert result["analysis_results"] is not None
        assert result["analysis_results"].is_clear is True
        assert result["analysis_results"].intent == "conceptual"

    def test_scenario_explanation(self, mock_llm, mock_mem0):
        p1, p2, p3, p4 = mock_llm
        m1, m2 = mock_mem0
        strategy = TeachingStrategy(
            pedagogical_move="explain_concept",
            focus_concept="conditional probability",
            tone="socratic",
            execution_plan=["Define conditional probability", "Give an example"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="conditional probability",
                status="active",
                control=SessionControl(awaiting_user_input=False),
            ),
        )
        response = ResponseBuilderOutput(
            draft_response="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)",
            tone="socratic",
            sources_used=[],
        )
        p2.return_value = _chain(strategy)
        p1.return_value = _chain(response)
        p3.return_value = _chain(
            InputAnalysisResult(
                is_clear=True,
                intent="conceptual",
                key_points=["conditional probability"],
                prerequisite_gaps=[],
                comments="Clear conceptual question.",
            )
        )
        p4.return_value = _content_chain(response.draft_response)

        result = graph.invoke({
            "messages": [HumanMessage("What is conditional probability?")],
            "user_message": HumanMessage("What is conditional probability?"),
        })

        assert result["teaching_strategy"].pedagogical_move == "explain_concept"

    def test_scenario_question(self, mock_llm, mock_mem0):
        p1, p2, p3, p4 = mock_llm
        m1, m2 = mock_mem0
        strategy = TeachingStrategy(
            pedagogical_move="ask_guiding_question",
            focus_concept="conditional probability",
            tone="socratic",
            execution_plan=["Ask about P(A∩B)"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="conditional probability",
                status="active",
                control=SessionControl(awaiting_user_input=True, pending_item_id="task_1"),
            ),
        )
        response = ResponseBuilderOutput(
            draft_response="## Follow-up\n\nCan you tell me what P(A∩B) means?",
            tone="socratic",
            sources_used=[],
        )
        p2.return_value = _chain(strategy)
        p1.return_value = _chain(response)
        p3.return_value = _chain(
            InputAnalysisResult(
                is_clear=False,
                intent="conceptual",
                key_points=["conditional probability"],
                prerequisite_gaps=[
                    PrerequisiteGap(
                        concept="joint probability",
                        status="rusty",
                        evidence="learner asked about P(A|B) without knowing P(A∩B)",
                    )
                ],
                comments="Needs guiding question.",
            )
        )
        p4.return_value = _content_chain(response.draft_response)

        result = graph.invoke({
            "messages": [HumanMessage("Can you explain P(A|B)?")],
            "user_message": HumanMessage("Can you explain P(A|B)?"),
        })

        assert result["teaching_strategy"].pedagogical_move == "ask_guiding_question"

    def test_scenario_propose_session(self, mock_llm, mock_mem0):
        p1, p2, p3, p4 = mock_llm
        m1, m2 = mock_mem0
        strategy = TeachingStrategy(
            pedagogical_move="propose_session",
            focus_concept="probability",
            tone="encouraging",
            execution_plan=["Propose a learning session on probability"],
            updated_session_context=SessionContext(
                goal="Learn probability",
                current_focus="probability",
                status="active",
                control=SessionControl(awaiting_user_input=False),
            ),
        )
        response = ResponseBuilderOutput(
            draft_response="## Session Proposal\n\nLet's start a session on probability.",
            tone="encouraging",
            sources_used=[],
        )
        p2.return_value = _chain(strategy)
        p1.return_value = _chain(response)
        p3.return_value = _chain(
            InputAnalysisResult(
                is_clear=True,
                intent="conceptual",
                key_points=["probability"],
                prerequisite_gaps=[],
                comments="Session request detected.",
            )
        )
        p4.return_value = _content_chain(response.draft_response)

        result = graph.invoke({
            "messages": [HumanMessage("Can you teach me step by step?")],
            "user_message": HumanMessage("Can you teach me step by step?"),
        })

        assert result["teaching_strategy"].pedagogical_move == "propose_session"

    def test_execute_code_branch_routing(self, mock_llm, mock_mem0):
        p1, p2, p3, p4 = mock_llm
        m1, m2 = mock_mem0
        with patch("graphs.learning_graph.nodes.execute_code._ssh_exec") as mock_ssh:
            mock_ssh.return_value = {
                "stdout": "42",
                "stderr": "",
                "returncode": 0,
            }
            strategy = TeachingStrategy(
                pedagogical_move="execute_code",
                focus_concept="python",
                tone="encouraging",
                execution_plan=["execute code: python3 -c \"print(42)\""],
                updated_session_context=SessionContext(
                    goal="Learn python",
                    current_focus="python",
                    status="active",
                    control=SessionControl(awaiting_user_input=False),
                ),
            )
            response = ResponseBuilderOutput(
                draft_response="## Execution result\n\nThe answer is 42.",
                tone="encouraging",
                sources_used=[],
            )
            p2.return_value = _chain(strategy)
            p1.return_value = _chain(response)
            p3.return_value = _chain(
                InputAnalysisResult(
                    is_clear=True,
                    intent="conceptual",
                    key_points=["python"],
                    prerequisite_gaps=[],
                    comments="Code execution requested.",
                )
            )
            p4.return_value = _content_chain(response.draft_response)

            result = graph.invoke({
                "messages": [HumanMessage("Run this code")],
                "user_message": HumanMessage("Run this code"),
            })

            assert "final_output" in result
            assert "42" in result["final_output"]

    def test_missing_user_message_raises(self, mock_llm, mock_mem0):
        p1, p2, p3, p4 = mock_llm
        with pytest.raises((ValueError, IndexError, KeyError)):
            graph.invoke({
                "messages": [HumanMessage("hello")],
            })

    def test_empty_messages_raises(self, mock_llm, mock_mem0):
        p1, p2, p3, p4 = mock_llm
        with pytest.raises((ValueError, IndexError, KeyError)):
            graph.invoke({
                "messages": [],
                "user_message": HumanMessage("hello"),
            })