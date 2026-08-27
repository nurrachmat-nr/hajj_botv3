import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amd.agents.dialogic_answering import DialogicAnsweringAgent
from amd.agents.reflective_feedback import ReflectiveFeedbackAgent
from amd.agents.socratic_questioning import SOCRATIC_DIMENSIONS, SocraticQuestioningAgent
from amd.llm import MockLLM

QUERY = "What causes inflation?"


def test_socratic_questioning_produces_three_dimensions():
    agent = SocraticQuestioningAgent(MockLLM())
    sub_questions = agent.run(QUERY)

    assert len(sub_questions) == 3
    assert {sq.dimension for sq in sub_questions} == set(SOCRATIC_DIMENSIONS)
    assert all(sq.text for sq in sub_questions)


def test_dialogic_answering_produces_one_answer_per_sub_question():
    socratic = SocraticQuestioningAgent(MockLLM())
    answering = DialogicAnsweringAgent(MockLLM())

    sub_questions = socratic.run(QUERY)
    answers = answering.run_all(QUERY, sub_questions)

    assert len(answers) == len(sub_questions)
    for answer, sub_question in zip(answers, sub_questions):
        assert answer.dimension == sub_question.dimension
        assert answer.text


def test_reflective_feedback_filters_by_threshold():
    llm = MockLLM()
    socratic = SocraticQuestioningAgent(llm)
    answering = DialogicAnsweringAgent(llm)

    sub_questions = socratic.run(QUERY)
    answers = answering.run_all(QUERY, sub_questions)

    lenient = ReflectiveFeedbackAgent(llm, relevance_threshold=0.0)
    strict = ReflectiveFeedbackAgent(llm, relevance_threshold=1.01)

    assert len(lenient.run_all(QUERY, answers)) == len(answers)
    assert len(strict.run_all(QUERY, answers)) == 0

    for refined in lenient.run_all(QUERY, answers):
        assert 0.0 <= refined.score <= 1.0
        assert refined.text
