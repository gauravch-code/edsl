import pytest

from edsl.questions import QuestionNumerical
from edsl.questions.exceptions import QuestionAnswerValidationError


@pytest.fixture
def probability_question():
    return QuestionNumerical(
        question_name="probability",
        question_text="Return one probability.",
        min_value=0,
        max_value=1,
    )


@pytest.mark.parametrize("answer", [0, 1, 0.1667, "0.1667", "1e-3"])
def test_fix_preserves_valid_structured_answer(probability_question, answer):
    response = {
        "answer": answer,
        "comment": "Observed 12 events in 2026.",
        "generated_tokens": "Final answer: 0.75",
    }
    assert probability_question.response_validator.fix(response) == response


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I counted 1 observation on day 12. The final probability is 0.1667.", 0.1667),
        ("Year 2026, day 12, $500, 25%. Final answer: 0.1667", 0.1667),
        ("I counted 1 observation on day 12.\nFinal answer: 0.1667", 0.1667),
        ("I counted 1 observation on day 12.\n0.1667\n", 0.1667),
        ("Initial estimate: 0.25. FINAL ANSWER = 0.5", 0.5),
        ("I counted 1 event. Final answer: 0", 0),
        ("I counted 0 events. Final answer: 1", 1),
        ("Initial estimate: 0.25. Final answer is +2e-3", 0.002),
        ("The probability is .25", 0.25),
        ("The probability is 2E-3", 0.002),
        ("The probability is 0.25.", 0.25),
    ],
)
def test_repair_selects_unambiguous_answer(probability_question, text, expected):
    response = {
        "answer": text,
        "comment": "Explanation with 99 unrelated observations.",
        "generated_tokens": "Original model output with 100 observations.",
    }
    original = response.copy()
    validator = probability_question.response_validator
    result = validator.validate(response)
    assert result["answer"] == pytest.approx(expected)
    assert result["comment"] == response["comment"]
    assert result["generated_tokens"] == response["generated_tokens"]
    assert response == original
    assert validator.original_exception.data == original


@pytest.mark.parametrize(
    "text",
    [
        "Either 0.25 or 0.75",
        "I counted 1 observation on day 12. Probability could be 0.1667.",
        "Final answer: 0.25 or 0.75",
        "Final answer: 0.25\nFinal answer: 0.75",
        "Initial estimate: 0.5. Final answer: 2",
        "Initial estimate: 0.5. Final answer: unknown",
        "The probability is -0.25",
        "The probability is 1e3",
        "The probability is 1,2",
        "The probability is 25%",
        "The probability is 0.5e",
    ],
)
def test_ambiguous_or_invalid_repair_stays_invalid(probability_question, text):
    response = {"answer": text, "generated_tokens": "Final answer: 0.5"}
    assert probability_question.response_validator.fix(response) == response
    with pytest.raises(QuestionAnswerValidationError):
        probability_question.response_validator.validate(response)


@pytest.mark.parametrize("answer", ["unknown", "", [], {}])
def test_repair_does_not_use_metadata_as_answer(probability_question, answer):
    response = {
        "answer": answer,
        "comment": "0.5",
        "generated_tokens": "Final answer: 0.75",
    }
    with pytest.raises(QuestionAnswerValidationError):
        probability_question.response_validator.validate(response)


@pytest.mark.parametrize("answer_field", [{}, {"answer": None}])
def test_missing_answer_can_use_generated_tokens(probability_question, answer_field):
    response = {
        **answer_field,
        "comment": "12 events",
        "generated_tokens": "Observed 1 event on day 12.\nFinal answer: 0.1667",
    }
    result = probability_question.response_validator.validate(response)
    assert result["answer"] == pytest.approx(0.1667)
    assert result["comment"] == response["comment"]
    assert result["generated_tokens"] == response["generated_tokens"]


@pytest.mark.parametrize("generated_tokens", ["0.25 or 0.75", {"answer": 0.5}, None])
def test_missing_answer_fallback_rejects_ambiguous_or_nontext_output(
    probability_question, generated_tokens
):
    with pytest.raises(QuestionAnswerValidationError):
        probability_question.response_validator.validate(
            {"generated_tokens": generated_tokens}
        )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The answer is -42", -42),
        ("The answer is +42", 42),
        ("The answer is -.25", -0.25),
        ("The answer is -2.5e+3", -2500),
        ("The answer is 1,234.5", 1234.5),
        ("The answer is -1,234.5e-2", -12.345),
    ],
)
def test_repair_preserves_numeric_notation(text, expected):
    question = QuestionNumerical(
        question_name="value", question_text="Return a number."
    )
    result = question.response_validator.validate({"answer": text})
    assert result["answer"] == pytest.approx(expected)


def test_out_of_range_structured_answer_is_not_replaced(probability_question):
    response = {"answer": -0.5, "generated_tokens": "Final answer: 0.5"}
    assert probability_question.response_validator.fix(response) == response
    with pytest.raises(QuestionAnswerValidationError):
        probability_question.response_validator.validate(response)


@pytest.mark.parametrize(
    "text",
    [
        "The answer is 1,2",
        "The answer is 12,34,567",
        "The answer is 1.2.3",
        "The answer is 1..",
        "The answer is 1e-",
        "The answer is 25%",
        "The answer is 0.5 %",
        "The answer is 0.5\uff05",
        "The answer is 0.5 \uff05",
        "The answer is 1/2",
        "The answer is 1-2",
        "The answer is - 0.5",
        "The answer is \u22120.5",
        "The answer is \uff0d0.5",
        "The answer is \ufe630.5",
        "The answer is \uff0b0.5",
        "The answer is \u20130.5",
        "The answer is \u2013 0.5",
        "The answer is 0.5e or 0.75",
        "There were \u0661 events; the probability might be 0.5",
    ],
)
def test_repair_does_not_accept_a_partial_numeric_literal(text):
    question = QuestionNumerical(
        question_name="value", question_text="Return a number."
    )
    with pytest.raises(QuestionAnswerValidationError):
        question.response_validator.validate({"answer": text})


def test_permissive_repair_still_preserves_constraints_policy(probability_question):
    probability_question.permissive = True
    result = probability_question.response_validator.validate(
        {"answer": "Final answer: 2"}
    )
    assert result["answer"] == 2


def test_local_job_records_final_answer_instead_of_intermediate_count(
    probability_question,
):
    from edsl import Model

    response = (
        "I counted 1 observation on day 12. The final probability is 0.1667.\n"
        "Explanation with 12 unrelated observations."
    )
    results = probability_question.by(Model("test", canned_response=response)).run(
        disable_remote_inference=True,
        disable_remote_cache=True,
        cache=False,
        stop_on_exception=True,
    )
    assert results.select("answer.probability").to_list() == [0.1667]
    assert results.select("comment.probability_comment").to_list() == [
        "Explanation with 12 unrelated observations."
    ]
    assert results.select(
        "generated_tokens.probability_generated_tokens"
    ).to_list() == [response]
