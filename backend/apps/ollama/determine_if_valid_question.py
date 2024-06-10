import os
from openai import OpenAI
import json


class OpenAIClient:
    """Client for the OPENAI."""

    def __init__(self, **kwargs):
        # self._load_env()
        self.client = self._return_client()

    # def _load_env(self) -> None:
    # load_dotenv(find_dotenv(raise_error_if_not_found=True))

    def _return_client(self) -> OpenAI:
        return OpenAI(
            api_key=os.environ["OPENAI_RELIGIOUS_FREEDOM_API_KEY"],
            organization=os.environ["OPENAI_ORG_ID"],
            project=os.environ["OPENAI_RELIGIOUS_FREEDOM_ID"],
        )

    @property
    def model_name(self) -> str:
        return "gpt-3.5-turbo"

    @property
    def system_message(self) -> str:
        return """
        You are going to evaluate if the text is a legitimate question, a follow-up question, or spam.
        If a legitimate question, return 'is valid'.
        If a follow-up queston, return 'is follow up'
        If not a valid question, return text 'not valid'.

        Definitions:
        several words: 'what questions can you answer?'
        random letters: lkjlkjlklj or similar.
        follow-up: A question meant to clarify or elaborate on a previous answer.

        Conditions for a valid question:
        1. Does the phrase contain several words?
        2. Does the phrase not contain random letters?
        3. Would a human understand the phrase?

        If the answer to those three questions is yes, then the question is valid.

        Conditions for a follow-up question:
        1. Does the phrase ask for clarification or explanation?
        2. Does the phrase appear to be referring to a previous question?
        3. Does the phrase seem informal and conversational?

        If the answer to the above three questions is yes, then the question is a follow-up.

        """


class DetermineIfValidQuestion(OpenAIClient):
    """Determine if the user input is a valid question.

    Example
    -------
    >>> valid_question = DetermineIfValidQuestion()
    >>> valid_question.test_evaluate_user_question()

    >>> valid_question.evaluate_user_question("Are lkdjfdkfjladjf;ljsa;ldjfks")
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_user_question(self, user_msg: str) -> bool:
        """Evaluate if the user question is valid."""
        return self._call_model_process_response(user_msg)

    def test_evaluate_user_question(self, user_msg: str = "") -> None:
        """Test the evaluate user question."""
        # Define the user message
        if not user_msg:
            user_msg = "Are lkdjfdkfjladjf;ljsa;ldjfks"

        assert not self._call_model_process_response(user_msg)

    def _call_model_process_response(self, user_msg: str) -> bool:
        """Call the model and validate the repsonse."""
        chat_completions_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_msg},
            ],
        )

        dumped_json = self._convert_response_to_json(chat_completions_response)
        self.content = self._find_response_content(dumped_json)

    def _convert_response_to_json(self, response) -> dict:
        """Convert the response to json."""
        return json.loads(response.model_dump_json())

    def _find_response_content(self, dumped_json: dict) -> str:
        """Find the response content."""
        return dumped_json["choices"][0]["message"]["content"]

    def is_follow_up(self) -> bool:
        return self.content == "is follow up"

    def is_valid_question(self) -> bool:
        return self.content == "is valid"

    def is_not_valid(self) -> bool:
        return self.content == "not valid"
