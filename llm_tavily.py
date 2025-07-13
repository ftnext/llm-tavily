import llm
from tavily import TavilyClient


class TavilyQnASearch(llm.KeyModel):
    model_id = "qa-tavily"
    needs_key = "tavily"
    key_env_var = "LLM_TAVILY_KEY"

    def execute(self, prompt, stream, response, conversation, key):
        tavily_client = TavilyClient(api_key=key)
        answer = tavily_client.qna_search(prompt.prompt)
        return answer


@llm.hookimpl
def register_models(register):
    register(TavilyQnASearch())
