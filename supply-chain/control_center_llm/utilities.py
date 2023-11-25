import sys
from io import StringIO

from langchain.chat_models import AzureChatOpenAI


def run_code(code, globals):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        exec(code, globals)
        sys.stdout = old_stdout
        output = mystdout.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        output = repr(e)

    return output


def get_llm(config, temperature=0.7, output_handler=None, stop=""):
    if output_handler is not None:
        llm = AzureChatOpenAI(
            openai_api_key=config["api_key"],
            openai_api_base=config["api_base"],
            openai_api_version=config["api_version"],
            openai_api_type=config["api_type"],
            deployment_name=config["text_deployment"],
            model=config["text_model"],
            temperature=temperature,
            streaming=True,
            callbacks=[output_handler],
            stop=stop
        )
    else:
        llm = AzureChatOpenAI(
            openai_api_key=config["api_key"],
            openai_api_base=config["api_base"],
            openai_api_version=config["api_version"],
            openai_api_type=config["api_type"],
            deployment_name=config["text_deployment"],
            model=config["text_model"],
            temperature=temperature,
            stop=stop
        )

    return llm
