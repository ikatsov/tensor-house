import sys
from io import StringIO

from langchain_google_genai import ChatGoogleGenerativeAI


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


def get_llm(config, temperature=0.7, streaming=True, stop=""):
    llm = ChatGoogleGenerativeAI(
        google_api_key=config["google_api_key"],
        model=config["text_model"],
        temperature=temperature,
        streaming=streaming,
        stop=stop
    )

    return llm
