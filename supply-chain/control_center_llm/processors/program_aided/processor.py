import re
from dataclasses import dataclass
from functools import partial

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain

from streamlit.logger import get_logger

import utilities
from processors.program_aided.prompt_chat import CHAT_TO_QUERY_PROMPT
from processors.program_aided.prompt_critic import *
from processors.program_aided.prompt_planner import *
from processors.program_aided.prompt_coder import *
from tools.database import Database
from tools.forecast import get_forecast
from tools.io import print_answer, show_line_chart, print_table
from tools.search import Searcher
from tools.shipping import get_shipping_cost
from tools.solver import stock_demand_difference


@dataclass
class ProcessingContext:
    full_chat_history: list[dict[str, str]]
    generated_outputs: dict[str, str]


def _extract_code(raw_plan: str) -> str:
    m = re.search("```python(.*?)```", raw_plan, re.DOTALL)
    if m:
        return m.group(1)
    else:
        return ""


class ProgramAidedProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        self.tool_searcher = Searcher(config)
        self.tool_database = Database(config)

        with open("tools/api.py", "r") as file:
            self.api_docs = file.read()

    def process(self, stage: str, context: ProcessingContext, on_new_token):
        match stage:
            case "create_standalone_query":
                self.chat_history_to_query(context, on_new_token)
            case "create_logic":
                self.create_logic(context, on_new_token)
            case "create_script":
                self.create_script(context, on_new_token)
            case "fix_script":
                self.fix_script(context, on_new_token)
            case "execute_script":
                self.execute_script(context, on_new_token)
            case _:
                raise Exception(f"Unknown stage [{stage}]")

    def chat_history_to_query(self, context: ProcessingContext, on_new_token):
        llm = utilities.get_llm(self.config, streaming=True)

        chat_history_str = ""
        for entry in context.full_chat_history:
            if entry['role'] in ('Assistant', 'User'):
                chat_history_str += f"{entry['role']}:\n{entry['content']}\n\n"

        prompt = CHAT_TO_QUERY_PROMPT.format_prompt(chat_history=chat_history_str)
        for chunk in llm.stream(prompt):
            on_new_token(chunk.content)

    def create_logic(self, context: ProcessingContext, on_new_token):
        llm = utilities.get_llm(self.config, streaming=True, stop=PLANNER_PROMPT_STOP_SEQ)

        prompt = PLANNER_PROMPT.format_prompt(api_docs=self.api_docs,
                                              query=context.generated_outputs["create_standalone_query"])
        for chunk in llm.stream(prompt):
            on_new_token(chunk.content)

    def create_script(self, context: ProcessingContext, on_new_token):
        llm = utilities.get_llm(self.config, streaming=True, stop=CODER_PROMPT_STOP_SEQ)

        prompt = CODER_PROMPT.format_prompt(api_docs=self.api_docs,
                                            logic_description=context.generated_outputs["create_logic"],
                                            query=context.generated_outputs["create_standalone_query"])
        for chunk in llm.stream(prompt):
            on_new_token(chunk.content)

    def fix_script(self, context: ProcessingContext, on_new_token):
        script = context.generated_outputs["create_script"]
        self.logger.info("Initial script:\n" + script)

        llm = utilities.get_llm(self.config, streaming=True, stop=CRITIC_PROMPT_STOP_SEQ)

        prompt = CRITIC_PROMPT.format_prompt(api_docs=self.api_docs,
                                             query=context.generated_outputs["create_standalone_query"],
                                             python_code=context.generated_outputs["create_script"])

        for chunk in llm.stream(prompt):
            on_new_token(chunk.content)

    def execute_script(self, context: ProcessingContext, on_new_token: BaseCallbackHandler):
        script = context.generated_outputs["fix_script"]
        self.logger.info("Final script:\n" + script)
        code = _extract_code(script)

        self.logger.info("Executing code:\n" + code)
        utilities.run_code(code, globals={
            "print_answer": partial(print_answer, on_new_token=on_new_token),
            "print_table": partial(print_table, on_new_token=on_new_token),
            "show_line_chart": show_line_chart,
            "search_documents": self.tool_searcher.search_documents,
            "query_inventory": self.tool_database.query_inventory,
            "query_suppliers": self.tool_database.query_suppliers,
            "get_forecast": get_forecast,
            "stock_demand_difference": partial(stock_demand_difference, db=self.tool_database),
            "get_shipping_cost": get_shipping_cost
        })
