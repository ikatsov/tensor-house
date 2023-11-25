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

    def process(self, stage: str, context: ProcessingContext, output_handler: BaseCallbackHandler):
        match stage:
            case "create_standalone_query":
                self.chat_history_to_query(context, output_handler)
            case "create_logic":
                self.create_logic(context, output_handler)
            case "create_script":
                self.create_script(context, output_handler)
            case "fix_script":
                self.fix_script(context, output_handler)
            case "execute_script":
                self.execute_script(context, output_handler)
            case _:
                raise Exception(f"Unknown stage [{stage}]")

    def chat_history_to_query(self, context: ProcessingContext, output_handler: BaseCallbackHandler):
        llm = utilities.get_llm(self.config, output_handler=output_handler)

        chat_history_str = ""
        for entry in context.full_chat_history:
            if entry['role'] in ('Assistant', 'User'):
                chat_history_str += f"{entry['role']}:\n{entry['content']}\n\n"

        llm_chain = LLMChain(prompt=CHAT_TO_QUERY_PROMPT, llm=llm)
        llm_chain.run(chat_history=chat_history_str)

    def create_logic(self, context: ProcessingContext, output_handler: BaseCallbackHandler):
        llm = utilities.get_llm(self.config, output_handler=output_handler, stop=PLANNER_PROMPT_STOP_SEQ)

        llm_chain = LLMChain(prompt=PLANNER_PROMPT, llm=llm)
        llm_chain.run(api_docs=self.api_docs,
                      query=context.generated_outputs["create_standalone_query"])

    def create_script(self, context: ProcessingContext, output_handler: BaseCallbackHandler):
        llm = utilities.get_llm(self.config, output_handler=output_handler, stop=CODER_PROMPT_STOP_SEQ)

        llm_chain = LLMChain(prompt=CODER_PROMPT, llm=llm)
        llm_chain.run(api_docs=self.api_docs,
                      logic_description=context.generated_outputs["create_logic"],
                      query=context.generated_outputs["create_standalone_query"])

    def fix_script(self, context: ProcessingContext, output_handler: BaseCallbackHandler):
        script = context.generated_outputs["create_script"]
        self.logger.info("Initial script:\n" + script)
        code = _extract_code(script)

        llm = utilities.get_llm(self.config, output_handler=output_handler, stop=CRITIC_PROMPT_STOP_SEQ)

        llm_chain = LLMChain(prompt=CRITIC_PROMPT, llm=llm)
        llm_chain.run(api_docs=self.api_docs,
                      query=context.generated_outputs["create_standalone_query"],
                      python_code=context.generated_outputs["create_script"])

    def execute_script(self, context: ProcessingContext, output_handler: BaseCallbackHandler):
        script = context.generated_outputs["fix_script"]
        self.logger.info("Generated plan:\n" + script)
        code = _extract_code(script)

        self.logger.info("Executing code:\n" + code)
        utilities.run_code(code, globals={
            "print_answer": partial(print_answer, output_handler=output_handler),
            "print_table": partial(print_table, output_handler=output_handler),
            "show_line_chart": show_line_chart,
            "search_documents": self.tool_searcher.search_documents,
            "query_inventory": self.tool_database.query_inventory,
            "query_suppliers": self.tool_database.query_suppliers,
            "get_forecast": get_forecast,
            "stock_demand_difference": partial(stock_demand_difference, db=self.tool_database),
            "get_shipping_cost": get_shipping_cost
        })
