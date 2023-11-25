from pathlib import Path

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from streamlit.logger import get_logger

import utilities


class Searcher:
    SEARCH_PROMPT_TEMPLATE = PromptTemplate(input_variables=["context", "question"], template="""
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """)

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        self.documents = []
        file_list = list(Path(config['data_folder']).glob("*.json"))
        for file in file_list:
            self.documents.append(Path(file).read_text())
        self.logger.info(f"Loaded documents [{self.documents}]")

    def search_documents(self, query) -> str:
        llm = utilities.get_llm(self.config)

        self.logger.info(f"Searching: [{query}]")
        llm_chain = LLMChain(prompt=self.SEARCH_PROMPT_TEMPLATE, llm=llm)
        result = llm_chain.run(context="\n\n".join(self.documents), question=query)
        self.logger.info(f"Search result: [{result}]")

        return result
