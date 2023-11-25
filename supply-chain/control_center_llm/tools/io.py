import pandas as pd
from streamlit.logger import get_logger
import streamlit as st


def print_answer(text: str, output_handler):
    get_logger(__name__).info(f"Output: [{text}]")
    output_handler.on_llm_new_token("\n\n")
    output_handler.on_llm_new_token(text)


def print_table(df: pd.DataFrame, output_handler):
    output_handler.on_llm_new_token("\n\n")
    output_handler.on_llm_new_token(df.to_markdown())


def show_line_chart(data: pd.DataFrame):
    st.line_chart(data)
