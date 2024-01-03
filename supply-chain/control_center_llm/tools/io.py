import pandas as pd
from streamlit.logger import get_logger
import streamlit as st


def print_answer(text: str, on_new_token):
    get_logger(__name__).info(f"Output: [{text}]")
    on_new_token("\n\n")
    on_new_token(text)


def print_table(df: pd.DataFrame, on_new_token):
    on_new_token("\n\n")
    on_new_token(df.to_markdown())


def show_line_chart(data: pd.DataFrame):
    st.line_chart(data)
