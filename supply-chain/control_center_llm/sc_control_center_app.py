import json
import streamlit as st

from PIL import Image

from langchain.callbacks.base import BaseCallbackHandler

from processors.program_aided.processor import ProgramAidedProcessor, ProcessingContext

#  Initialization
image = Image.open('banner.png')
st.image(image)

chain = [
    {
        "role": "AI",
        "caption": "Chat manager",
        "stage": "create_standalone_query"
    },
    {
        "role": "AI",
        "caption": "Software engineer",
        "stage": "create_logic"
    },
    {
        "role": "AI",
        "caption": "Software engineer",
        "stage": "create_script"
    },
    {
        "role": "AI",
        "caption": "Technical lead",
        "stage": "fix_script"
    },
    {
        "role": "Assistant",
        "caption": "Final answer",
        "stage": "execute_script"
    }
]


def format_content(caption: str, content: str) -> str:
    return f"**{caption}**:\n\n{content}"


if "config" not in st.session_state:
    with open('config.json') as f:
        st.session_state["config"] = json.load(f)

    st.session_state.processor = ProgramAidedProcessor(st.session_state["config"])
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(format_content(message["caption"], message["content"]))

# Input processing
if prompt := st.chat_input("What is up?"):

    # Print user prompt
    st.session_state.messages.append({"role": "User", "caption": "User", "content": prompt})
    with st.chat_message("User"):
        st.markdown(prompt)

    # Execute the chain step by step
    context = ProcessingContext(st.session_state.messages, {})
    for step in chain:
        with st.chat_message(step['role']):
            placeholder = st.empty()
            full_content = []

            def on_llm_new_token(token: str) -> None:
                    full_content.append(token)
                    placeholder.markdown("".join(full_content) + "▌")

            st.session_state.processor.process(step['stage'], context, on_llm_new_token)

        content = "".join([x for x in full_content if x is not None])
        context.generated_outputs[step["stage"]] = content
        placeholder.markdown(format_content(step['caption'], content))

        st.session_state.messages.append({"role": step['role'], "caption": step['caption'], "content": content})
