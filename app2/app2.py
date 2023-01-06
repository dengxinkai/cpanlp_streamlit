import spacy_streamlit
import streamlit as st

st.title("My cool app")

doc = [{
    "text": "你真的很厉害呀哈哈哈哈",
    "ents": [{"start": 4, "end": 10, "label": "ORG"}],
    "title": None
}]

spacy_streamlit.visualize_ner(
    doc,
    labels=["ORG"],
    show_table=False,
    title="你真的很厉害呀哈哈哈哈",
    manual=True
)
