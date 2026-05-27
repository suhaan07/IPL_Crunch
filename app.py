import streamlit as st

st.set_page_config(
    page_title="IPL Crunch '26",
    page_icon="🏏",
    layout="wide"
)

# Hide Streamlit's default chrome so the HTML fills the screen cleanly
st.markdown("""
    <style>
        #MainMenu, header, footer { display: none !important; }
        .block-container { padding: 0 !important; max-width: 100% !important; }
        .stApp { background: #08090d; }
    </style>
""", unsafe_allow_html=True)

with open("ipl_crunch_deliverable.html", "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=6000, scrolling=True)
