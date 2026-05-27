import streamlit as st

st.set_page_config(
    page_title="IPL Crunch '26",
    page_icon="🏏",
    layout="wide"
)

st.markdown("""
    <style>
        #MainMenu, header, footer { display: none !important; }
        .block-container { padding: 0 !important; max-width: 100% !important; margin: 0 !important; }

        /* Lock the outer Streamlit page so only the iframe scrolls (one scrollbar) */
        html, body, .stApp { overflow: hidden !important; height: 100vh !important; margin: 0 !important; padding: 0 !important; }

        /* Make the iframe fill the full browser viewport.
           This also fixes the hero: inside the iframe 100vh now equals the real
           viewport height, not the 6000 px height attribute. */
        iframe { height: 100vh !important; display: block !important; border: none !important; width: 100% !important; }
    </style>
""", unsafe_allow_html=True)

with open("ipl_crunch_deliverable.html", "r", encoding="utf-8") as f:
    html = f.read()

# height= is overridden to 100vh by the CSS above; scrolling=True keeps one scrollbar inside the iframe
st.components.v1.html(html, height=800, scrolling=True)
