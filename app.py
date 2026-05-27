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
        /* Take the iframe out of normal flow so Streamlit has nothing to scroll */
        iframe {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            border: none !important;
            z-index: 9999 !important;
        }
        /* Kill the outer page scroll */
        html, body { overflow: hidden !important; }
    </style>
""", unsafe_allow_html=True)

with open("ipl_crunch_deliverable.html", "r", encoding="utf-8") as f:
    html = f.read()

# Inject a script that runs inside the iframe and locks the parent page's
# scrollbar — leaving only the iframe's own scrollbar active.
lock_parent_scroll = """
<script>
(function () {
    var pdoc = window.parent ? window.parent.document : document;
    pdoc.documentElement.style.overflow = "hidden";
    pdoc.body.style.overflow = "hidden";
})();
</script>
"""

html_fixed = html.replace("</body>", lock_parent_scroll + "\n</body>")
st.components.v1.html(html_fixed, height=900, scrolling=True)
