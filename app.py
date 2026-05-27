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
    </style>
""", unsafe_allow_html=True)

with open("ipl_crunch_deliverable.html", "r", encoding="utf-8") as f:
    html = f.read()

# Inject a script that runs inside the iframe and locks the parent page's
# scrollbar — leaving only the iframe's own scrollbar active.
lock_parent_scroll = """
<script>
(function () {
    var iframe = window.frameElement;
    if (!iframe) return;
    var pdoc = window.parent.document;
    pdoc.documentElement.style.overflow = "hidden";
    pdoc.body.style.overflow = "hidden";
    iframe.style.height = window.parent.innerHeight + "px";
    iframe.style.width = "100%";
    iframe.style.border = "none";
    iframe.style.display = "block";
    window.parent.addEventListener("resize", function () {
        iframe.style.height = window.parent.innerHeight + "px";
    });
})();
</script>
"""

html_fixed = html.replace("</body>", lock_parent_scroll + "\n</body>")
st.components.v1.html(html_fixed, height=800, scrolling=True)
