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
        html, body,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        section[data-testid="stMain"],
        .main {
            overflow: hidden !important;
            height: 100vh !important;
            max-height: 100vh !important;
        }
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
    var lockScroll = function () {
        pdoc.documentElement.style.overflow = "hidden";
        pdoc.body.style.overflow = "hidden";
        var stMain = pdoc.querySelector('[data-testid="stMain"]');
        if (stMain) stMain.style.overflow = "hidden";
        var h = window.parent.innerHeight + "px";
        iframe.style.height = h;
        iframe.style.width = "100%";
        iframe.style.border = "none";
        iframe.style.display = "block";
    };
    lockScroll();
    window.parent.addEventListener("resize", lockScroll);
    // Re-apply after Streamlit re-renders
    var obs = new MutationObserver(lockScroll);
    obs.observe(pdoc.body, { childList: true, subtree: true });
})();
</script>
"""

html_fixed = html.replace("</body>", lock_parent_scroll + "\n</body>")
st.components.v1.html(html_fixed, height=900, scrolling=True)
