from io import StringIO

import streamlit as st
from Levenshtein import distance, opcodes

from mateo_st.components.ed_visualizer import ed_visualizer
from mateo_st.utils import cli_args, load_css, print_citation_info


def _calculate_edit_distances(s1: str, s2: str):
    data = {"ref": s1, "mt": s2}
    for unit in ("char", "token"):
        if unit == "token":
            s1 = s1.split()
            s2 = s2.split()

        sep = "" if isinstance(s1, str) else " "
        data[unit] = []
        for op, i1, i2, j1, j2 in opcodes(s1, s2):
            ref_chunk = sep.join(s1[i1:i2])
            mt_chunk = sep.join(s2[j1:j2])
            if op == "replace":
                data[unit].append(("replace", ref_chunk, mt_chunk))
            elif op == "delete":
                data[unit].append(("delete", ref_chunk, None))
            elif op == "insert":
                data[unit].append(("insert", None, mt_chunk))
            elif op == "equal":
                data[unit].append(("equal", ref_chunk, mt_chunk))

        data[f"{unit}_score"] = distance(s1, s2)

    return data


def _init():
    st.set_page_config(page_title="Visualizing Machine Translations | MATEO", page_icon="💫", layout="wide")
    load_css("base")
    load_css("visualize")

    if "viz_ref_segments" not in st.session_state:
        st.session_state["viz_ref_segments"] = []

    if "viz_mt_segments" not in st.session_state:
        st.session_state["viz_mt_segments"] = []

    if "viz_idx" not in st.session_state:
        st.session_state["viz_idx"] = 0

    st.title("💫 Visualizing Machine Translations")
    st.markdown(
        "Here you can visualize edit operations on the word and character-level for given inputs, like a"
        " reference  and a machine translation."
    )

    print_citation_info()

    st.markdown(
        "A distinction is made between substitutions on the one hand"
        " and insertions/deletions on the other, as indicated by the legend on the left. "
        " Move your mouse over matches or substitutions to highlight the aligned items in the other sentence."
    )


def _data_input():    
    fupload_check = st.checkbox("File upload?")

    form = st.form(key="viz_form", enter_to_submit=False, border=False)
    ref_col, mt_col = form.columns(2)
    if fupload_check:
        uploaded_ref_file = ref_col.file_uploader("Reference file")
        if uploaded_ref_file is not None:
            stringio = StringIO(uploaded_ref_file.getvalue().decode("utf-8"))
            st.session_state["viz_ref_segments"] = stringio.read().splitlines()
        else:
            st.session_state["viz_ref_segments"] = []

        uploaded_mt_file = mt_col.file_uploader("MT file")
        if uploaded_mt_file is not None:
            stringio = StringIO(uploaded_mt_file.getvalue().decode("utf-8"))
            st.session_state["viz_mt_segments"] = stringio.read().splitlines()
        else:
            st.session_state["viz_mt_segments"] = []
    else:
        st.session_state["viz_ref_segments"] = ref_col.text_area(
            label="Reference sentences"
        ).splitlines()
        st.session_state["viz_mt_segments"] = mt_col.text_area(
            label="MT sentences"
        ).splitlines()
    
    submitted = form.form_submit_button(
        "Visualize",
        type="primary",
        use_container_width=False,
    )

    if submitted:
        # To avoid index errors
        st.session_state["viz_idx"] = 0


def _rotator():
    def next_idx():
        st.session_state["viz_idx"] += 1

    def prev_idx():
        st.session_state["viz_idx"] -= 1

    sidebar_ct, main_ct = st.columns((1, 3))
    prev_col, next_col = sidebar_ct.columns(2)
    prev_col.button("Prev", disabled=st.session_state["viz_idx"] == 0, on_click=prev_idx)
    next_col.button(
        "Next",
        disabled=st.session_state["viz_idx"] >= len(st.session_state["viz_ref_segments"]) - 1,
        on_click=next_idx,
    )
    sidebar_ct.info(f"Sentence {st.session_state['viz_idx'] + 1}/{len(st.session_state['viz_ref_segments'])}")

    sidebar_ct.markdown(
        """
        <aside class="ed-legend" aria-label="legend">
        <ul>
            <li><span class="ed-sub-ref">substitution ref</span></li>
            <li><span class="ed-sub-mt">substitution MT</span></li>
            <li><span class="ed-ins">insertion</span></li>
            <li><span class="ed-del">deletion</span></li>
            <li><span class="ed-match">match</span></li>
        </ul>
    </aside>""",
        unsafe_allow_html=True,
    )

    edit_distances = _calculate_edit_distances(
        st.session_state["viz_ref_segments"][st.session_state["viz_idx"]],
        st.session_state["viz_mt_segments"][st.session_state["viz_idx"]],
    )
    with main_ct:
        ed_visualizer(**edit_distances)


def _visualize():
    info_ct = st.empty()
    if (
        "viz_ref_segments" in st.session_state
        and st.session_state["viz_ref_segments"]
        and "viz_mt_segments" in st.session_state
        and st.session_state["viz_mt_segments"]
    ):
        if len(st.session_state["viz_ref_segments"]) != len(st.session_state["viz_mt_segments"]):
            info_ct.warning("Make sure that the reference text and MT text have the same number of lines")
        else:
            info_ct.empty()
            _rotator()
    else:
        info_ct.warning("Make sure to specify content for both the reference and MT")


def main():
    _init()
    _data_input()
    _visualize()


if __name__ == "__main__":
    main()
    # Call this to immediately disable CUDA if needed
    cli_args()
