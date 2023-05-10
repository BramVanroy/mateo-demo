import streamlit as st
from Levenshtein import opcodes
from mateo_st.components.ed_visualizer import ed_visualizer
from mateo_st.utils import cli_args, load_css


def calculate_edit_distances(s1: str, s2: str):
    if type(s1) != type(s2):
        raise ValueError("Both inputs have to be of the same type")

    data = {"src": s1, "tgt": s2}
    for unit in ("char", "token"):
        if unit == "token":
            s1 = s1.split()
            s2 = s2.split()

        sep = "" if isinstance(s1, str) else " "
        data[unit] = []
        for op, i1, i2, j1, j2 in opcodes(s1, s2):
            src_chunk = sep.join(s1[i1:i2])
            tgt_chunk = sep.join(s2[j1:j2])
            if op == "replace":
                data[unit].append(("replace", src_chunk, tgt_chunk))
            elif op == "delete":
                data[unit].append(("delete", src_chunk, None))
            elif op == "insert":
                data[unit].append(("insert", None, tgt_chunk))
            elif op == "equal":
                data[unit].append(("equal", src_chunk, tgt_chunk))

    return data


def _init():
    st.set_page_config(page_title="Visualizing Machine Translations | MATEO", page_icon="💫")
    load_css("base")

    st.title("💫 Visualizing Machine Translations")
    st.markdown("Here you can visualize edit operations on the word and character-level for given input")


def _visualize():
    s1 = "I like your style"
    s2 = "He likes his hair style"
    edit_distances = calculate_edit_distances(s1, s2)

    ed_visualizer(**edit_distances)


def main():
    _init()
    _visualize()


if __name__ == "__main__":
    main()
    # Call this to immediately disable CUDA if needed
    cli_args()
