from typing import Union, Sequence

import streamlit as st
from Levenshtein import opcodes

from mateo_st.utils import cli_args, load_css, load_js


def generate_html_edit_distance(s1: Union[str, Sequence[str]], s2: Union[str, Sequence[str]]):
    if type(s1) != type(s2):
        raise ValueError("Both inputs have to be of the same type")

    sep = "" if isinstance(s1, str) else " "
    src_els = []
    tgt_els = []
    aligned_idx = 0
    for op, i1, i2, j1, j2 in opcodes(s1, s2):
        src_chunk = sep.join(s1[i1:i2])
        tgt_chunk = sep.join(s2[j1:j2])
        onmouseenter = f"enter('aligned-{aligned_idx}')"
        onmouseleave = f"leave('aligned-{aligned_idx}')"
        js_events = f'onmouseenter="{onmouseenter}" onmouseleave="{onmouseleave}"'
        if op == "replace":
            src_els.append(f'<span className="ed-sub-src aligned-{aligned_idx}" {js_events}>{src_chunk}</span>')
            tgt_els.append(f'<span className="ed-sub-tgt aligned-{aligned_idx}" {js_events}>{tgt_chunk}</span>')
            aligned_idx += 1
        elif op == "delete":
            src_els.append(f'<span className="ed-del">{src_chunk}</span>')
        elif op == "insert":
            tgt_els.append(f'<span className="ed-ins">{tgt_chunk}</span>')
        elif op == "equal":
            src_els.append(f'<span className="ed-match aligned-{aligned_idx}" {js_events}>{src_chunk}</span>')
            tgt_els.append(f'<span className="ed-match aligned-{aligned_idx}" {js_events}>{tgt_chunk}</span>')
            aligned_idx += 1

    src_html = f'<div className="ed-src"><p>{sep.join(s1)}</p>' + sep.join(src_els) + '</div>'
    tgt_html = f'<div className="ed-tgt"><p>{sep.join(s2)}</p>' + sep.join(tgt_els) + '</div>'
    return src_html, tgt_html


def _init():
    st.set_page_config(page_title="Visualizing Machine Translations | MATEO", page_icon="💫")
    load_css("base")
    load_css("edit_distance")
    load_js("visualize-hover")

    st.title("💫 Visualizing Machine Translations")
    st.markdown("Here you can visualize edit operations on the word and character-level for given input")


def _visualize():
    html = '<div className="ed-char-level">'
    html += '''<aside className="ed-legend">
    <ul>
    <li><span className="ed-sub-src">substitution source</span></li>
    <li><span className="ed-sub-tgt">substitution target</span></li>
    <li><span className="ed-ins">insertion</span></li>
    <li><span className="ed-del">deletion</span></li>
    <li><span className="ed-match">match</span></li>
    </ul>
    </aside>'''
    html += '<main className="ed-content">'
    src_html, tgt_html = generate_html_edit_distance(
        "I like your style",
        "He likes his hair style"
    )

    html += src_html
    html += tgt_html
    html += '</main></div>'
    st.markdown(html, unsafe_allow_html=True)


def main():
    _init()
    _visualize()




if __name__ == "__main__":
    main()
    # Call this to immediately disable CUDA if needed
    cli_args()
