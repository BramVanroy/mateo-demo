from pathlib import Path

import streamlit.components.v1 as components


ed_visualizer = components.declare_component("ed_visualizer", path=str(Path(__file__).resolve().parent))
