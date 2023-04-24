# TODO MATEO

- Add "visualization" page for CharCUT
- Add score calculation (in parallel?)
- Add score download options and visualizations
- Maybe move CSS to CSS files and load like so:

```python
def load_css(file_name: str):
    pfcss = Path(__file__).parent.joinpath(f"css/{file_name}.css")
    st.markdown(f'<style>{Path(file_name).read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)

load_css("base")
```

- Add more options to argparse
  - parallel compute
