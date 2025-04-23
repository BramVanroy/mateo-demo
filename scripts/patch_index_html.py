import shutil
from pathlib import Path

import streamlit as st
from bs4 import BeautifulSoup


def patch_index(input_file: str | None, restore: bool = False) -> None:
    if not input_file and not restore:
        raise ValueError("Either --input_file and/or --restore must be provided.")

    pf_index = Path(st.__file__).parent / "static" / "index.html"
    pf_index_bak = pf_index.with_suffix(".bak.html")

    if restore:
        if pf_index_bak.exists():
            pf_index.unlink()
            shutil.copy(pf_index_bak, pf_index)
            print(f"Restored the original index.html from {pf_index_bak}.")
        else:
            print("No backup file found. Nothing to restore.")

    if not input_file:
        print("No input file provided. Nothing to inject.")
        return

    if not pf_index.exists():
        raise FileNotFoundError(f"Streamlit index file {pf_index} not found.")

    if not pf_index_bak.exists():
        shutil.copy(pf_index, pf_index_bak)

    pf_text = Path(input_file)
    if not pf_text.exists() or pf_text.stat().st_size == 0:
        raise FileNotFoundError(f"Input file {pf_text} does not exist or is empty.")

    parsed_input_soup = BeautifulSoup(pf_text.read_text(encoding="utf-8"), "html.parser")

    # Find ID of the outer element in the input file. If no id attribute is found, throw an error.
    elements_to_inject = parsed_input_soup.contents
    top_el = next((el for el in elements_to_inject if el.name is not None), None)
    if top_el is None:
        raise ValueError("No HTML elements found in the file.")

    element_id = top_el.get("id")
    if not element_id:
        raise ValueError(
            "Top-most element does not have a non-empty 'id' attribute"
            " and `id` is required to inject the content to avoid duplication."
        )

    target_soup = BeautifulSoup(pf_index.read_text(encoding="utf-8"), "html.parser")
    head = target_soup.head
    if not head:
        raise ValueError("The Streamlit HTML does not have a <head> tag, which is unexpected.")

    if head.find(id=element_id) is not None:
        print(
            f"Nothing to do: the element with ID '{element_id}' already exists in the <head> of the Streamlit index.html file."
            " If you want to restore the original index.html file before injecting, use the --restore option."
        )
        return

    for el in elements_to_inject:
        head.append(el)

    # Save the modified HTML back to the index.html file
    pf_index.write_text(target_soup.prettify(), encoding="utf-8")

    print(f"Injected content into the <head> of {pf_index}.")
    print(f"Backup of the original index.html created at {pf_index_bak}.")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(
        description="Inject content into the head of the Streamlit index.html file. Example: python patch_index.py --input_file my_content.html"
    )
    cparser.add_argument(
        "--input_file",
        help="Path to the input file containing the content to inject. Make sure the outer-most element has a non-empty id attribute that is very specific (unique)."
        " If not given and restore=True, the script will restore the original index.html file.",
    )
    cparser.add_argument(
        "--restore",
        action="store_true",
        help="Restore the original index.html file from the backup before patching. This will remove any previously injected content.",
    )
    cargs = cparser.parse_args()
    patch_index(cargs.input_file, cargs.restore)
