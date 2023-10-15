import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect


def test_source_input_field(page: Page):
    evaluation_url = f"{pytest.mateo_st_local_url}/Evaluate"
    page.goto(evaluation_url)

    page.get_by_label("Use COMET").wait_for(state="attached")

    # Uncheck all metrics and then only enable COMET and then check that the source input is indeed present
    for label in page.locator("label").filter(has=page.get_by_label(re.compile(r"^Use .*"))).all():
        # Have to click on the label because the checkbox is hidden and can't be clicked
        if label.get_by_label("Use").is_checked():
            if label.text_content() != "Use COMET":
                label.click()
        # If checkbox not checked and it's COMET, check it
        elif label.text_content() == "Use COMET":
            label.click()

    expect(page.get_by_label("Source file", exact=True)).to_be_visible()

    # Unchecking COMET option (so no metrics attached)
    page.locator("label").filter(has=page.get_by_label("Use COMET")).click()
    expect(page.get_by_label("Source file", exact=True)).not_to_be_attached()
    # ... but the source file can be optionally provided (useful for output table)
    expect(page.get_by_label("Source file (optional)", exact=True)).to_be_attached()


def _set_input_file(page, label_text: str, test_file_path: Path):
    with page.expect_file_chooser() as fc_info:
        page.get_by_role("button", name=label_text).click()

    file_chooser = fc_info.value
    file_chooser.set_files(test_file_path)


def test_pipeline(page: Page, test_data_dir):
    evaluation_url = f"{pytest.mateo_st_local_url}/Evaluate"
    page.goto(evaluation_url)

    page.get_by_label("Use TER").wait_for(state="attached")

    # Only test for the non-neural metrics; disable neural metrics
    for label in page.locator("label").filter(has=page.get_by_label(re.compile(r"^Use .*"))).all():
        # Have to click on the label because the checkbox is hidden and can't be clicked
        if label.get_by_label("Use").is_checked():
            if label.text_content() not in ["Use BLEU", "Use ChrF", "Use TER"]:
                label.click()

    # TEST THAT SOURCE FILE (OPTIONAL) IS VISIBLE
    expect(page.get_by_label("Source file", exact=True)).not_to_be_visible()
    expect(page.get_by_label("Source file (optional)", exact=True)).to_be_visible()

    # Add data
    _set_input_file(page, "Reference file", test_data_dir / "refs.txt")
    _set_input_file(page, "Source file (optional)", test_data_dir / "srcs.txt")
    _set_input_file(page, "System #1 file", test_data_dir / "mt1.txt")

    # TEST THAT ALL REQUIREMENTS WERE FILLED AND EVALUATE BUTTON IS VISIBLE
    expect(page.get_by_role("button", name="Evaluate MT")).to_be_visible()
