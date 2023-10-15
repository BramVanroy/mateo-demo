import re

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
