import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect


def test_visualization(page: Page):
    evaluation_url = f"{pytest.mateo_st_local_url}/Visualize"
    page.goto(evaluation_url)

    ref_textarea = page.get_by_label("Reference sentences")
    mt_textarea = page.get_by_label("MT sentences")

    ref_textarea.fill("This is a reference sentence.\nAnother reference sentence.")
    mt_textarea.fill("This is a machine translation sentence.\nAnother machine translation sentence.")

    page.get_by_role("button", name="Visualize").click()

    # markdowncontainer with text "Sentence 1/2" is visible
    expect(page.get_by_text("Sentence 1/2")).to_be_visible()

    # expect element with class ed-legend to be visible
    expect(page.get_by_role("complementary", name="legend")).to_be_visible()

    # expect iFrame with class stCustomComponentV1 to be visible
    expect(page.get_by_test_id("stCustomComponentV1")).to_be_visible()

