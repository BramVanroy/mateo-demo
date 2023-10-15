import pytest
from playwright.sync_api import Page, expect

translation_url = f"{pytest.mateo_st_local_url}/Translate"


def test_swap_languages(page: Page):
    page.goto(translation_url)
    expect(page.get_by_label("Selected English. Source language")).to_be_attached()
    expect(page.get_by_label("Selected Dutch. Target language")).to_be_attached()
    page.get_by_role("button", name="⇄").click()
    expect(page.get_by_label("Selected Dutch. Source language")).to_be_attached()
    expect(page.get_by_label("Selected English. Target language")).to_be_attached()


def test_textarea(page: Page, test_data_dir):
    """Test textarea input on the translation page. The table with translations must contain the
    same number of rows as there are non-empty lines in the input text.
    """
    page.goto(translation_url)

    page.get_by_label("File upload?").wait_for(state="attached")
    expect(page.get_by_label("File upload?")).not_to_be_checked()

    text = (test_data_dir / "mt1.txt").read_text(encoding="utf-8")
    num_lines = len([lstrip for l in text.splitlines() if (lstrip := l.strip())])

    page.get_by_label("Sentences to translate", exact=True).fill(text)
    page.keyboard.press("Control+Enter")
    page.get_by_role("button", name="Translate").click()

    page.get_by_text("Done translating!").wait_for()
    results_table_el = page.locator(f'table[aria-rowcount="{num_lines + 1}"]')
    # The table itself is hidden through some streamlit magic, so only wait until it is attached (not 'visible') in DOM
    results_table_el.wait_for(state="attached")
    expect(results_table_el).to_have_attribute("aria-rowcount", str(num_lines + 1))
    expect(results_table_el).to_have_attribute("aria-colcount", "3")


def test_file_upload(page: Page, test_data_dir):
    """Test file upload input on the translation page. The table with translations must contain the
    same number of rows as there are non-empty lines in the input text.
    """
    page.goto(translation_url)

    page.locator("label").filter(has=page.get_by_label("File upload?", exact=True)).click()
    expect(page.get_by_label("File upload?")).to_be_checked()

    page.locator('[data-testid="stFileUploader"]').wait_for()

    with page.expect_file_chooser() as fc_info:
        page.get_by_text("Browse files").click()
    file_chooser = fc_info.value
    num_lines = len(
        [lstrip for l in (test_data_dir / "mt1.txt").read_text(encoding="utf-8").splitlines() if (lstrip := l.strip())]
    )
    file_chooser.set_files(test_data_dir / "mt1.txt")

    page.get_by_role("button", name="Translate").click()

    page.get_by_text("Done translating!").wait_for()
    results_table_el = page.locator(f'table[aria-rowcount="{num_lines + 1}"]')
    # The table itself is hidden through some streamlit magic, so only wait until it is attached (not 'visible') in DOM
    results_table_el.wait_for(state="attached")
    expect(results_table_el).to_have_attribute("aria-rowcount", str(num_lines + 1))
    expect(results_table_el).to_have_attribute("aria-colcount", "3")
