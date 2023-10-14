from playwright.sync_api import Page


def test_translation_file_upload(page: Page, test_data_dir):
    translation_url = "http://localhost:8505/Translate"

    page.goto(translation_url)
    page.locator("label").filter(has=page.get_by_label("File upload?", exact=True)).click()
    assert page.get_by_label("File upload?").is_checked()

    page.locator('[data-testid="stFileUploader"]').wait_for()

    with page.expect_file_chooser() as fc_info:
        page.get_by_text("Browse files").click()
    file_chooser = fc_info.value
    num_lines = len(
        [lstrip for l in (test_data_dir / "mt1.txt").read_text(encoding="utf-8").splitlines() if (lstrip := l.strip())]
    )
    file_chooser.set_files(test_data_dir / "mt1.txt")

    page.get_by_role("button", name="Translate").click()

    # Listen for DOM change event
    # Check if table exists
    # Check that the table has as many rows as the file has lines
    page.get_by_text("Done translating!").wait_for()
    assert page.locator(".stDataFrame").is_visible()
    results_table_el = page.locator(".stDataFrame table")
    assert int(results_table_el.get_attribute("aria-rowcount")) == num_lines + 1
    assert int(results_table_el.get_attribute("aria-colcount")) == 3
