import pytest
from playwright.sync_api import Page, expect


def test_pages_exist(page: Page):
    base_url = pytest.mateo_st_local_url

    page.goto(base_url)
    page.locator('[data-testid="stSidebar"] a:has-text("Translate")').click()
    expect(page).to_have_url(f"{base_url}/Translate")

    page.goto(base_url)
    page.locator('[data-testid="stSidebar"] a:has-text("Evaluate")').click()
    expect(page).to_have_url(f"{base_url}/Evaluate")

    page.goto(base_url)
    page.locator('[data-testid="stSidebar"] a:has-text("Background")').click()
    expect(page).to_have_url(f"{base_url}/Background")

    page.goto(base_url)
    page.locator('[data-testid="stSidebar"] a:has-text("Visualize")').click()
    expect(page).to_have_url(f"{base_url}/Visualize")
