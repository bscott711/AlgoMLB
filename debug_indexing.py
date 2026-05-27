import sys
from playwright.sync_api import sync_playwright
from fadegoblin.browser_fliff import _create_fliff_context, _setup_radar_bypass, _dismiss_modals
from pathlib import Path

STATE_PATH = Path("/home/opc/AlgoMLB/fliff_state.json")

with sync_playwright() as p:
    browser, ctx = _create_fliff_context(p)
    page = ctx.new_page()
    _setup_radar_bypass(page)
    
    page.goto("https://sports.getfliff.com/")
    page.wait_for_timeout(8000)
    _dismiss_modals(page)

    team_el = page.get_by_text("TOR", exact=False).first
    if team_el.is_visible():
        team_row = team_el.locator("xpath=./ancestor::div[contains(@class, 'home-card__row')]").first
        odds_row = team_row.locator("xpath=./following-sibling::div[contains(@class, 'home-card__row')]").first
        odds_buttons = odds_row.locator("div[class*='card-home-proposal']").all()
        
        team_names = team_row.locator("div[class*='home-card__name']").all()
        print("Team Names:")
        for i, tn in enumerate(team_names):
            print(f"  {i}: {tn.inner_text()}")
            
        print("\nOdds Buttons:")
        for i, btn in enumerate(odds_buttons):
            print(f"  {i}: {btn.inner_text()}")
            
    browser.close()
