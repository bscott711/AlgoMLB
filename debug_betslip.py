import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
from fadegoblin.browser_fliff import _create_fliff_context, _setup_radar_bypass, _dismiss_modals

STATE_PATH = Path("/home/opc/AlgoMLB/fliff_state.json")

with sync_playwright() as p:
    browser, ctx = _create_fliff_context(p)
    page = ctx.new_page()
    _setup_radar_bypass(page)
    
    print("Navigating to sports page...")
    page.goto("https://sports.getfliff.com/")
    page.wait_for_timeout(8000)
    _dismiss_modals(page)

    team_el = page.get_by_text("TOR", exact=False).first
    if team_el.is_visible():
        team_row = team_el.locator("xpath=./ancestor::div[contains(@class, 'home-card__row')]").first
        odds_row = team_row.locator("xpath=./following-sibling::div[contains(@class, 'home-card__row')]").first
        odds_buttons = odds_row.locator("div[class*='card-home-proposal']").all()
        target_btn = odds_buttons[1]  # TOR is home
        target_btn.click(force=True)
        page.wait_for_timeout(2000)
        
        ticket = page.locator("div.minimized-ticket-container").first
        if ticket.is_visible():
            ticket.click(force=True)
            page.wait_for_timeout(2000)
            page.screenshot(path="fliff_betslip_open.png")
            
            with open("betslip.html", "w") as f:
                f.write(page.locator("div.mobile-ticket-container").inner_html())
            print("Dumped betslip HTML.")

    browser.close()
