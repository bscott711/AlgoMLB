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
        odds_buttons = odds_row.locator("div.card-home-proposal").all()
        target_btn = odds_buttons[1]  # TOR is home
        target_btn.click(force=True)
        page.wait_for_timeout(2000)
        
        ticket = page.locator("div.minimized-ticket-container").first
        if ticket.is_visible():
            ticket.click(force=True)
            page.wait_for_timeout(2000)
            
            amount_display = page.locator("div.risk-amount-input").first
            amount_display.click(force=True)
            page.wait_for_timeout(1000)
            
            for _ in range(6):
                page.keyboard.press("Backspace")
                page.wait_for_timeout(50)
            
            page.keyboard.press("1")
            page.wait_for_timeout(1000)
            
            submit_btn = page.locator("button.ticket-submit-button, button:has-text('SUBMIT'), button:has-text('Submit')").first
            if submit_btn.is_visible():
                submit_btn.click()
                page.wait_for_timeout(4000)
                
                # Bet is submitted. Let's find the "Pick Submitted" text
                submitted = page.get_by_text("Pick Submitted", exact=False).first
                if submitted.is_visible():
                    # Find the outermost container that isn't the body
                    container = submitted.locator("xpath=./ancestor::div[contains(@class, 'mobile-ticket-container') or contains(@class, 'modal') or contains(@class, 'bottom-sheet') or contains(@class, 'ticket')]").last
                    
                    if not container.is_visible():
                        container = submitted.locator("xpath=./ancestor::div[1]").first
                    
                    print(f"Container class: {container.get_attribute('class')}")
                    container.screenshot(path="fliff_receipt_element.png")
                    print("Dumped screenshot of just the element.")

    browser.close()
