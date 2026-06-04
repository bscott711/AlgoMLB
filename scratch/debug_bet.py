import json
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

    team_el = page.get_by_text("Diamondbacks", exact=False).first
    if team_el.is_visible():
        team_row = team_el.locator("xpath=./ancestor::div[contains(@class, 'home-card__row')]").first
        odds_row = team_row.locator("xpath=./following-sibling::div[contains(@class, 'home-card__row')]").first
        odds_buttons = odds_row.locator("div[class*='card-home-proposal']").all()
        target_btn = odds_buttons[0]
        print(f"Clicking odds button... HTML: {target_btn.inner_html()[:200]}")
        target_btn.click(force=True)
        page.wait_for_timeout(3000)
        page.screenshot(path="fliff_bet_debug1.png")
        
        ticket = page.locator("div.minimized-ticket-container").first
        if ticket.is_visible():
            print(f"Clicking minimized ticket... text: {ticket.inner_text()}")
            ticket.click(force=True)
            page.wait_for_timeout(3000)
            page.screenshot(path="fliff_bet_debug2.png")
            
            print("Dumping inputs...")
            inputs = page.locator("input").all()
            for i in inputs:
                if i.is_visible():
                    print(f"Input: type={i.get_attribute('type')} class={i.get_attribute('class')} id={i.get_attribute('id')}")
            
            print("Dumping body text around 'Risk'...")
            body_text = page.locator("body").inner_text()
            if "Risk" in body_text:
                print("Found Risk in body text!")
            else:
                print("Did NOT find Risk in body text")

            print(f"Ticket container HTML: {page.locator('.mobile-ticket-container').inner_html()[:1000]}")

    browser.close()
