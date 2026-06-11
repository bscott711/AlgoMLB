import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

from fadegoblin.browser_fliff import _setup_radar_bypass, _dismiss_modals

STATE_PATH = Path("/home/opc/AlgoMLB/fliff_state.json")

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            viewport={"width": 390, "height": 844},
            storage_state=STATE_PATH,
        )
        page = ctx.new_page()
        _setup_radar_bypass(page)
        page.goto("https://sports.getfliff.com/my-picks")
        page.wait_for_timeout(5000)
        
        _dismiss_modals(page)
        
        # Switch to Fliff Cash
        try:
            switcher = page.locator("div.switcher").first
            if switcher.is_visible(timeout=3000):
                switcher.click()
                page.wait_for_timeout(2000)
        except Exception as e:
            pass

        _dismiss_modals(page)
        
        # switch to settled
        try:
            page.get_by_text("Settled", exact=True).first.click()
            page.wait_for_timeout(3000)
        except Exception as e:
            print(f"could not click settled: {e}")

        slips = page.locator("div.activity-feed-row").all()
        for i, slip in enumerate(slips[:3]):
            print(f"--- SLIP {i} ---")
            print(slip.inner_text())
        browser.close()

if __name__ == "__main__":
    main()
