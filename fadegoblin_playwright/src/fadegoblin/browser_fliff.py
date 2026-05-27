"""Automates Fliff interaction via Playwright to fetch green slips.

Bypasses Radar.io geolocation verification by intercepting the track API call.
Uses mobile viewport to avoid the desktop phone-mockup wrapper.
"""

import json
import random
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from fadegoblin import config

STATE_PATH = Path("/home/opc/AlgoMLB/fliff_state.json")

STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
window.chrome = { runtime: {} };
"""

# Spoofed Radar.io response — tells Fliff we're in Chicago, IL
_FAKE_RADAR_RESPONSE = json.dumps({
    "meta": {"code": 200},
    "location": {"type": "Point", "coordinates": [-87.6298, 41.8781]},
    "user": {
        "_id": "radar_user", "userId": "2037443",
        "deviceId": "feac6199-0963-4474-b392-67e78cd776b3",
        "location": {"type": "Point", "coordinates": [-87.6298, 41.8781]},
        "locationAuthorization": "GRANTED_FOREGROUND",
        "country": {"code": "US", "name": "United States"},
        "state": {"code": "IL", "name": "Illinois"},
        "dma": {"code": "602", "name": "Chicago"},
        "postalCode": {"code": "60601"},
        "geofences": [],
        "fraud": {
            "passed": True, "bypassed": False, "verified": True,
            "proxy": False, "mocked": False, "compromised": False,
            "jumped": False, "sharing": False,
        },
        "insights": {"state": {"home": True, "office": False, "traveling": False}},
    },
})


def _rand_sleep(lo: float = 0.8, hi: float = 2.0) -> None:
    time.sleep(random.uniform(lo, hi))


def _create_fliff_context(playwright_instance):
    """Create a Playwright browser context configured for Fliff with Radar bypass."""
    browser = playwright_instance.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
    )
    ctx = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
            "Mobile/15E148 Safari/604.1"
        ),
        viewport={"width": 390, "height": 844},
        locale="en-US",
        geolocation={"latitude": 41.8781, "longitude": -87.6298},
        permissions=["geolocation"],
        storage_state=STATE_PATH,
    )
    ctx.add_init_script(STEALTH_JS)
    return browser, ctx


def _setup_radar_bypass(page):
    """Intercept Radar.io geolocation calls and return spoofed PA location."""
    def _handle(route):
        route.fulfill(
            status=200,
            content_type="application/json; charset=utf-8",
            body=_FAKE_RADAR_RESPONSE,
        )
    page.route("https://api-verified.radar.io/v1/track", _handle)
    page.route("https://api.radar.io/v1/track", _handle)


def _dismiss_modals(page):
    """Dismiss any popups/modals that overlay the UI (e.g. cash credit modal)."""
    for selector in [
        "section.cash-credit-modal button",
        "button:has-text('Collect')",
        "button:has-text('Close')",
        "button:has-text('Got it')",
        "button:has-text('OK')",
        "[class*='modal'] button",
        "[class*='close']",
    ]:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=1000):
                el.click(timeout=2000)
                page.wait_for_timeout(500)
        except Exception:
            pass


def interactive_login_session() -> None:
    """Launches a visible Chromium window for manual login, and saves session cookies once successful."""
    print("🔑 Launching visible Chromium browser for manual Fliff login...")
    print("👉 Please log in to Fliff. Once logged in and on the main sports page, the session will save automatically after 60 seconds (or you can close the browser when ready).")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )
        if STATE_PATH.exists():
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-US",
                storage_state=STATE_PATH
            )
        else:
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )

        ctx.add_init_script(STEALTH_JS)

        page = ctx.new_page()
        page.goto("https://sports.getfliff.com/")

        print("⏳ Waiting for you to log in...")

        try:
            page.wait_for_timeout(60000)
            ctx.storage_state(path=STATE_PATH)

            print(f"🍪 Fresh session state successfully written to {STATE_PATH}!")
            print("✅ Manual session capture complete.")
        except Exception as e:
            print(f"❌ Error during manual login capture: {e}")
        finally:
            browser.close()


def fetch_green_slip(pick_name: str) -> Path | None:
    """Fetch a screenshot of the Fliff slip matching the given pick name.

    Uses Radar.io interception to bypass geolocation verification,
    then navigates to My Picks > Settled to find the matching bet.
    """
    if not STATE_PATH.exists():
        print("❌ Fliff state not found. Please run: uv run python -m fadegoblin.browser_fliff --login")
        return None

    with sync_playwright() as p:
        print(f"🍭 Launching browser to fetch Fliff slip for '{pick_name}' ...")
        browser, ctx = _create_fliff_context(p)
        page = ctx.new_page()
        _setup_radar_bypass(page)

        target_path = config.BASE_DIR / "fliff_slip.png"

        try:
            # Navigate directly to My Picks (avoids cash-credit modal on /sports)
            print("🍭 Navigating to My Picks...")
            page.goto("https://sports.getfliff.com/my-picks")
            page.wait_for_timeout(8000)

            if "verify-location" in page.url:
                print("⚠️ Location verification still blocking. Retrying...")
                page.goto("https://sports.getfliff.com/my-picks")
                page.wait_for_timeout(8000)

            _dismiss_modals(page)

            # Switch to Fliff Cash mode (default is Fliff Coins)
            # The toggle is a div.switcher element in the header
            print("🍭 Switching to Fliff Cash...")
            try:
                switcher = page.locator("div.switcher").first
                if switcher.is_visible(timeout=3000):
                    switcher.click()
                    page.wait_for_timeout(2000)
                    print("   ✅ Toggled to Fliff Cash")
            except Exception as e:
                print(f"   ⚠️ Could not toggle to Fliff Cash: {e}")

            _dismiss_modals(page)

            # Click the Settled tab
            print("🍭 Clicking Settled tab...")
            try:
                settled_tab = page.get_by_text("Settled", exact=True).first
                if settled_tab.is_visible(timeout=3000):
                    settled_tab.click()
                    page.wait_for_timeout(3000)
            except Exception:
                pass

            # Scroll through settled bets looking for the pick
            print(f"🍭 Searching for '{pick_name}' in settled bets...")
            found = False
            for scroll_attempt in range(5):
                try:
                    pick_el = page.get_by_text(pick_name, exact=False).first
                    if pick_el.is_visible(timeout=2000):
                        print(f"✅ Found '{pick_name}'!")
                        
                        # Find the parent slip container
                        slip_container = pick_el.locator("xpath=./ancestor::div[contains(@class, 'activity-feed-row')]").first
                        
                        if slip_container.is_visible(timeout=2000):
                            slip_container.screenshot(path=str(target_path))
                        else:
                            # Fallback to the text element if container not found
                            pick_el.screenshot(path=str(target_path))
                        
                        found = True
                        break
                except Exception:
                    pass

                # Scroll down to load more bets
                page.mouse.wheel(0, 600)
                page.wait_for_timeout(1500)

            if not found:
                # Fallback: screenshot the entire settled page
                print(f"⚠️ Could not find '{pick_name}'. Taking full settled page screenshot.")
                page.screenshot(path=str(target_path), full_page=True)

            return target_path

        except PlaywrightTimeoutError as e:
            print(f"❌ Timed out: {e}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error during Fliff browser automation: {e}", file=sys.stderr)
        finally:
            try:
                ctx.storage_state(path=STATE_PATH)
            except Exception:
                pass
            browser.close()

    return None


def place_fliff_bet(pick_name: str, amount: int, use_coins: bool = True) -> bool:
    """Place a bet on Fliff for a specific team.
    
    1. Finds the game containing pick_name.
    2. Identifies if the pick is the home (index 1) or away (index 0) team.
    3. Clicks the corresponding moneyline odds button.
    4. Enters the amount in the betslip.
    5. Clicks submit.
    """
    if not STATE_PATH.exists():
        print("❌ Fliff state not found. Please run: uv run python -m fadegoblin.browser_fliff --login")
        return False

    with sync_playwright() as p:
        print(f"💰 Launching browser to place {amount} {'Coins' if use_coins else 'Cash'} bet on '{pick_name}'...")
        browser, ctx = _create_fliff_context(p)
        page = ctx.new_page()
        _setup_radar_bypass(page)

        try:
            print("💰 Navigating to Fliff Sports...")
            page.goto("https://sports.getfliff.com/")
            page.wait_for_timeout(8000)
            
            _dismiss_modals(page)

            # Ensure we are on the right currency
            print(f"💰 Ensuring we are in Fliff {'Coins' if use_coins else 'Cash'} mode...")
            try:
                for _ in range(3):
                    header_text = page.locator("body").inner_text()
                    is_coins = "Fliff Coins" in header_text
                    is_cash = "Fliff Cash" in header_text
                    
                    switcher = page.locator("div.switcher").first
                    if switcher.is_visible(timeout=3000):
                        if use_coins and is_cash:
                            print("   Toggling from Cash to Coins...")
                            switcher.click()
                            page.wait_for_timeout(2000)
                        elif not use_coins and is_coins:
                            print("   Toggling from Coins to Cash...")
                            switcher.click()
                            page.wait_for_timeout(2000)
                        else:
                            print(f"   ✅ Confirmed {'Coins' if use_coins else 'Cash'} mode")
                            break
            except Exception as e:
                print(f"   ⚠️ Could not verify/toggle currency: {e}")

            _dismiss_modals(page)

            # Search for the team
            print(f"💰 Searching for '{pick_name}'...")
            team_el = page.get_by_text(pick_name, exact=False).first
            
            # Check multiple times and scroll if needed
            found = False
            for _ in range(5):
                if team_el.is_visible(timeout=2000):
                    found = True
                    break
                page.mouse.wheel(0, 600)
                page.wait_for_timeout(1000)

            if not found:
                print(f"❌ Could not find '{pick_name}' on the board.")
                return False

            print(f"✅ Found '{pick_name}'!")
            
            # Find the row containing the team names
            team_row = team_el.locator("xpath=./ancestor::div[contains(@class, 'home-card__row')]").first
            if not team_row.is_visible(timeout=2000):
                print("❌ Could not find the game container row.")
                return False

            # Determine if target is team 0 (Away) or team 1 (Home)
            team_names = team_row.locator("div[class*='home-card__name']").all()
            target_index = -1
            for i, name_el in enumerate(team_names):
                if pick_name.lower() in name_el.inner_text().lower():
                    target_index = i
                    break
                    
            if target_index == -1:
                # Fallback to index 0
                target_index = 0
                
            print(f"   Team is index {target_index} (0=Away, 1=Home)")

            # The odds row is usually the immediate next sibling
            odds_row = team_row.locator("xpath=./following-sibling::div[contains(@class, 'home-card__row')]").first
            if not odds_row.is_visible(timeout=2000):
                print("❌ Could not find the odds row sibling.")
                return False

            # Find the odds buttons
            odds_buttons = odds_row.locator("div.card-home-proposal").all()
            if len(odds_buttons) <= target_index:
                print(f"❌ Not enough odds buttons found ({len(odds_buttons)}).")
                return False

            target_btn = odds_buttons[target_index]
            
            # Check if it's locked
            if "lock" in target_btn.inner_html().lower():
                print(f"❌ The odds for '{pick_name}' are currently locked (live game suspended/pitch happening).")
                return False

            print(f"💰 Clicking odds button for '{pick_name}'...")
            target_btn.click(force=True)
            page.wait_for_timeout(2000)

            # Check if bet was added to ticket by checking the counter
            ticket = page.locator("div.minimized-ticket-container").first
            if ticket.is_visible(timeout=3000):
                ticket_text = ticket.inner_text()
                if "0" in ticket_text.split():
                    print("❌ Bet was not added to the ticket (counter still 0).")
                    return False
                print(f"💰 Opening betslip (ticket counter: {ticket_text})...")
                ticket.click(force=True)
                page.wait_for_timeout(2000)

            # Look for betslip input
            print("💰 Checking betslip input...")
            amount_display = page.locator("div.risk-amount-input").first
            
            if amount_display.is_visible(timeout=3000):
                print(f"💰 Entering stake: {amount}")
                amount_display.click(force=True)
                page.wait_for_timeout(1000)
                
                # Clear existing value using the virtual keypad backspace (if visible) or keyboard
                for _ in range(6):
                    page.keyboard.press("Backspace")
                    page.wait_for_timeout(50)
                
                # Use the virtual keypad buttons to enter the amount
                for digit in str(amount):
                    try:
                        # The virtual numpad uses elements with class keyboard-button and exact text
                        key = page.locator(f"div.keyboard-button:has-text('{digit}')").first
                        if key.is_visible(timeout=1000):
                            key.click(force=True)
                        else:
                            # Fallback to standard keyboard press if custom UI is missing
                            page.keyboard.press(digit)
                    except Exception:
                        page.keyboard.press(digit)
                    page.wait_for_timeout(100)
                page.wait_for_timeout(1000)
                
                current_amount = page.locator("span.risk-amount-input__amount").first.inner_text()
                print(f"   Slip now shows wager is: {current_amount}")
                
                # Submit button
                submit_btn = page.locator("button.ticket-submit-button, button:has-text('SUBMIT'), button:has-text('Submit')").first
                if submit_btn.is_visible(timeout=2000):
                    print("💰 Clicking SUBMIT button...")
                    submit_btn.click()
                    page.wait_for_timeout(5000)  # Wait for submission to complete
                    print("✅ Bet placed successfully!")
                    
                    # Take screenshot of receipt
                    receipt_path = config.BASE_DIR / "fliff_receipt.png"
                    try:
                        # Try to capture just the ticket container
                        ticket_container = page.locator("div.mobile-ticket-container").first
                        if ticket_container.is_visible(timeout=2000):
                            ticket_container.screenshot(path=str(receipt_path))
                        else:
                            page.screenshot(path=str(receipt_path))
                    except Exception:
                        page.screenshot(path=str(receipt_path))
                        
                    print(f"   Saved receipt to {receipt_path}")
                    return True
                else:
                    print("❌ Could not find SUBMIT button.")
            else:
                print("❌ Could not find wager input field in betslip.")

        except Exception as e:
            print(f"❌ Error during Fliff bet placement: {e}")
        finally:
            try:
                ctx.storage_state(path=STATE_PATH)
            except Exception:
                pass
            browser.close()

    return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--login":
        interactive_login_session()
    else:
        print("💡 FadeGoblin Fliff Module")
        print("Use '--login' to manually authenticate and capture cookies:")
        print("  uv run python -m fadegoblin.browser_fliff --login")
