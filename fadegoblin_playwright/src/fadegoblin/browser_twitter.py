"""Post to Twitter/X via Playwright using pre-authenticated cookies.

Skips login entirely by injecting saved session cookies.
If cookies are expired, falls back to twikit to re-authenticate and writes
fresh cookies that Playwright can pick up on the next run.

Run with:  xvfb-run --auto-servernum -- python this_script.py
"""

import asyncio
import json
import random
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from fadegoblin import config

COOKIES_PATH = Path("/home/opc/AlgoMLB/twitter_cookies.json")

STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
window.chrome = { runtime: {} };
"""


def _rand_sleep(lo: float = 0.8, hi: float = 2.0) -> None:
    time.sleep(random.uniform(lo, hi))


# ---------------------------------------------------------------------------
# twikit helpers (async, run via asyncio.run when needed)
# ---------------------------------------------------------------------------

async def _twikit_refresh_cookies() -> bool:
    """Use twikit to log in and write fresh cookies for Playwright.

    Credentials are sourced from dotenv (config.py) — no raw os.environ calls.
    Returns True if new cookies were saved successfully.
    """
    try:
        from twikit import Client as TwikitClient  # lazy import – optional dep
    except ImportError:
        print("⚠️  twikit is not installed. Cannot refresh cookies automatically.")
        return False

    username = config.TWITTER_USERNAME
    email = getattr(config, "TWITTER_EMAIL", None)
    password = config.TWITTER_PASSWORD

    if not all([username, email, password]):
        print("⚠️  TWITTER_USERNAME / TWITTER_EMAIL / TWITTER_PASSWORD not set in .env.")
        return False

    print("🔑 Attempting twikit re-login to refresh session cookies …")
    twikit_cookies_path = COOKIES_PATH.parent / "twikit_cookies.json"

    try:
        client = TwikitClient("en-US")

        # If we have a saved twikit session, try that first
        if twikit_cookies_path.exists():
            try:
                client.load_cookies(str(twikit_cookies_path))
                await client.get_user_by_screen_name(username)
                print("✅ twikit session still valid — reusing saved twikit cookies.")
            except Exception:
                print("   twikit session expired; logging in with credentials …")
                await client.login(
                    auth_info_1=username,
                    auth_info_2=email,
                    password=password,
                )
                client.save_cookies(str(twikit_cookies_path))
        else:
            await client.login(
                auth_info_1=username,
                auth_info_2=email,
                password=password,
            )
            client.save_cookies(str(twikit_cookies_path))

        # Convert twikit cookies → Playwright cookie format and persist
        raw = client.get_cookies()  # returns a dict {name: value, …}
        playwright_cookies = []
        for name, value in raw.items():
            entry: dict = {
                "name": name,
                "value": value,
                "domain": ".x.com",
                "path": "/",
                "secure": True,
                "httpOnly": name in ("auth_token",),
                "sameSite": "Lax" if name == "ct0" else "None",
            }
            playwright_cookies.append(entry)

        with open(COOKIES_PATH, "w") as f:
            json.dump(playwright_cookies, f, indent=2)

        print(f"🍪 Fresh cookies written to {COOKIES_PATH}")
        return True

    except Exception as e:
        print(f"❌ twikit login failed: {e}")
        print("   Twitter may have triggered a CAPTCHA — manual cookie extraction required.")
        return False


def _refresh_cookies_sync() -> bool:
    """Synchronous wrapper around the async twikit refresh."""
    return asyncio.run(_twikit_refresh_cookies())


# ---------------------------------------------------------------------------
# Main Playwright poster
# ---------------------------------------------------------------------------

def post_to_twitter_browser(
    tweet_text: str,
    image_path: Path | None = None,
) -> None:
    """Post a tweet by injecting authenticated cookies — no login needed.

    If the cookies are missing or expired, automatically attempts to refresh
    them via twikit before retrying once.
    """
    for attempt in range(2):  # up to 2 attempts: fresh cookies → retry
        if not COOKIES_PATH.exists():
            print("⚠️  No twitter_cookies.json found.")
            if attempt == 0:
                if not _refresh_cookies_sync():
                    return
                continue
            return

        with open(COOKIES_PATH) as f:
            cookies = json.load(f)

        with sync_playwright() as p:
            print("🐦 Launching browser with saved session …")

            browser = p.chromium.launch(
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
            )
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

            # Inject cookies BEFORE navigating
            ctx.add_cookies(cookies)

            page = ctx.new_page()
            cookies_expired = False

            try:
                # ── 1. Go straight to compose ────────────────────────────
                print("🐦 Navigating to compose …")
                page.goto("https://x.com/compose/tweet")
                _rand_sleep(3, 5)

                page.screenshot(path="twitter_debug_compose.png")

                # Check if we got redirected to login (cookies expired)
                if "/login" in page.url or "/flow/login" in page.url:
                    print("❌ Cookies expired — redirected to login.")
                    page.screenshot(path="twitter_error_expired.png")
                    cookies_expired = True
                    browser.close()
                    if attempt == 0:
                        print("🔄 Refreshing session via twikit and retrying …")
                        if not _refresh_cookies_sync():
                            return
                        break  # restart loop with fresh cookies
                    return

                # ── 2. Type the tweet ────────────────────────────────────
                print("🐦 Drafting tweet …")
                textarea = page.wait_for_selector(
                    '[data-testid="tweetTextarea_0"]', timeout=15000
                )
                textarea.click()
                _rand_sleep(0.3, 0.6)

                # Tweet box is contenteditable — use insertText for full Unicode support
                page.keyboard.insert_text(tweet_text)

                _rand_sleep(1, 2)
                page.screenshot(path="twitter_debug_drafted.png")

                # ── 3. Image upload (optional) ───────────────────────────
                if image_path and image_path.exists():
                    print(f"🐦 Uploading image: {image_path.name}")
                    file_input = page.locator('input[data-testid="fileInput"]').first
                    file_input.set_input_files(str(image_path))

                    page.wait_for_selector(
                        '[data-testid="attachments"]', timeout=20000
                    )
                    _rand_sleep(1, 2)

                # ── 4. Post ──────────────────────────────────────────────
                page.get_by_role("button", name="Post", exact=True).click()
                print("🐦 POSTING …")

                # Wait for completion
                page.wait_for_url("https://x.com/home", timeout=30000)
                print("✅ Tweet posted successfully!")
                return  # done — exit the retry loop

            except PlaywrightTimeoutError as e:
                print(f"❌ Timed out: {e}", file=sys.stderr)
                page.screenshot(path="twitter_error.png")
                raise e
            except Exception as e:
                if cookies_expired:
                    pass  # handled above
                else:
                    print(f"❌ Error during Twitter browser automation: {e}", file=sys.stderr)
                    if "page" in locals():
                        page.screenshot(path="twitter_error.png")
                    raise e
            finally:
                # ── 5. Save fresh cookies (session persistence) ──────────
                if not cookies_expired:
                    try:
                        new_cookies = ctx.cookies()
                        with open(COOKIES_PATH, "w") as f:
                            json.dump(new_cookies, f, indent=2)
                        print("🍪 Session cookies refreshed and saved.")
                    except Exception as cookie_err:
                        print(f"⚠️ Failed to save refreshed cookies: {cookie_err}")

                if "browser" in locals():
                    browser.close()
