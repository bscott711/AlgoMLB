"""Post to Twitter/X via Playwright using pre-authenticated cookies.

Skips login entirely by injecting saved session cookies.
Run with:  xvfb-run --auto-servernum -- python this_script.py
"""

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


def post_to_twitter_browser(
    tweet_text: str,
    image_path: Path | None = None,
) -> None:
    """Post a tweet by injecting authenticated cookies — no login needed."""

    # Load cookies
    cookies_file = COOKIES_PATH
    if not cookies_file.exists():
        print("⚠️  No twitter_cookies.json found. Skipping X post.")
        return

    with open(cookies_file) as f:
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

        try:
            # ── 1. Go straight to compose ────────────────────────────
            print("🐦 Navigating to compose …")
            page.goto("https://x.com/compose/tweet")
            _rand_sleep(3, 5)

            page.screenshot(path="twitter_debug_compose.png")

            # Check if we got redirected to login (cookies expired)
            if "/login" in page.url or "/flow/login" in page.url:
                print("❌ Cookies expired — redirected to login. Need fresh cookies.")
                page.screenshot(path="twitter_error_expired.png")
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

        except PlaywrightTimeoutError as e:
            print(f"❌ Timed out: {e}", file=sys.stderr)
            page.screenshot(path="twitter_error.png")
            raise e
        except Exception as e:
            print(f"❌ Error during Twitter browser automation: {e}", file=sys.stderr)
            if "page" in locals():
                page.screenshot(path="twitter_error.png")
            raise e
        finally:
            # ── 5. Save fresh cookies (Session persistence) ──────────
            try:
                # We save from context if it exists, regardless of success/failure
                # as long as we were logged in at some point.
                if "context" in locals():
                    new_cookies = context.cookies()
                    with open(COOKIES_PATH, "w") as f:
                        json.dump(new_cookies, f, indent=2)
                    print("🍪 Session cookies refreshed and saved.")
            except Exception as cookie_err:
                print(f"⚠️ Failed to save refreshed cookies: {cookie_err}")
            
            if "browser" in locals():
                browser.close()
