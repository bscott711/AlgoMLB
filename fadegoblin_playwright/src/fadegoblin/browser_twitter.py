"""Post to Twitter/X via Playwright using pre-authenticated cookies.

Skips login entirely by injecting saved session cookies.
Provides an interactive manual CLI login captures to renew expired cookies securely.
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


def _show_cookie_instructions() -> None:
    print("\n" + "=" * 80)
    print("❌ ERROR: Twitter/X Session Cookies Expired or Missing!")
    print("=" * 80)
    print(
        "Programmatic credentials-based background login is highly brittle and often blocked by Twitter."
    )
    print(
        "To easily and securely refresh your session cookies, please run the following command in your terminal:"
    )
    print("\n   python -m fadegoblin.browser_twitter --login\n")
    print(
        "This will open a visible browser window where you can log in manually, solve CAPTCHAs,"
    )
    print(
        "and complete 2FA. Once logged in, the active session will be captured automatically."
    )
    print("=" * 80 + "\n")


def interactive_login_session() -> None:
    """Launches a visible Chromium window for manual login, and saves session cookies once successful."""
    print("🔑 Launching visible Chromium browser for manual Twitter login...")
    print(
        "👉 Please log in, solve any CAPTCHAs, and complete 2FA in the browser window."
    )
    print(
        "⏳ Waiting for navigation to the Home page (https://x.com/home or https://twitter.com/home)..."
    )

    with sync_playwright() as p:
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

        # Inject existing cookies if they exist to ease re-auth
        if COOKIES_PATH.exists():
            try:
                with open(COOKIES_PATH) as f:
                    ctx.add_cookies(json.load(f))
            except Exception:
                pass

        page = ctx.new_page()
        page.goto("https://x.com/i/flow/login")

        # Wait up to 5 minutes (300,000 ms) for the user to reach the home page
        try:
            page.wait_for_url(
                lambda url: "x.com/home" in url or "twitter.com/home" in url,
                timeout=300000,
            )
            print("🎉 Detected successful login and landing on home page!")
            _rand_sleep(2, 4)  # let sessions stabilize

            cookies = ctx.cookies()
            with open(COOKIES_PATH, "w") as f:
                json.dump(cookies, f, indent=2)

            print(f"🍪 Fresh session cookies successfully written to {COOKIES_PATH}!")
            print("✅ Manual session capture complete. You can now close the browser.")
        except PlaywrightTimeoutError:
            print(
                "❌ Timeout (5 minutes elapsed) waiting for home page navigation. Login session not saved."
            )
        except Exception as e:
            print(f"❌ Error during manual login capture: {e}")
        finally:
            browser.close()


def post_to_twitter_browser(
    tweet_text: str,
    image_path: Path | None = None,
) -> str | None:
    """Post a tweet by injecting authenticated cookies — no login needed."""
    if not COOKIES_PATH.exists():
        _show_cookie_instructions()
        return None

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
                _show_cookie_instructions()
                return None

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

                page.wait_for_selector('[data-testid="attachments"]', timeout=20000)
                _rand_sleep(1, 2)

            # ── 4. Post ──────────────────────────────────────────────
            print("🐦 POSTING …")
            # Try keyboard shortcut first (Control+Enter)
            print("🐦 Sending keyboard shortcut Control+Enter to post...")
            page.keyboard.press("Control+Enter")
            _rand_sleep(2, 4)

            # Check if modal is still open, if so, fallback to clicking
            textarea_loc = page.locator('[data-testid="tweetTextarea_0"]').first
            if textarea_loc.is_visible():
                print(
                    "🐦 Hotkey post didn't close textarea, trying button click fallbacks..."
                )
                post_btn = None
                for selector in [
                    '[data-testid="tweetButton"]',
                    '[data-testid="tweetButtonInline"]',
                ]:
                    loc = page.locator(selector)
                    for i in range(loc.count()):
                        candidate = loc.nth(i)
                        if candidate.is_visible():
                            post_btn = candidate
                            break
                    if post_btn:
                        break

                if post_btn:
                    try:
                        print("🐦 Clicking post button using selector...")
                        post_btn.click(force=True, timeout=5000)
                    except Exception as click_err:
                        print(
                            f"⚠️ Regular force click failed: {click_err}. Trying DOM click dispatch..."
                        )
                        post_btn.dispatch_event("click")
                else:
                    try:
                        print("🐦 Clicking post button via role backup...")
                        page.get_by_role("button", name="Post", exact=True).click(
                            force=True, timeout=5000
                        )
                    except Exception as role_err:
                        print(
                            f"⚠️ Role force click failed: {role_err}. Trying DOM click dispatch..."
                        )
                        page.get_by_role(
                            "button", name="Post", exact=True
                        ).dispatch_event("click")

            # Wait for completion (url contains x.com/home or is just x.com root)
            page.wait_for_url(
                lambda url: (
                    "x.com/home" in url
                    or "twitter.com/home" in url
                    or url.strip("/") in ["https://x.com", "https://twitter.com"]
                ),
                timeout=30000,
            )
            print("✅ Tweet posted successfully!")

            # Navigate to profile to retrieve tweet ID
            tweet_id = None
            try:
                print(
                    f"🐦 Navigating to profile: https://x.com/{config.TWITTER_USERNAME} ..."
                )
                page.goto(f"https://x.com/{config.TWITTER_USERNAME}")
                _rand_sleep(3, 5)
                # Find the first link containing '/status/'
                first_link = page.locator("a[href*='/status/']").first
                href = first_link.get_attribute("href")
                if href:
                    tweet_id = href.split("/status/")[-1].split("?")[0]
                    print(f"🐦 Extracted latest tweet ID from profile: {tweet_id}")
            except Exception as ex:
                print(f"⚠️ Could not extract tweet ID from profile: {ex}")

            return tweet_id

        except PlaywrightTimeoutError as e:
            print(f"❌ Timed out: {e}", file=sys.stderr)
            page.screenshot(path="twitter_error.png")
            raise e
        except Exception as e:
            if not cookies_expired:
                print(
                    f"❌ Error during Twitter browser automation: {e}", file=sys.stderr
                )
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


def reply_to_twitter_browser(
    reply_text: str,
    parent_tweet_id: str,
) -> str | None:
    """Replies to an existing tweet using Playwright browser automation."""
    if not parent_tweet_id:
        print("⚠️ No parent_tweet_id provided for Twitter reply.")
        return None

    if not COOKIES_PATH.exists():
        _show_cookie_instructions()
        return None

    with open(COOKIES_PATH) as f:
        cookies = json.load(f)

    with sync_playwright() as p:
        print(f"🐦 Launching browser to reply to tweet {parent_tweet_id} …")

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
        ctx.add_cookies(cookies)

        page = ctx.new_page()
        cookies_expired = False

        try:
            # ── 1. Navigate to target tweet ──────────────────────────
            target_url = f"https://x.com/i/status/{parent_tweet_id}"
            print(f"🐦 Navigating to target status URL: {target_url} …")
            page.goto(target_url)
            _rand_sleep(4, 6)

            page.screenshot(path="twitter_debug_reply_page.png")

            # Check if we got redirected to login
            if "/login" in page.url or "/flow/login" in page.url:
                print("❌ Cookies expired — redirected to login.")
                cookies_expired = True
                _show_cookie_instructions()
                return None

            # ── 2. Click the reply/comment box ───────────────────────
            print("🐦 Locating reply text area …")
            textarea = page.wait_for_selector(
                '[data-testid="tweetTextarea_0"]', timeout=15000
            )
            textarea.click()
            _rand_sleep(0.5, 1.0)

            # Type the reply text
            page.keyboard.insert_text(reply_text)
            _rand_sleep(1, 2)

            page.screenshot(path="twitter_debug_reply_drafted.png")

            # Prefer visible tweetButtonInline, then tweetButton, then Reply text role
            reply_btn = None
            for selector in [
                '[data-testid="tweetButtonInline"]',
                '[data-testid="tweetButton"]',
            ]:
                loc = page.locator(selector)
                for i in range(loc.count()):
                    candidate = loc.nth(i)
                    if candidate.is_visible():
                        reply_btn = candidate
                        break
                if reply_btn:
                    break

            # Try keyboard shortcut first (Control+Enter)
            print("🐦 Sending keyboard shortcut Control+Enter to reply...")
            page.keyboard.press("Control+Enter")
            _rand_sleep(2, 4)

            # Check if reply area is still visible
            textarea_loc = page.locator('[data-testid="tweetTextarea_0"]').first
            if textarea_loc.is_visible():
                print(
                    "🐦 Hotkey reply didn't close textarea, trying button click fallbacks..."
                )
                if reply_btn:
                    try:
                        print("🐦 Clicking reply button using selector...")
                        reply_btn.click(force=True, timeout=5000)
                    except Exception as click_err:
                        print(
                            f"⚠️ Regular force click failed: {click_err}. Trying DOM click dispatch..."
                        )
                        reply_btn.dispatch_event("click")
                else:
                    try:
                        print("🐦 Clicking reply button via role backup...")
                        page.get_by_role("button", name="Reply", exact=True).click(
                            force=True, timeout=5000
                        )
                    except Exception as role_err:
                        print(
                            f"⚠️ Role force click failed: {role_err}. Trying DOM click dispatch..."
                        )
                        page.get_by_role(
                            "button", name="Reply", exact=True
                        ).dispatch_event("click")

            print("🐦 CLICKED REPLY …")
            _rand_sleep(4, 6)
            print("✅ Reply posted successfully on Twitter!")

            # Grab the new reply ID by navigating to profile
            reply_tweet_id = None
            try:
                print(
                    f"🐦 Navigating to profile: https://x.com/{config.TWITTER_USERNAME} to find reply ID..."
                )
                page.goto(f"https://x.com/{config.TWITTER_USERNAME}")
                _rand_sleep(3, 5)
                first_link = page.locator("a[href*='/status/']").first
                href = first_link.get_attribute("href")
                if href:
                    reply_tweet_id = href.split("/status/")[-1].split("?")[0]
                    print(f"🐦 Extracted reply tweet ID: {reply_tweet_id}")
            except Exception as ex:
                print(f"⚠️ Could not extract reply tweet ID: {ex}")

            return reply_tweet_id

        except PlaywrightTimeoutError as e:
            print(f"❌ Timed out replying: {e}", file=sys.stderr)
            page.screenshot(path="twitter_error_reply.png")
            raise e
        except Exception as e:
            if not cookies_expired:
                print(f"❌ Error during Twitter reply automation: {e}", file=sys.stderr)
                if "page" in locals():
                    page.screenshot(path="twitter_error_reply.png")
                raise e
        finally:
            if not cookies_expired:
                try:
                    new_cookies = ctx.cookies()
                    with open(COOKIES_PATH, "w") as f:
                        json.dump(new_cookies, f, indent=2)
                except Exception as cookie_err:
                    print(f"⚠️ Failed to save refreshed cookies: {cookie_err}")

            if "browser" in locals():
                browser.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--login":
        interactive_login_session()
    else:
        print("💡 FadeGoblin Twitter Module")
        print("Use '--login' to manually authenticate and capture cookies:")
        print("  python -m fadegoblin.browser_twitter --login")
