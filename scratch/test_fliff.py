"""Script to inspect the DOM of a settled slip."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from fadegoblin.browser_fliff import _create_fliff_context, _setup_radar_bypass, _dismiss_modals

STATE_PATH = Path("/home/opc/AlgoMLB/fliff_state.json")

with sync_playwright() as p:
    browser, ctx = _create_fliff_context(p)
    page = ctx.new_page()
    _setup_radar_bypass(page)
    
    page.goto("https://sports.getfliff.com/my-picks")
    page.wait_for_timeout(8000)
    
    # Toggle to Cash
    try:
        switcher = page.locator("div.switcher").first
        if switcher.is_visible(timeout=3000):
            switcher.click()
            page.wait_for_timeout(2000)
    except Exception:
        pass

    _dismiss_modals(page)

    # Click Settled
    try:
        settled_tab = page.get_by_text("Settled", exact=True).first
        if settled_tab.is_visible(timeout=3000):
            settled_tab.click()
            page.wait_for_timeout(3000)
    except Exception:
        pass

    # Find 'Guardians'
    pick_el = page.get_by_text("Guardians", exact=False).first
    if pick_el.is_visible(timeout=2000):
        # We want to find the top-most container of the slip.
        # Let's inspect its ancestors.
        els = page.evaluate('''() => {
            const results = [];
            const textNode = document.evaluate("//*[contains(text(), 'Guardians')]", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            if (textNode) {
                let curr = textNode;
                let depth = 0;
                while (curr && curr.tagName !== 'BODY' && depth < 10) {
                    const rect = curr.getBoundingClientRect();
                    results.push({
                        tag: curr.tagName,
                        cls: (curr.className || "").toString().slice(0, 80),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    });
                    curr = curr.parentNode;
                    depth++;
                }
            }
            return results;
        }''')
        for el in els:
            print(f"<{el['tag']}> cls='{el['cls']}' {el['width']}x{el['height']}")
    
    browser.close()
