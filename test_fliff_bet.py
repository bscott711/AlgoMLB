"""Script to inspect elements around Diamondbacks."""
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

    print("Looking for Diamondbacks...")
    try:
        team_el = page.get_by_text("Diamondbacks", exact=False).first
        if team_el.is_visible(timeout=3000):
            print("✅ Found Diamondbacks")
            
            # Find the position of the text
            box = team_el.bounding_box()
            print(f"Diamondbacks box: {box}")
            
            # Use JS to find all elements near this y-coordinate
            els = page.evaluate('''([tx, ty]) => {
                const results = [];
                document.querySelectorAll('*').forEach(el => {
                    const r = el.getBoundingClientRect();
                    // if it's within 150px vertically and is a reasonable size
                    if (Math.abs(r.y - ty) < 150 && r.width > 20 && r.height > 20 && r.height < 100) {
                        results.push({
                            tag: el.tagName,
                            cls: (el.className || '').toString().slice(0, 60),
                            text: (el.textContent || '').trim().slice(0, 30),
                            x: Math.round(r.x), y: Math.round(r.y), 
                            w: Math.round(r.width), h: Math.round(r.height)
                        });
                    }
                });
                return results;
            }''', [box['x'], box['y']])
            
            # Print unique elements
            seen = set()
            for el in els:
                key = f"{el['tag']}-{el['cls']}-{el['text']}"
                if key not in seen:
                    print(f"<{el['tag']}> {el['w']}x{el['h']} at ({el['x']},{el['y']}) cls='{el['cls']}' text='{el['text']}'")
                    seen.add(key)
                    
            # Let's click the first thing that looks like an odds button next to it
            # In Fliff, odds are usually div classes like `odds-value` or `selection`
            # We'll just click at offset
            print("Clicking right side of the row...")
            page.mouse.click(box['x'] + 200, box['y'] + 10)
            page.wait_for_timeout(2000)
            page.screenshot(path="fliff_bet_test.png")
            
            # Check betslip
            print("Dumping all inputs...")
            inputs = page.locator("input").all()
            for i, input_el in enumerate(inputs):
                if input_el.is_visible():
                    print(f"Input {i}: type={input_el.get_attribute('type')} placeholder={input_el.get_attribute('placeholder')}")
            
    except Exception as e:
        print(f"Error: {e}")

    browser.close()
