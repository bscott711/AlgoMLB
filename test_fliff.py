"""Minimal: find the cash toggle element."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

STATE_PATH = Path("/home/opc/AlgoMLB/fliff_state.json")
FAKE_RADAR = json.dumps({"meta":{"code":200},"location":{"type":"Point","coordinates":[-79.9959,40.4406]},"user":{"_id":"x","userId":"2037443","deviceId":"x","location":{"type":"Point","coordinates":[-79.9959,40.4406]},"locationAuthorization":"GRANTED_FOREGROUND","country":{"code":"US","name":"United States"},"state":{"code":"PA","name":"Pennsylvania"},"fraud":{"passed":True,"bypassed":False,"verified":True,"proxy":False,"mocked":False,"compromised":False,"jumped":False,"sharing":False},"insights":{"state":{"home":True}}}})

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    ctx = browser.new_context(
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
        viewport={"width": 390, "height": 844}, locale="en-US",
        geolocation={"latitude": 40.4406, "longitude": -79.9959},
        permissions=["geolocation"], storage_state=STATE_PATH,
    )
    page = ctx.new_page()
    page.route("https://api-verified.radar.io/v1/track", lambda r: r.fulfill(status=200, content_type="application/json; charset=utf-8", body=FAKE_RADAR))
    page.route("https://api.radar.io/v1/track", lambda r: r.fulfill(status=200, content_type="application/json; charset=utf-8", body=FAKE_RADAR))

    page.goto("https://sports.getfliff.com/my-picks")
    page.wait_for_timeout(8000)

    # Use pure JS to find elements — no Playwright locators that might timeout
    els = page.evaluate('''() => {
        const results = [];
        document.querySelectorAll("*").forEach(el => {
            const r = el.getBoundingClientRect();
            if (r.top < 60 && r.height > 5 && r.width > 5 && r.height < 200) {
                results.push({
                    tag: el.tagName, cls: (el.className || "").toString().slice(0,80),
                    text: (el.textContent || "").trim().slice(0,40),
                    x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)
                });
            }
        });
        return results;
    }''')
    for el in els:
        print(f'<{el["tag"]}> x={el["x"]} y={el["y"]} {el["w"]}x{el["h"]} cls="{el["cls"]}" text="{el["text"]}"')

    browser.close()
