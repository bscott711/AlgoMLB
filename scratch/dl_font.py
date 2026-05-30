import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating...")
        try:
            async with page.expect_download(timeout=10000) as download_info:
                await page.goto("https://www.fontsaddict.com/font/download/freight-train-gangsta.ttf")
            download = await download_info.value
            await download.save_as("fadegoblin_playwright/src/fadegoblin/assets/freight-train-gangsta.ttf")
            print("Done")
        except Exception as e:
            print(f"Failed: {e}")
            # Try elfont block
            async with page.expect_download(timeout=10000) as download_info:
                await page.goto("https://www.fontsaddict.com/font/download/elfont-block.ttf")
            download = await download_info.value
            await download.save_as("fadegoblin_playwright/src/fadegoblin/assets/elfont-block.ttf")
            print("Done elfont")
        await browser.close()

asyncio.run(run())
