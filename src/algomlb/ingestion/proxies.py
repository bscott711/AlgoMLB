import requests
from loguru import logger


def fetch_free_proxies() -> list[str]:
    """Fetch a list of alive HTTP proxies from ProxyScrape API."""
    url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        proxies = [p.strip() for p in response.text.splitlines() if p.strip()]

        # Format as http URLs if they aren't already
        formatted = []
        for p in proxies:
            if not p.startswith("http"):
                formatted.append(f"http://{p}")
            else:
                formatted.append(p)

        logger.info(f"Successfully fetched {len(formatted)} free proxies.")
        return formatted
    except Exception as e:
        logger.error(f"Failed to fetch free proxies: {e}")
        return []
