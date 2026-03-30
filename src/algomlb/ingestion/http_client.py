import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from algomlb.core.logger import logger


class BaseAPIClient:
    """Base HTTP client with standardized retry logic and resilient error handling."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """
        Execute an HTTP request with exponential backoff and retry logic.
        Retries on connection errors, timeouts, and 429/50x status codes.
        """
        try:
            response = self.client.request(method, path, **kwargs)
            # Raise for 429 (Rate Limit) and 50x (Server Errors) to trigger tenacity retry
            if response.status_code in (429, 500, 502, 503, 504):
                logger.warning(
                    f"API Error {response.status_code} for {path}. Triggering retry..."
                )
                response.raise_for_status()

            # For other errors (400, 401, 403, 404), we probably shouldn't retry
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            logger.error(f"HTTP Request failed: {exc}")
            raise

    def close(self):
        """Close the underlying httpx client."""
        self.client.close()
