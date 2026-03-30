import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from algomlb.core.logger import logger


class CircuitBreaker:
    """A simple token-bucket style circuit breaker to protect external API resources."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"

    def is_available(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entered HALF_OPEN state.")
                return True
            return False
        return True

    def record_success(self):
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            logger.info("Circuit breaker recovered to CLOSED state.")

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"Circuit breaker OPENED after {self.failure_count} failures.")


class BaseAPIClient:
    """Base HTTP client with standardized retry logic and resilient error handling."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=timeout)
        self._circuit_breaker = CircuitBreaker()

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
        if not self._circuit_breaker.is_available():
            raise RuntimeError(f"Circuit breaker is OPEN for {self.base_url}")

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

            self._circuit_breaker.record_success()
            return response
        except (
            httpx.HTTPStatusError,
            httpx.TimeoutException,
            httpx.ConnectError,
        ) as exc:
            self._circuit_breaker.record_failure()
            logger.error(f"HTTP Request failed: {exc}")
            raise

    def close(self):
        """Close the underlying httpx client."""
        self.client.close()
