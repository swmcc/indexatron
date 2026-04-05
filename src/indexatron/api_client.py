"""HTTP client for the-mcculloughs.org API."""

import time
from pathlib import Path
from typing import Any

import httpx

from .config import get_settings
from .logging import (
    debug,
    debug_request,
    debug_response,
    error,
    failure,
    info,
)


class ApiError(Exception):
    """API request failed."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class McculloghsClient:
    """HTTP client for the-mcculloughs.org photo API."""

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.api_base_url.rstrip("/")
        self.api_key = settings.api_key
        self.timeout = settings.api_timeout
        self.download_dir = settings.download_dir

        self._client = httpx.Client(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._client.close()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def fetch_pending_uploads(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Fetch uploads that need analysis.

        Args:
            limit: Maximum number of uploads to fetch (uses config batch_size if None)

        Returns:
            List of upload dicts with id, short_code, image_url, created_at
        """
        settings = get_settings()
        per_page = limit or settings.batch_size

        url = f"{self.base_url}/api/uploads/pending"
        params = {"per_page": per_page}

        debug_request("GET", url, params=params)
        start = time.time()

        try:
            response = self._client.get(url, params=params)
            elapsed = time.time() - start
            debug_response(response.status_code, elapsed=elapsed)

            if response.status_code == 401:
                raise ApiError("Unauthorized - check your API key", 401)

            response.raise_for_status()
            data = response.json()

            uploads = data.get("uploads", [])
            total = data.get("total", len(uploads))

            info(f"Fetched {len(uploads)} pending uploads (total: {total})")
            return uploads

        except httpx.HTTPStatusError as e:
            error(f"API error: {e.response.status_code}")
            raise ApiError(str(e), e.response.status_code) from e
        except httpx.RequestError as e:
            error(f"Request failed: {e}")
            raise ApiError(str(e)) from e

    def download_image(self, upload: dict[str, Any]) -> Path:
        """Download an image to the temp directory.

        Args:
            upload: Upload dict with image_url and short_code

        Returns:
            Path to downloaded file
        """
        image_url = upload["image_url"]
        short_code = upload["short_code"]

        # Determine file extension from URL or default to .jpg
        ext = ".jpg"
        if "." in image_url.split("/")[-1]:
            url_ext = "." + image_url.split(".")[-1].split("?")[0]
            if url_ext in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                ext = url_ext

        output_path = self.download_dir / f"{short_code}{ext}"

        debug(f"Downloading {image_url} to {output_path}")
        debug_request("GET", image_url)
        start = time.time()

        try:
            # Use stream for large files
            with self._client.stream("GET", image_url) as response:
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)

            elapsed = time.time() - start
            size_mb = output_path.stat().st_size / (1024 * 1024)
            debug_response(200, body=f"{size_mb:.2f}MB downloaded", elapsed=elapsed)

            return output_path

        except httpx.HTTPStatusError as e:
            error(f"Download failed: {e.response.status_code}")
            raise ApiError(f"Failed to download image: {e}", e.response.status_code) from e
        except httpx.RequestError as e:
            error(f"Download request failed: {e}")
            raise ApiError(f"Failed to download image: {e}") from e

    def post_analysis(
        self,
        upload_id: int,
        analysis_data: dict[str, Any],
        embedding: list[float],
    ) -> bool:
        """Post analysis results back to the API.

        Args:
            upload_id: The upload ID
            analysis_data: Analysis data dict
            embedding: 768-dimensional embedding vector

        Returns:
            True if successful
        """
        url = f"{self.base_url}/api/uploads/{upload_id}/analysis"
        payload = {
            "analysis_data": analysis_data,
            "embedding": embedding,
        }

        debug_request("PATCH", url, body=payload)
        start = time.time()

        try:
            response = self._client.patch(url, json=payload)
            elapsed = time.time() - start
            debug_response(response.status_code, body=response.json(), elapsed=elapsed)

            if response.status_code == 401:
                raise ApiError("Unauthorized - check your API key", 401)

            response.raise_for_status()
            data = response.json()

            if data.get("success"):
                debug(f"Analysis posted for upload {upload_id}")
                return True
            else:
                errors = data.get("errors", ["Unknown error"])
                failure(f"Failed to post analysis: {errors}")
                return False

        except httpx.HTTPStatusError as e:
            error(f"API error: {e.response.status_code}")
            raise ApiError(str(e), e.response.status_code) from e
        except httpx.RequestError as e:
            error(f"Request failed: {e}")
            raise ApiError(str(e)) from e

    def test_connection(self) -> bool:
        """Test the API connection.

        Returns:
            True if connection is successful
        """
        try:
            # Try to fetch 1 upload to verify auth works
            self.fetch_pending_uploads(limit=1)
            return True
        except ApiError:
            return False
