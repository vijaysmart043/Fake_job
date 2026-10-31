
import ipaddress
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; JobSafetyResearchBot/1.0)"
}

TIMEOUT = 12
MAX_HTML_SIZE = 2_000_000


def validate_url(url):
    """Validate URL and block obvious unsafe destinations."""

    parsed = urlparse(url.strip())

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    if not parsed.hostname:
        raise ValueError("Invalid URL.")

    hostname = parsed.hostname.lower()

    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Local URLs are not allowed.")

    try:
        addresses = socket.getaddrinfo(
            hostname, None, type=socket.SOCK_STREAM
        )
    except socket.gaierror:
        raise ValueError("Could not resolve the website address.")

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])

        if not ip.is_global:
            raise ValueError("Private or local network URLs are not allowed.")

    return parsed


def scan_job_url(url):
    """Download a public webpage and extract likely job details."""

    validate_url(url)

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT,
        allow_redirects=False,
        stream=True,
    )

    try:
        if response.status_code in (301, 302, 303, 307, 308):
            raise ValueError(
                "This URL redirects. Please paste the final job-page URL."
            )

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" not in content_type:
            raise ValueError("The URL does not point to an HTML webpage.")

        chunks = []
        total_size = 0

        for chunk in response.iter_content(chunk_size=8192):
            total_size += len(chunk)

            if total_size > MAX_HTML_SIZE:
                raise ValueError("The webpage is too large to scan.")

            chunks.append(chunk)

        html = b"".join(chunks)

    finally:
        response.close()

    soup = BeautifulSoup(html, "html.parser")

    for element in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
    ]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    description_tag = soup.find(
        "meta",
        attrs={"name": "description"},
    )

    description = (
        description_tag.get("content", "")
        if description_tag else ""
    )

    page_text = soup.get_text(" ", strip=True)

    return {
        "title": title,
        "description": description,
        "page_text": page_text[:20000],
        "url": url,
    }
