"""Website screenshot automation tool.

This script reads all URLs from a sitemap or crawls the site and generates
full page screenshots using Selenium ChromeDriver.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


load_dotenv()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

AUTH = None
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
if USERNAME and PASSWORD:
    AUTH = (USERNAME, PASSWORD)

BASE_URL = os.getenv("TARGET_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not BASE_URL:
    logging.error("TARGET_URL not provided")
    print("Please set TARGET_URL in .env or pass as command line argument.")
    sys.exit(1)

MAX_PAGES = int(os.getenv("MAX_PAGES", "50"))


def read_sitemap(base_url: str) -> List[str]:
    """Return URLs found in sitemap.xml if available."""
    sitemap_url = urljoin(base_url, "/sitemap.xml")
    try:
        response = requests.get(sitemap_url, auth=AUTH, timeout=10)
        response.raise_for_status()
    except Exception as exc:  # pylint: disable=broad-except
        logging.info("Sitemap not found: %s", exc)
        return []
    urls: List[str] = []
    try:
        from xml.etree import ElementTree as ET

        root = ET.fromstring(response.content)
        for loc in root.iter():
            if loc.tag.endswith("loc") and loc.text:
                urls.append(loc.text.strip())
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Failed to parse sitemap: %s", exc)
        return []
    return urls


def crawl_site(base_url: str, max_pages: int) -> List[str]:
    """Crawl the site to collect URLs."""
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    queue = [base_url]
    visited = set()
    urls = []
    while queue and len(visited) < max_pages:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        try:
            res = requests.get(current, auth=AUTH, timeout=10)
            res.raise_for_status()
        except Exception as exc:  # pylint: disable=broad-except
            logging.error("Error crawling %s: %s", current, exc)
            continue
        urls.append(current)
        soup = BeautifulSoup(res.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(current, a["href"])
            parsed = urlparse(href)
            if parsed.netloc == base_domain:
                clean_url = href.split("#")[0]
                if clean_url not in visited and clean_url not in queue:
                    queue.append(clean_url)
    return urls


def create_driver() -> webdriver.Chrome:
    """Create and return a headless Chrome driver."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def sanitize_filename(url: str) -> str:
    """Create a file name from the URL."""
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "root"
    path = path.replace("/", "_")
    domain = parsed.netloc.replace(":", "_")
    date_str = datetime.now().strftime("%Y%m%d")
    return f"date_{date_str}_{domain}_{path}.png"


def save_screenshot(url: str) -> None:
    """Capture screenshot of a single URL."""
    driver = create_driver()
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(5)
        width = driver.execute_script("return document.body.scrollWidth")
        height = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(width, height)
        now = datetime.now()
        dir_path = Path("captures") / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d")
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / sanitize_filename(url)
        driver.save_screenshot(str(file_path))
        logging.info("Captured %s", url)
    except Exception as exc:  # pylint: disable=broad-except
        logging.error("Failed to capture %s: %s", url, exc)
    finally:
        driver.quit()


def capture_all(urls: Iterable[str]) -> None:
    """Capture screenshots for all URLs using a thread pool."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(save_screenshot, urls)


def main() -> None:
    """Entry point of the application."""
    urls = read_sitemap(BASE_URL)
    if not urls:
        logging.info("No sitemap found. Starting crawler...")
        urls = crawl_site(BASE_URL, MAX_PAGES)
    logging.info("Found %d URLs", len(urls))
    capture_all(urls)


if __name__ == "__main__":
    main()
