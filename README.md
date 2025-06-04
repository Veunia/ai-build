# Website Screenshot Automation

This project provides a Python 3.10 application that automatically captures full page screenshots for every URL available in a website's sitemap or discovered by crawling the site.

## Installation

1. Clone the repository and move into the project directory.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Copy the `.env.example` file to `.env` and fill in the variables:

```
cp .env.example .env
```

- `TARGET_URL` – base URL of the website (e.g., `https://example.com`).
- `USERNAME` and `PASSWORD` – optional credentials if the site is protected.
- `MAX_PAGES` – number of pages to crawl when no sitemap is available.

## Usage

Run the application with Python:

```bash
python src/main.py
```

Screenshots are saved in the `captures/` directory under a tree organised by year, month and day. Log messages are written to `logs/app.log`.

## Requirements

- Python 3.10+
- Google Chrome browser (for ChromeDriver)


