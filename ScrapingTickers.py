import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    StaleElementReferenceException,
)

from scraping.config import EMAIL, PWD, LOGIN_URL, DEFAULT_WAIT, POLL
from scraping.logger_setup import build_logger


SECURITY_SEARCH_URL = "https://app.koyfin.com/search/security"


def wait(driver, timeout, poll=POLL):
    """Return a WebDriverWait instance."""
    from selenium.webdriver.support.ui import WebDriverWait
    return WebDriverWait(driver, timeout, poll_frequency=poll)


def clean_text(txt: str) -> str:
    """Normalize whitespace."""
    return " ".join((txt or "").split()).strip()


def js_click(driver, el):
    """Click with JS fallback."""
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    try:
        el.click()
    except Exception:
        driver.execute_script("arguments[0].click();", el)


def accept_cookies_if_present(driver):
    """Accept cookies if a banner is shown."""
    xpaths = [
        "//button[contains(., 'Accept All')]",
        "//button[contains(., 'Accept all')]",
        "//button[contains(., 'Accept')]",
    ]
    for xp in xpaths:
        try:
            wait(driver, 0.8).until(EC.element_to_be_clickable((By.XPATH, xp))).click()
            return True
        except Exception:
            pass
    return False


class KoyfinSecurityScraper:
    def __init__(self, logger=None, headless=False):
        """Initialize the scraper."""
        self.logger = logger or build_logger("koyfin_security_scraper")
        self.results = []
        self.global_seen = set()

        opts = webdriver.FirefoxOptions()
        if headless:
            opts.add_argument("-headless")

        self.driver = webdriver.Firefox(options=opts)
        self.driver.maximize_window()
        self.base_wait = wait(self.driver, DEFAULT_WAIT, poll=POLL)

    def login(self):
        """Log in and bypass the plans redirect."""
        self.driver.get(LOGIN_URL)
        accept_cookies_if_present(self.driver)

        self.base_wait.until(
            EC.element_to_be_clickable((By.NAME, "email"))
        ).send_keys(EMAIL)

        self.base_wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password']"))
        ).send_keys(PWD)

        self.base_wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[@type='submit' or normalize-space()='Sign in']")
            )
        ).click()

        self.base_wait.until(
            lambda d: "login" not in d.current_url.lower() or not d.find_elements(By.NAME, "email")
        )

        time.sleep(1.5)

        if "/user-profile/plans" in self.driver.current_url.lower():
            self.logger.info("post_login_redirect_detected | page=plans")
            self.driver.get(SECURITY_SEARCH_URL)

        self.base_wait.until(lambda d: "login" not in d.current_url.lower())

    def open_security_search(self):
        """Open the security search page."""
        self.driver.get(SECURITY_SEARCH_URL)
        self.base_wait.until(lambda d: "login" not in d.current_url.lower())
        accept_cookies_if_present(self.driver)

        self.base_wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "//button[.//*[contains(normalize-space(), 'Search Securities')]]")
            )
        )

    def select_equity_asset_category(self):
        """Select Equities in the asset category dropdown."""
        dropdown_btn = self.base_wait.until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//button[.//span[normalize-space()='Search Everything'] or .//*[normalize-space()='Search Everything']]",
                )
            )
        )
        self.driver.execute_script("arguments[0].click();", dropdown_btn)
        time.sleep(0.4)

        self.base_wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//*[contains(@class, 'dropdown__menu') or contains(@class, 'dropdown-single-selection')]",
                )
            )
        )

        equities_option = self.base_wait.until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//*[contains(@class, 'dropdown__label') and normalize-space()='Equities']"
                    " | "
                    "//*[self::div or self::span or self::label][normalize-space()='Equities']",
                )
            )
        )
        self.driver.execute_script("arguments[0].click();", equities_option)
        time.sleep(0.5)

        self.base_wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//button[.//span[normalize-space()='Equities'] or .//*[normalize-space()='Equities']]",
                )
            )
        )

        self.logger.info("asset_category_selected | value=Equities")

    def click_search_securities(self):
        """Run the search."""
        btn = self.base_wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[.//*[contains(normalize-space(), 'Search Securities')]]")
            )
        )
        js_click(self.driver, btn)

        self.base_wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[class*='table-styles__table__scrollContainer']")
            )
        )
        time.sleep(0.8)

    def get_results_container(self):
        """Return the table scroll container."""
        return self.base_wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div[class*='table-styles__table__scrollContainer']")
            )
        )

    def extract_visible_rows(self):
        """Extract visible rows from the current viewport."""
        rows_data = self.driver.execute_script(
            """
            const container = document.querySelector("div[class*='table-styles__table__scrollContainer']");
            if (!container) return [];

            const rows = Array.from(
                container.querySelectorAll("div[class*='table-styles__table__row']")
            ).filter(r => r.innerText && r.innerText.trim());

            const out = [];

            for (const row of rows) {
                const text = (row.innerText || "").trim();
                if (!text) continue;

                const parts = text.split("\\n").map(x => x.trim()).filter(Boolean);

                const joined = parts.join(" | ").toLowerCase();
                if (
                    joined.includes("ticker") &&
                    joined.includes("country") &&
                    joined.includes("security name")
                ) {
                    continue;
                }

                if (parts.length >= 4) {
                    const ticker = parts[0];
                    const country = parts[1];
                    const company_name = parts[2];

                    if (
                        ticker.toLowerCase() === "ticker" ||
                        country.toLowerCase() === "country" ||
                        company_name.toLowerCase() === "security name"
                    ) {
                        continue;
                    }

                    out.push({
                        ticker: ticker,
                        country: country,
                        company_name: company_name
                    });
                }
            }

            return out;
            """
        )

        cleaned = []
        for item in rows_data:
            ticker = clean_text(item.get("ticker", ""))
            country = clean_text(item.get("country", ""))
            company_name = clean_text(item.get("company_name", ""))

            if ticker and country and company_name:
                cleaned.append(
                    {
                        "ticker": ticker,
                        "country": country,
                        "company_name": company_name,
                    }
                )

        return cleaned

    def scrape_current_page_rows(self):
        """Scroll the page table and collect all rows."""
        container = self.get_results_container()

        page_seen = set()
        stable_passes = 0
        stable_needed = 3
        step = 1200

        while stable_passes < stable_needed:
            before_count = len(page_seen)

            visible_rows = self.extract_visible_rows()
            for item in visible_rows:
                page_seen.add(
                    (item["ticker"], item["country"], item["company_name"])
                )

            try:
                top, h, ch = self.driver.execute_script(
                    "const e=arguments[0]; return [e.scrollTop, e.scrollHeight, e.clientHeight];",
                    container
                )

                near_bottom = top + ch >= h - 20

                if near_bottom:
                    time.sleep(0.25)
                else:
                    self.driver.execute_script(
                        "arguments[0].scrollTop = arguments[0].scrollTop + arguments[1];",
                        container,
                        step
                    )
                    try:
                        self.driver.execute_script(
                            "arguments[0].dispatchEvent(new Event('scroll'));",
                            container
                        )
                    except Exception:
                        pass
                    time.sleep(0.2)

                try:
                    container = self.get_results_container()
                except Exception:
                    pass

            except StaleElementReferenceException:
                container = self.get_results_container()
                time.sleep(0.2)

            after_count = len(page_seen)
            if after_count == before_count:
                stable_passes += 1
            else:
                stable_passes = 0

        return [
            {
                "ticker": ticker,
                "country": country,
                "company_name": company_name,
            }
            for ticker, country, company_name in sorted(page_seen)
        ]

    def get_page_label(self):
        """Read the current pagination label."""
        try:
            el = self.driver.find_element(
                By.XPATH,
                "//*[contains(normalize-space(), 'Page ') and contains(normalize-space(), ' of ')]"
            )
            return clean_text(el.text)
        except Exception:
            return ""

    def is_next_enabled(self):
        """Check if Next is clickable."""
        try:
            buttons = self.driver.find_elements(
                By.XPATH,
                "//button[.//*[contains(normalize-space(), 'Next')]]"
            )

            for btn in buttons:
                try:
                    if not btn.is_displayed():
                        continue

                    disabled_attr = (btn.get_attribute("disabled") or "").lower()
                    data_disabled = (btn.get_attribute("data-disabled") or "").lower()
                    aria_disabled = (btn.get_attribute("aria-disabled") or "").lower()
                    cls = (btn.get_attribute("class") or "").lower()

                    if disabled_attr in {"true", "disabled"}:
                        return False
                    if data_disabled == "true":
                        return False
                    if aria_disabled == "true":
                        return False
                    if "disabled" in cls:
                        return False

                    return True
                except StaleElementReferenceException:
                    continue
        except Exception:
            pass

        return False

    def click_next_page(self):
        """Go to the next pagination page."""
        buttons = self.driver.find_elements(
            By.XPATH,
            "//button[.//*[contains(normalize-space(), 'Next')]]"
        )

        target = None
        for btn in buttons:
            try:
                if btn.is_displayed():
                    target = btn
                    break
            except StaleElementReferenceException:
                continue

        if target is None:
            return False

        old_label = self.get_page_label()
        js_click(self.driver, target)

        try:
            self.base_wait.until(lambda d: self.get_page_label() != old_label)
        except TimeoutException:
            time.sleep(0.8)

        time.sleep(0.5)
        return True

    def save_progress(self, output_path):
        """Save all collected rows."""
        df = pd.DataFrame(self.results)
        df.to_parquet(output_path, index=False)
        return df

    def scrape_all_pages(self, output_path, max_pages=None):
        """Scrape every page and save after each page."""
        page_idx = 1

        while True:
            page_label = self.get_page_label()
            self.logger.info(
                f"security_page_start | page_index={page_idx} | page_label={page_label or 'NA'}"
            )

            page_rows = self.scrape_current_page_rows()

            new_count = 0
            for row in page_rows:
                key = (row["ticker"], row["country"], row["company_name"])
                if key not in self.global_seen:
                    self.global_seen.add(key)
                    self.results.append(row)
                    new_count += 1

            df = self.save_progress(output_path)

            self.logger.info(
                f"security_page_scraped | page_index={page_idx} | page_rows={len(page_rows)} | "
                f"new_rows={new_count} | total_rows={len(self.results)} | saved_rows={len(df)}"
            )

            if max_pages is not None and page_idx >= max_pages:
                self.logger.info("security_pagination_stop | reason=max_pages_reached")
                break

            if not self.is_next_enabled():
                self.logger.info("security_pagination_stop | reason=next_disabled")
                break

            if not self.click_next_page():
                self.logger.info("security_pagination_stop | reason=next_click_failed")
                break

            page_idx += 1

        return pd.DataFrame(self.results)

    def run(self, max_pages=None, output_path="security_equities.parquet"):
        """Run the full scraping flow."""
        output_path = str(Path(output_path))

        try:
            self.login()
            self.open_security_search()
            self.select_equity_asset_category()
            self.click_search_securities()

            df = self.scrape_all_pages(output_path=output_path, max_pages=max_pages)

            self.logger.info(
                f"security_scraping_completed | rows={len(df)} | output={output_path}"
            )
            return df

        finally:
            self.driver.quit()


if __name__ == "__main__":
    logger = build_logger("koyfin_security_scraper")
    scraper = KoyfinSecurityScraper(logger=logger, headless=False)
    scraper.run(max_pages=None, output_path="security_equities.parquet")