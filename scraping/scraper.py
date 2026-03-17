from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException,
    TimeoutException,
)

from scraping.config import (
    EMAIL,
    PWD,
    LOGIN_URL,
    TARGET_URL,
    GLOBAL_START,
    GLOBAL_END,
    WINDOW_DAYS,
    DEFAULT_WAIT,
    POLL,
    SHARD_SIZE,
    WRITE_BATCH_SIZE,
    SHARD_DIR,
    SHARD_PREFIX,
)
from scraping.selectors import (
    DATE_BOX_SEL,
    APPLY_BTN_SEL,
    SEARCH_BTN_SEL,
    EMPTY_SEL,
    DATE_INP_CSS,
    SEARCH_LIST_BOX_SEL,
)
from scraping.storage import ParquetShardWriter
from scraping.koyfin_helpers import (
    wait,
    exists,
    accept_cookies_if_present,
    ensure_earnings_calls_only,
    earnings_calls_is_selected,
    scroll_first_results_list_to_bottom,
    get_bottom_search_row,
    get_search_row_title,
    get_panel_fast_state,
    click_search_row,
    wait_split_view,
    scroll_left_list_to_top,
    get_left_list_box,
    visible_left_items,
    item_key,
    scrape_current_transcript_with_retry,
    make_windows,
    fill_date_range,
    wait_until_transcript_ready,
)


class KoyfinScraper:
    def __init__(self, logger):
        self.logger = logger

        self.writer = ParquetShardWriter(
            shard_dir=SHARD_DIR,
            shard_prefix=SHARD_PREFIX,
            shard_size=SHARD_SIZE,
            write_batch_size=WRITE_BATCH_SIZE,
            logger=logger,
        )

        self.stats = {
            "total_scraped": 0,
            "total_failed": 0,
            "total_stale": 0,
        }

        opts = webdriver.FirefoxOptions()
        self.driver = webdriver.Firefox(options=opts)
        self.driver.maximize_window()
        self.base_wait = wait(self.driver, DEFAULT_WAIT, poll=POLL)

    def login(self):
        self.driver.get(LOGIN_URL)
        accept_cookies_if_present(self.driver)

        self.base_wait.until(EC.element_to_be_clickable((By.NAME, "email"))).send_keys(EMAIL)
        self.base_wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password']"))).send_keys(PWD)
        self.base_wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[@type='submit' or normalize-space()='Sign in']"))
        ).click()
        self.base_wait.until(
            lambda x: "login" not in x.current_url.lower() or not x.find_elements(By.NAME, "email")
        )

    def initialize_transcripts_page(self):
        self.driver.get(TARGET_URL)
        self.base_wait.until(lambda x: "login" not in x.current_url.lower())
        accept_cookies_if_present(self.driver)
        ensure_earnings_calls_only(self.driver, self.base_wait, self.logger)

    def run_one_period(self, start_date, end_date, results, first_run=False):
        self.logger.info(f"period_start | start={start_date} | end={end_date}")

        if not first_run:
            self.driver.get(TARGET_URL)
            self.base_wait.until(lambda x: "login" not in x.current_url.lower())
            accept_cookies_if_present(self.driver)

            if not earnings_calls_is_selected(self.driver):
                self.logger.warning(
                    f"earnings_calls_was_unchecked | start={start_date} | end={end_date}"
                )
                ensure_earnings_calls_only(self.driver, self.base_wait, self.logger)

        self.base_wait.until(EC.element_to_be_clickable(DATE_BOX_SEL)).click()
        self.base_wait.until(
            lambda _: len(
                [i for i in self.driver.find_elements(By.CSS_SELECTOR, DATE_INP_CSS) if i.is_displayed()]
            ) >= 2
        )

        if not fill_date_range(self.driver, start_date, end_date):
            self.logger.warning(f"date_fill_failed | start={start_date} | end={end_date}")
            return results

        self.base_wait.until(EC.element_to_be_clickable(APPLY_BTN_SEL)).click()
        self.base_wait.until(EC.element_to_be_clickable(SEARCH_BTN_SEL)).click()
        self.base_wait.until(
            lambda drv: exists(drv, By.CSS_SELECTOR, SEARCH_LIST_BOX_SEL) or exists(drv, By.XPATH, EMPTY_SEL)
        )

        if exists(self.driver, By.XPATH, EMPTY_SEL):
            self.logger.warning(f"period_no_results | start={start_date} | end={end_date}")
            return results

        scroll_first_results_list_to_bottom(self.driver, self.base_wait, self.logger)

        bottom_row = get_bottom_search_row(self.driver, retries=3)
        if not bottom_row:
            self.logger.warning(f"bottom_row_missing | start={start_date} | end={end_date}")
            return results

        bottom_title = get_search_row_title(bottom_row)
        self.logger.info(f"bottom_row_click | title={bottom_title}")

        prev_title, prev_body = get_panel_fast_state(self.driver)

        click_search_row(self.driver, bottom_row)
        wait_split_view(self.driver)

        try:
            wait_until_transcript_ready(self.driver, prev_title, prev_body)
        except TimeoutException:
            pass

        self.logger.info("split_view_opened")
        scroll_left_list_to_top(self.driver)

        left_box = get_left_list_box(self.driver)
        if left_box is None:
            self.logger.warning("left_list_box_missing")
            return results

        step = 3500
        max_idle = 4
        idle = 0
        prev_scroll = -1

        while idle < max_idle:
            current_items = visible_left_items(self.driver)

            for item in current_items:
                try:
                    key = item_key(item)
                    if not key:
                        continue

                    data = scrape_current_transcript_with_retry(self.driver, item, key)

                    results.append(data)
                    self.stats["total_scraped"] += 1

                    self.logger.info(
                        f"item_scraped | total_scraped={self.stats['total_scraped']} | "
                        f"title={data['title'] or key}"
                    )
                    results = self.writer.save_results(results)

                except StaleElementReferenceException:
                    self.stats["total_stale"] += 1
                    self.logger.warning("item_stale | error_type=StaleElementReferenceException")
                    continue

                except Exception as e:
                    self.stats["total_failed"] += 1
                    self.logger.error(
                        f"item_error | error_type={type(e).__name__} | message={e}"
                    )

            try:
                top, h, ch = self.driver.execute_script(
                    "const e=arguments[0]; return [e.scrollTop, e.scrollHeight, e.clientHeight];",
                    left_box
                )

                if top + ch >= h - 5:
                    self.logger.info("left_list_bottom_reached")
                    break

                self.driver.execute_script("arguments[0].scrollTop += arguments[1];", left_box, step)

                try:
                    wait(self.driver, 0.25).until(
                        lambda drv: drv.execute_script("return arguments[0].scrollTop;", left_box) != top
                    )
                except TimeoutException:
                    pass

                new_scroll = self.driver.execute_script("return arguments[0].scrollTop;", left_box)
                idle = idle + 1 if new_scroll == prev_scroll else 0
                prev_scroll = new_scroll

            except StaleElementReferenceException:
                left_box = get_left_list_box(self.driver)
                if left_box is None:
                    break

        return results

    def run(self):
        self.logger.info(
            f"run_config | global_start={GLOBAL_START} | global_end={GLOBAL_END} | "
            f"window_days={WINDOW_DAYS} | shard_size={SHARD_SIZE} | write_batch_size={WRITE_BATCH_SIZE}"
        )

        try:
            self.login()
            self.initialize_transcripts_page()

            results = []
            first_run = True

            for start_date, end_date in make_windows(GLOBAL_START, GLOBAL_END, WINDOW_DAYS):
                try:
                    results = self.run_one_period(
                        start_date=start_date,
                        end_date=end_date,
                        results=results,
                        first_run=first_run,
                    )
                    first_run = False
                except Exception as e:
                    self.logger.error(
                        f"period_error | start={start_date} | end={end_date} | "
                        f"error_type={type(e).__name__} | message={e}"
                    )
                    first_run = False

            results = self.writer.save_results(results, force=True)

            self.logger.info(
                f"scraping_completed | total_scraped={self.stats['total_scraped']} | "
                f"total_failed={self.stats['total_failed']} | total_stale={self.stats['total_stale']}"
            )

        finally:
            self.driver.quit()