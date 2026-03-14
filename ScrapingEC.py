import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException,
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
)


# CONFIG
EMAIL = "ftd2026_user.fjifdidf@outlook.com"
PWD   = "ftd2026_pwd"

LOGIN_URL  = "https://app.koyfin.com/login?prevUrl=%2Fsearch%2Ftranscripts"
TARGET_URL = "https://app.koyfin.com/search/transcripts"

# global period to cover
GLOBAL_START = "02/16/2026"
GLOBAL_END   = "02/28/2026"

# scraping block = 2 days
WINDOW_DAYS = 2

DEFAULT_WAIT = 6
CLICK_WAIT = 1.5
READY_WAIT = 1.2
POLL = 0.01
RETRY_COUNT = 1

PLACEHOLDER_TEXTS = [
    "your document is on its way...",
    "your document is on its way",
]


# LOGGER
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "koyfin_scraping.log"


class ConsoleOnlyImportantFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return (
            "run_config" in msg
            or "period_start" in msg
            or "batch_flush" in msg
            or "shard_completed" in msg
            or "scraping_completed" in msg
            or record.levelno >= logging.ERROR
        )


logger = logging.getLogger("koyfin_scraper")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
stream_handler.addFilter(ConsoleOnlyImportantFilter())

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# PARQUET SHARDING + BATCH WRITING
# logical size of one parquet shard
SHARD_SIZE = 200

# in-memory buffer size before writing
WRITE_BATCH_SIZE = 20

SHARD_DIR = Path("data_raw")
SHARD_DIR.mkdir(parents=True, exist_ok=True)

SHARD_PREFIX = "koyfin_transcripts"
shard_id = 0

current_writer = None
current_schema = None
current_shard_rows = 0
current_shard_path = None

total_scraped = 0
total_failed = 0
total_stale = 0
batch_id = 0

logger.info(
    f"run_config | global_start={GLOBAL_START} | global_end={GLOBAL_END} | "
    f"window_days={WINDOW_DAYS} | shard_size={SHARD_SIZE} | write_batch_size={WRITE_BATCH_SIZE}"
)

# DRIVER
opts = webdriver.FirefoxOptions()
d = webdriver.Firefox(options=opts)
d.maximize_window()
w = WebDriverWait(d, DEFAULT_WAIT, poll_frequency=POLL)


# SELECTORS
DATE_BOX_SEL   = (By.CSS_SELECTOR, "div[class*='time-range__root']")
APPLY_BTN_SEL  = (By.XPATH, "//button[.//label[normalize-space()='Apply Dates'] or normalize-space()='Apply Dates']")
DATE_INP_CSS   = "input[placeholder='MM/DD/YYYY']"
SEARCH_BTN_SEL = (By.XPATH, "//button[contains(., 'Search Transcripts')]")
EMPTY_SEL      = "//*[contains(., 'was not found anywhere') or contains(., 'searched all available transcripts')]"

SEARCH_LIST_BOX_SEL = "div[class*='news-virtual-list__newsVirtualList__container']"
SEARCH_ROWS_SEL     = "div[class*='news-virtual-list__newsVirtualList__items'] > div[style*='position: absolute']"
SEARCH_TITLE_SEL    = "label[class*='text-label']"
SEARCH_ITEM_SEL     = "div[class*='koy-news-item__koyNewsItem']"

LEFT_ITEMS_SEL   = "div[class*='koy-news-item__koyNewsItem']"
RIGHT_PANEL_SEL  = "div[class*='news-article-panel__newsArticlePanel__root']"

ARTICLE_ROOT_SEL                  = "div[class*='common-news-article-content__commonNewsArticleContent__root']"
ARTICLE_TITLE_SEL                 = "div[class*='article-title__articleTitle__heading']"
ARTICLE_SUBHEADER_SEL             = "div[class*='article-title__articleTitle__subHeading']"
ARTICLE_TRANSCRIPT_SUBHEADER_SEL  = "div[class*='article-title__articleTitle__transcriptSubHeader']"
ARTICLE_SPEAKERS_SEL              = "div[class*='transcript-speakers__transcriptSpeakers__content']"
ARTICLE_BODY_SEL                  = "div[class*='slate-editor__container']"

EARNINGS_CALLS_FILTER_SEL = (
    By.XPATH,
    "//div[@role='button' and .//label[normalize-space()='Earnings Calls']]"
)



# HELPERS
def wait(timeout, poll=POLL):
    return WebDriverWait(d, timeout, poll_frequency=poll)

def exists(by, sel):
    return len(d.find_elements(by, sel)) > 0

def clean_text(txt):
    txt = (txt or "").replace("\u00a0", " ")
    txt = re.sub(r"\r", "", txt)

    txt = re.sub(r"[ \t]+\n", "\n", txt)
    txt = re.sub(r"\n[ \t]+", "\n", txt)

    txt = re.sub(r"[ \t]+", " ", txt)

    txt = re.sub(r"\n{2,}", "\n", txt)

    return txt.strip()

def is_placeholder_text(txt):
    t = clean_text(txt).lower()
    return any(p in t for p in PLACEHOLDER_TEXTS)

def accept_cookies_if_present():
    for xp in [
        "//button[contains(., 'Accept All')]",
        "//button[contains(., 'Accept all')]",
        "//button[contains(., 'Accept')]",
    ]:
        try:
            wait(0.7).until(EC.element_to_be_clickable((By.XPATH, xp))).click()
            return True
        except Exception:
            pass
    return False

def set_val(el, val):
    d.execute_script(
        "const el=arguments[0],v=arguments[1];"
        "el.focus();"
        "Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,v);"
        "el.dispatchEvent(new Event('input',{bubbles:true}));"
        "el.dispatchEvent(new Event('change',{bubbles:true}));"
        "el.blur();",
        el, val
    )

def js_click(el):
    d.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    try:
        el.click()
    except Exception:
        d.execute_script("arguments[0].click();", el)

def is_filter_selected(el):
    return d.execute_script(
        """
        const node = arguments[0];

        const attrs = [
            "aria-pressed",
            "aria-selected",
            "data-selected",
            "data-active",
            "data-checked",
            "aria-checked"
        ];

        for (const a of attrs) {
            const v = node.getAttribute(a);
            if (v === "true") return true;
        }

        const cls = (node.className || "").toLowerCase();
        if (
            cls.includes("selected") ||
            cls.includes("active") ||
            cls.includes("checked")
        ) {
            return true;
        }

        return false;
        """,
        el
    )

def ensure_earnings_calls_only():
    btn = w.until(EC.presence_of_element_located(EARNINGS_CALLS_FILTER_SEL))
    d.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)

    if not is_filter_selected(btn):
        js_click(btn)
        try:
            wait(2).until(lambda drv: is_filter_selected(btn))
        except TimeoutException:
            pass

def row_top_from_style(row):
    try:
        style = row.get_attribute("style") or ""
        m = re.search(r"top:\s*([0-9.]+)px", style)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return -1.0

def get_real_search_rows():
    rows = d.find_elements(By.CSS_SELECTOR, SEARCH_ROWS_SEL)
    out = []
    for row in rows:
        try:
            if row.is_displayed() and row.text.strip():
                out.append(row)
        except Exception:
            pass
    return out

def get_bottom_search_row(retries=3):
    for _ in range(retries):
        rows = get_real_search_rows()
        if rows:
            rows = sorted(rows, key=row_top_from_style)
            return rows[-1]
    return None

def get_search_row_title(row):
    try:
        return clean_text(row.text).split("\n")[0]
    except Exception:
        return ""

def click_search_row(row):
    try:
        target = row.find_element(By.CSS_SELECTOR, SEARCH_ITEM_SEL)
    except Exception:
        target = row
    d.execute_script("arguments[0].scrollIntoView({block:'center'});", target)
    try:
        target.click()
    except Exception:
        d.execute_script("arguments[0].click();", target)

def scroll_first_results_list_to_bottom():
    box = w.until(lambda drv: drv.find_element(By.CSS_SELECTOR, SEARCH_LIST_BOX_SEL))
    step = 1000
    pause = 1
    max_idle = 10
    idle = 0
    prev_top = -1

    while idle < max_idle:
        try:
            top, h, ch = d.execute_script(
                "const e=arguments[0]; return [e.scrollTop, e.scrollHeight, e.clientHeight];",
                box
            )

            if top + ch >= h - 5:
                logger.info("search_list_bottom_reached")
                return

            d.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollTop + arguments[1];",
                box, step
            )
            time.sleep(pause)

            new_top = d.execute_script("return arguments[0].scrollTop;", box)
            if new_top == prev_top:
                idle += 1
            else:
                idle = 0
            prev_top = new_top

        except StaleElementReferenceException:
            time.sleep(0.8)
            box = w.until(lambda drv: drv.find_element(By.CSS_SELECTOR, SEARCH_LIST_BOX_SEL))

def wait_split_view():
    wait(2).until(lambda drv: len(drv.find_elements(By.CSS_SELECTOR, LEFT_ITEMS_SEL)) > 0)
    wait(2).until(lambda drv: len(drv.find_elements(By.CSS_SELECTOR, RIGHT_PANEL_SEL)) > 0)

def get_left_list_box():
    candidates = d.find_elements(By.CSS_SELECTOR, "div[class*='box__box__'][data-fill='true']")
    visible = []
    for el in candidates:
        try:
            if el.is_displayed():
                visible.append(el)
        except Exception:
            pass

    if visible:
        visible.sort(key=lambda x: d.execute_script("return arguments[0].getBoundingClientRect().left;", x))
        return visible[0]

    items = d.find_elements(By.CSS_SELECTOR, LEFT_ITEMS_SEL)
    if items:
        return items[0]

    return None

def scroll_left_list_to_top():
    box = get_left_list_box()
    if box is None:
        return
    try:
        d.execute_script("arguments[0].scrollTop = 0;", box)
    except Exception:
        pass

def visible_left_items():
    items = d.find_elements(By.CSS_SELECTOR, LEFT_ITEMS_SEL)
    out = []
    for it in items:
        try:
            txt = clean_text(it.text)
            if it.is_displayed() and txt:
                out.append(it)
        except Exception:
            pass

    out.sort(key=lambda el: d.execute_script("return arguments[0].getBoundingClientRect().top;", el))
    return out

def item_key(el):
    try:
        return clean_text(el.text)
    except Exception:
        return ""

def click_left_item(el):
    d.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    try:
        el.click()
    except (ElementClickInterceptedException, StaleElementReferenceException):
        d.execute_script("arguments[0].click();", el)

def get_article_root():
    roots = d.find_elements(By.CSS_SELECTOR, ARTICLE_ROOT_SEL)
    for r in roots:
        try:
            if r.is_displayed():
                return r
        except Exception:
            pass

    panels = d.find_elements(By.CSS_SELECTOR, RIGHT_PANEL_SEL)
    for p in panels:
        try:
            if p.is_displayed():
                return p
        except Exception:
            pass

    return None

def get_panel_fast_state():
    root = get_article_root()
    if root is None:
        return "", ""

    title, body = d.execute_script(
        """
        const root = arguments[0];

        const q = (sel) => {
            const el = root.querySelector(sel);
            return el ? (el.innerText || el.textContent || "").trim() : "";
        };

        const title = q("div[class*='article-title__articleTitle__heading']");
        const body  = q("div[class*='slate-editor__container']");

        return [title, body];
        """,
        root
    )

    return clean_text(title), clean_text(body)

def wait_until_transcript_ready(prev_title="", prev_body="", timeout=READY_WAIT):
    def _cond(_):
        title, body = get_panel_fast_state()

        if not body:
            return False

        if is_placeholder_text(body):
            return False

        if body and body != prev_body:
            return True

        if title and title != prev_title:
            return True

        if len(body) > 80:
            return True

        return False

    wait(timeout).until(_cond)

def scrape_current_transcript_fast():
    root = get_article_root()
    if root is None:
        return {
            "title": "",
            "subheader": "",
            "transcript_subheader": "",
            "speakers": "",
            "body": "",
        }

    title, subheader, transcript_subheader, speakers, body = d.execute_script(
        """
        const root = arguments[0];

        const getOne = (sel) => {
            const el = root.querySelector(sel);
            return el ? (el.innerText || el.textContent || "").trim() : "";
        };

        const getAll = (sel) => {
            return Array.from(root.querySelectorAll(sel))
                .map(el => (el.innerText || el.textContent || "").trim())
                .filter(Boolean)
                .filter((v, i, a) => a.indexOf(v) === i)
                .join("\\n");
        };

        const title = getOne("div[class*='article-title__articleTitle__heading']");
        const subheader = getOne("div[class*='article-title__articleTitle__subHeading']");
        const transcript_subheader = getOne("div[class*='article-title__articleTitle__transcriptSubHeader']");
        const speakers = getAll("div[class*='transcript-speakers__transcriptSpeakers__content']");
        let body = getAll("div[class*='slate-editor__container']");

        return [title, subheader, transcript_subheader, speakers, body];
        """,
        root
    )

    return {
        "title": clean_text(title),
        "subheader": clean_text(subheader),
        "transcript_subheader": clean_text(transcript_subheader),
        "speakers": clean_text(speakers),
        "body": clean_text(body),
    }

def scrape_current_transcript_with_retry(item, key, retries=RETRY_COUNT):
    last_data = None

    for _ in range(retries):
        try:
            prev_title, prev_body = get_panel_fast_state()

            click_left_item(item)
            wait_until_transcript_ready(prev_title, prev_body, timeout=READY_WAIT)

            data = scrape_current_transcript_fast()
            data["list_item"] = key
            last_data = data

            if not is_placeholder_text(data["body"]):
                return data

        except TimeoutException:
            pass
        except StaleElementReferenceException:
            pass

    return last_data or {
        "title": "",
        "subheader": "",
        "transcript_subheader": "",
        "speakers": "",
        "body": "",
        "list_item": key,
    }


# PARQUET BATCH WRITER
def _open_new_writer(schema: pa.Schema):
    global shard_id, current_writer, current_schema, current_shard_rows, current_shard_path

    current_shard_path = SHARD_DIR / f"{SHARD_PREFIX}_{shard_id:05d}.parquet"
    current_writer = pq.ParquetWriter(
        current_shard_path,
        schema=schema,
        compression="snappy"
    )
    current_schema = schema
    current_shard_rows = 0

def _close_current_writer():
    global shard_id, current_writer, current_schema, current_shard_rows, current_shard_path

    if current_writer is not None:
        current_writer.close()
        logger.info(
            f"shard_completed | shard_id={shard_id:05d} | "
            f"rows={current_shard_rows}/{SHARD_SIZE} | path={current_shard_path}"
        )
        current_writer = None
        current_schema = None
        current_shard_rows = 0
        current_shard_path = None
        shard_id += 1

def save_results(results, force=False):
    global current_writer, current_schema, current_shard_rows, batch_id

    if not force and len(results) < WRITE_BATCH_SIZE:
        return results

    if not results:
        if force:
            _close_current_writer()
        return []

    pending = results

    while pending:
        if current_writer is None:
            room = SHARD_SIZE
        else:
            room = SHARD_SIZE - current_shard_rows

        if room <= 0:
            _close_current_writer()
            room = SHARD_SIZE

        chunk = pending[:room]
        pending = pending[room:]

        table = pa.Table.from_pylist(chunk)

        if current_writer is None:
            _open_new_writer(table.schema)
        else:
            if table.schema != current_schema:
                table = pa.Table.from_pylist(chunk, schema=current_schema)

        batch_id += 1
        logger.info(
            f"batch_flush | batch_id={batch_id} | shard_id={shard_id:05d} | "
            f"rows={len(chunk)} | shard_rows={current_shard_rows + len(chunk)}/{SHARD_SIZE}"
        )

        current_writer.write_table(table)
        current_shard_rows += len(chunk)

        if current_shard_rows >= SHARD_SIZE:
            _close_current_writer()

    if force:
        _close_current_writer()

    return []

def make_windows(start_str, end_str, window_days=2):
    start_dt = datetime.strptime(start_str, "%m/%d/%Y")
    end_dt = datetime.strptime(end_str, "%m/%d/%Y")

    cur = start_dt
    while cur <= end_dt:
        win_end = min(cur + timedelta(days=window_days - 1), end_dt)
        yield cur.strftime("%m/%d/%Y"), win_end.strftime("%m/%d/%Y")
        cur += timedelta(days=window_days)

def run_one_period(start_date, end_date, results, first_run=False):
    global total_scraped, total_failed, total_stale

    logger.info(f"period_start | start={start_date} | end={end_date}")


    # SEARCH PAGE
    if not first_run:
        d.get(TARGET_URL)
        w.until(lambda x: "login" not in x.current_url.lower())
        accept_cookies_if_present()

    # IMPORTANT:
    # on the first pass, stay on the already open page
    # and already filtered on Earnings Calls.
    # On the following passes, reload the page but do not click the filter again.

    w.until(EC.element_to_be_clickable(DATE_BOX_SEL)).click()
    w.until(lambda _: len([i for i in d.find_elements(By.CSS_SELECTOR, DATE_INP_CSS) if i.is_displayed()]) >= 2)

    for _ in range(2):
        try:
            a, b = [i for i in d.find_elements(By.CSS_SELECTOR, DATE_INP_CSS) if i.is_displayed()][:2]
            set_val(a, start_date)
            set_val(b, end_date)
            b.send_keys(Keys.TAB)
            break
        except StaleElementReferenceException:
            continue

    w.until(EC.element_to_be_clickable(APPLY_BTN_SEL)).click()
    w.until(EC.element_to_be_clickable(SEARCH_BTN_SEL)).click()
    w.until(lambda drv: exists(By.CSS_SELECTOR, SEARCH_LIST_BOX_SEL) or exists(By.XPATH, EMPTY_SEL))

    if exists(By.XPATH, EMPTY_SEL):
        logger.warning(f"period_no_results | start={start_date} | end={end_date}")
        return results


    # GO TO THE BOTTOM OF THE FIRST LIST AND CLICK THE LAST ITEM
    scroll_first_results_list_to_bottom()

    bottom_row = get_bottom_search_row(retries=3)
    if not bottom_row:
        logger.warning(f"bottom_row_missing | start={start_date} | end={end_date}")
        return results

    bottom_title = get_search_row_title(bottom_row)
    logger.info(f"bottom_row_click | title={bottom_title}")

    prev_title, prev_body = get_panel_fast_state()

    click_search_row(bottom_row)
    wait_split_view()

    try:
        wait_until_transcript_ready(prev_title, prev_body, timeout=READY_WAIT)
    except TimeoutException:
        pass

    logger.info("split_view_opened")


    # SCROLL BACK TO THE TOP OF THE LEFT LIST
    scroll_left_list_to_top()


    # CLICK EACH TRANSCRIPT IN THE LEFT LIST AND SCRAPE IT
    left_box = get_left_list_box()
    if left_box is None:
        logger.warning("left_list_box_missing")
        return results

    step = 3500
    max_idle = 4
    idle = 0
    prev_scroll = -1

    while idle < max_idle:
        current_items = visible_left_items()

        for item in current_items:
            try:
                key = item_key(item)
                if not key:
                    continue

                data = scrape_current_transcript_with_retry(item, key, retries=RETRY_COUNT)

                results.append(data)
                total_scraped += 1

                logger.info(
                    f"item_scraped | total_scraped={total_scraped} | "
                    f"title={data['title'] or key}"
                )
                results = save_results(results)

            except StaleElementReferenceException:
                total_stale += 1
                logger.warning("item_stale | error_type=StaleElementReferenceException")
                continue

            except Exception as e:
                total_failed += 1
                logger.error(
                    f"item_error | error_type={type(e).__name__} | message={e}"
                )

        try:
            top, h, ch = d.execute_script(
                "const e=arguments[0]; return [e.scrollTop, e.scrollHeight, e.clientHeight];",
                left_box
            )

            if top + ch >= h - 5:
                logger.info("left_list_bottom_reached")
                break

            d.execute_script("arguments[0].scrollTop += arguments[1];", left_box, step)

            try:
                wait(0.25).until(
                    lambda drv: drv.execute_script("return arguments[0].scrollTop;", left_box) != top
                )
            except TimeoutException:
                pass

            new_scroll = d.execute_script("return arguments[0].scrollTop;", left_box)
            idle = idle + 1 if new_scroll == prev_scroll else 0
            prev_scroll = new_scroll

        except StaleElementReferenceException:
            left_box = get_left_list_box()
            if left_box is None:
                break

    return results


# 1) LOGIN
d.get(LOGIN_URL)
accept_cookies_if_present()

w.until(EC.element_to_be_clickable((By.NAME, "email"))).send_keys(EMAIL)
w.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password']"))).send_keys(PWD)
w.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit' or normalize-space()='Sign in']"))).click()
w.until(lambda x: "login" not in x.current_url.lower() or not x.find_elements(By.NAME, "email"))


# 2) ONE-TIME INITIALIZATION OF THE EARNINGS CALLS FILTER
d.get(TARGET_URL)
w.until(lambda x: "login" not in x.current_url.lower())
accept_cookies_if_present()
ensure_earnings_calls_only()


# 3) LOOP OVER 2-DAY PERIODS
results = []
first_run = True

for start_date, end_date in make_windows(GLOBAL_START, GLOBAL_END, WINDOW_DAYS):
    try:
        results = run_one_period(
            start_date,
            end_date,
            results,
            first_run=first_run
        )
        first_run = False
    except Exception as e:
        logger.error(
            f"period_error | start={start_date} | end={end_date} | "
            f"error_type={type(e).__name__} | message={e}"
        )
        first_run = False


# 4) FINAL SAVE
results = save_results(results, force=True)
logger.info(
    f"scraping_completed | total_scraped={total_scraped} | "
    f"total_failed={total_failed} | total_stale={total_stale}"
)