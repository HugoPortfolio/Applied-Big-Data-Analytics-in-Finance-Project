import re
import time
from datetime import datetime, timedelta

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException,
    TimeoutException,
    ElementClickInterceptedException,
)

from scraping.config import POLL, READY_WAIT, RETRY_COUNT, PLACEHOLDER_TEXTS
from scraping.selectors import (
    SEARCH_ROWS_SEL,
    SEARCH_ITEM_SEL,
    SEARCH_LIST_BOX_SEL,
    LEFT_ITEMS_SEL,
    RIGHT_PANEL_SEL,
    ARTICLE_ROOT_SEL,
    ARTICLE_TITLE_SEL,
    ARTICLE_BODY_SEL,
    EARNINGS_CALLS_FILTER_SEL,
    DATE_INP_CSS,
)


def wait(driver, timeout, poll=POLL):
    return WebDriverWait(driver, timeout, poll_frequency=poll)


def exists(driver, by, sel):
    return len(driver.find_elements(by, sel)) > 0


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


def accept_cookies_if_present(driver):
    for xp in [
        "//button[contains(., 'Accept All')]",
        "//button[contains(., 'Accept all')]",
        "//button[contains(., 'Accept')]",
    ]:
        try:
            wait(driver, 0.7).until(EC.element_to_be_clickable((By.XPATH, xp))).click()
            return True
        except Exception:
            pass
    return False


def set_val(driver, el, val):
    driver.execute_script(
        "const el=arguments[0],v=arguments[1];"
        "el.focus();"
        "Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,v);"
        "el.dispatchEvent(new Event('input',{bubbles:true}));"
        "el.dispatchEvent(new Event('change',{bubbles:true}));"
        "el.blur();",
        el, val
    )


def js_click(driver, el):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    try:
        el.click()
    except Exception:
        driver.execute_script("arguments[0].click();", el)


def is_filter_selected(driver, el):
    return driver.execute_script(
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


def ensure_earnings_calls_only(driver, base_wait):
    btn = base_wait.until(EC.presence_of_element_located(EARNINGS_CALLS_FILTER_SEL))
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)

    if not is_filter_selected(driver, btn):
        js_click(driver, btn)
        try:
            wait(driver, 2).until(lambda drv: is_filter_selected(drv, btn))
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


def get_real_search_rows(driver):
    rows = driver.find_elements(By.CSS_SELECTOR, SEARCH_ROWS_SEL)
    out = []
    for row in rows:
        try:
            if row.is_displayed() and row.text.strip():
                out.append(row)
        except Exception:
            pass
    return out


def get_bottom_search_row(driver, retries=3):
    for _ in range(retries):
        rows = get_real_search_rows(driver)
        if rows:
            rows = sorted(rows, key=row_top_from_style)
            return rows[-1]
    return None


def get_search_row_title(row):
    try:
        return clean_text(row.text).split("\n")[0]
    except Exception:
        return ""


def click_search_row(driver, row):
    try:
        target = row.find_element(By.CSS_SELECTOR, SEARCH_ITEM_SEL)
    except Exception:
        target = row

    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", target)
    try:
        target.click()
    except Exception:
        driver.execute_script("arguments[0].click();", target)


def scroll_first_results_list_to_bottom(driver, base_wait, logger):
    box = base_wait.until(lambda drv: drv.find_element(By.CSS_SELECTOR, SEARCH_LIST_BOX_SEL))
    step = 1000
    pause = 1
    max_idle = 10
    idle = 0
    prev_top = -1

    while idle < max_idle:
        try:
            top, h, ch = driver.execute_script(
                "const e=arguments[0]; return [e.scrollTop, e.scrollHeight, e.clientHeight];",
                box
            )

            if top + ch >= h - 5:
                logger.info("search_list_bottom_reached")
                return

            driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollTop + arguments[1];",
                box, step
            )
            time.sleep(pause)

            new_top = driver.execute_script("return arguments[0].scrollTop;", box)
            if new_top == prev_top:
                idle += 1
            else:
                idle = 0
            prev_top = new_top

        except StaleElementReferenceException:
            time.sleep(0.8)
            box = base_wait.until(lambda drv: drv.find_element(By.CSS_SELECTOR, SEARCH_LIST_BOX_SEL))


def wait_split_view(driver):
    wait(driver, 2).until(lambda drv: len(drv.find_elements(By.CSS_SELECTOR, LEFT_ITEMS_SEL)) > 0)
    wait(driver, 2).until(lambda drv: len(drv.find_elements(By.CSS_SELECTOR, RIGHT_PANEL_SEL)) > 0)


def get_left_list_box(driver):
    candidates = driver.find_elements(By.CSS_SELECTOR, "div[class*='box__box__'][data-fill='true']")
    visible = []
    for el in candidates:
        try:
            if el.is_displayed():
                visible.append(el)
        except Exception:
            pass

    if visible:
        visible.sort(key=lambda x: driver.execute_script("return arguments[0].getBoundingClientRect().left;", x))
        return visible[0]

    items = driver.find_elements(By.CSS_SELECTOR, LEFT_ITEMS_SEL)
    if items:
        return items[0]

    return None


def scroll_left_list_to_top(driver):
    box = get_left_list_box(driver)
    if box is None:
        return
    try:
        driver.execute_script("arguments[0].scrollTop = 0;", box)
    except Exception:
        pass


def visible_left_items(driver):
    items = driver.find_elements(By.CSS_SELECTOR, LEFT_ITEMS_SEL)
    out = []
    for it in items:
        try:
            txt = clean_text(it.text)
            if it.is_displayed() and txt:
                out.append(it)
        except Exception:
            pass

    out.sort(key=lambda el: driver.execute_script("return arguments[0].getBoundingClientRect().top;", el))
    return out


def item_key(el):
    try:
        return clean_text(el.text)
    except Exception:
        return ""


def click_left_item(driver, el):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    try:
        el.click()
    except (ElementClickInterceptedException, StaleElementReferenceException):
        driver.execute_script("arguments[0].click();", el)


def get_article_root(driver):
    roots = driver.find_elements(By.CSS_SELECTOR, ARTICLE_ROOT_SEL)
    for r in roots:
        try:
            if r.is_displayed():
                return r
        except Exception:
            pass

    panels = driver.find_elements(By.CSS_SELECTOR, RIGHT_PANEL_SEL)
    for p in panels:
        try:
            if p.is_displayed():
                return p
        except Exception:
            pass

    return None


def get_panel_fast_state(driver):
    root = get_article_root(driver)
    if root is None:
        return "", ""

    title, body = driver.execute_script(
        """
        const root = arguments[0];

        const q = (sel) => {
            const el = root.querySelector(sel);
            return el ? (el.innerText || el.textContent || "").trim() : "";
        };

        const title = q(arguments[1]);
        const body  = q(arguments[2]);

        return [title, body];
        """,
        root,
        ARTICLE_TITLE_SEL,
        ARTICLE_BODY_SEL,
    )

    return clean_text(title), clean_text(body)


def wait_until_transcript_ready(driver, prev_title="", prev_body="", timeout=READY_WAIT):
    def _cond(_):
        title, body = get_panel_fast_state(driver)

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

    wait(driver, timeout).until(_cond)


def scrape_current_transcript_fast(driver):
    root = get_article_root(driver)
    if root is None:
        return {
            "title": "",
            "subheader": "",
            "transcript_subheader": "",
            "speakers": "",
            "body": "",
        }

    title, subheader, transcript_subheader, speakers, body = driver.execute_script(
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


def scrape_current_transcript_with_retry(driver, item, key, retries=RETRY_COUNT):
    last_data = None

    for _ in range(retries):
        try:
            prev_title, prev_body = get_panel_fast_state(driver)

            click_left_item(driver, item)
            wait_until_transcript_ready(driver, prev_title, prev_body, timeout=READY_WAIT)

            data = scrape_current_transcript_fast(driver)
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


def make_windows(start_str, end_str, window_days=2):
    start_dt = datetime.strptime(start_str, "%m/%d/%Y")
    end_dt = datetime.strptime(end_str, "%m/%d/%Y")

    cur = start_dt
    while cur <= end_dt:
        win_end = min(cur + timedelta(days=window_days - 1), end_dt)
        yield cur.strftime("%m/%d/%Y"), win_end.strftime("%m/%d/%Y")
        cur += timedelta(days=window_days)


def get_visible_date_inputs(driver):
    return [i for i in driver.find_elements(By.CSS_SELECTOR, DATE_INP_CSS) if i.is_displayed()]


def fill_date_range(driver, start_date, end_date):
    for _ in range(2):
        try:
            a, b = get_visible_date_inputs(driver)[:2]
            set_val(driver, a, start_date)
            set_val(driver, b, end_date)
            b.send_keys(Keys.TAB)
            return True
        except StaleElementReferenceException:
            continue
    return False