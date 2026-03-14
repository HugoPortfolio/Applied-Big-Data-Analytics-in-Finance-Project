from selenium.webdriver.common.by import By


DATE_BOX_SEL = (By.CSS_SELECTOR, "div[class*='time-range__root']")
APPLY_BTN_SEL = (
    By.XPATH,
    "//button[.//label[normalize-space()='Apply Dates'] or normalize-space()='Apply Dates']"
)
DATE_INP_CSS = "input[placeholder='MM/DD/YYYY']"
SEARCH_BTN_SEL = (By.XPATH, "//button[contains(., 'Search Transcripts')]")
EMPTY_SEL = "//*[contains(., 'was not found anywhere') or contains(., 'searched all available transcripts')]"

SEARCH_LIST_BOX_SEL = "div[class*='news-virtual-list__newsVirtualList__container']"
SEARCH_ROWS_SEL = "div[class*='news-virtual-list__newsVirtualList__items'] > div[style*='position: absolute']"
SEARCH_TITLE_SEL = "label[class*='text-label']"
SEARCH_ITEM_SEL = "div[class*='koy-news-item__koyNewsItem']"

LEFT_ITEMS_SEL = "div[class*='koy-news-item__koyNewsItem']"
RIGHT_PANEL_SEL = "div[class*='news-article-panel__newsArticlePanel__root']"

ARTICLE_ROOT_SEL = "div[class*='common-news-article-content__commonNewsArticleContent__root']"
ARTICLE_TITLE_SEL = "div[class*='article-title__articleTitle__heading']"
ARTICLE_SUBHEADER_SEL = "div[class*='article-title__articleTitle__subHeading']"
ARTICLE_TRANSCRIPT_SUBHEADER_SEL = "div[class*='article-title__articleTitle__transcriptSubHeader']"
ARTICLE_SPEAKERS_SEL = "div[class*='transcript-speakers__transcriptSpeakers__content']"
ARTICLE_BODY_SEL = "div[class*='slate-editor__container']"

EARNINGS_CALLS_FILTER_SEL = (
    By.XPATH,
    "//div[@role='button' and .//label[normalize-space()='Earnings Calls']]"
)