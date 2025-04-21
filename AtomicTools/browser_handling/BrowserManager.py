
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
class BrowserManager:
    def __init__(self, user_agent: str = None, headless: bool = True, viewport: dict = None):
        self._playwright = sync_playwright().start()
        self.browser: Browser = self._playwright.chromium.launch(headless=headless)
        self.context: BrowserContext = self.browser.new_context(
            user_agent=user_agent or (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            locale="en-US",
            viewport=viewport or {"width": 1280, "height": 800},
        )

    def get_page(self) -> Page:
        try:
            return self.context.new_page()
        except Exception as e:
            print(f"[BrowserManager] Failed to get new page: {e}. Resetting context.")
            self.reset_context()
            return self.context.new_page()

    def reset_context(self):
        if self.context:
            self.context.close()
        self.context = self.browser.new_context()

    def close(self):
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self._playwright:
            self._playwright.stop()


# Singleton instance
_browser_manager: BrowserManager = None

def get_browser_manager(user_agent: str = None, headless: bool = True, viewport: dict = None) -> BrowserManager:
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = BrowserManager(user_agent=user_agent, headless=headless, viewport=viewport)
    return _browser_manager