import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


class SmartprixScraper:
    """
    A scraper class to automate interactions with the Smartprix website
    using Selenium and save the page source to a local HTML file.
    """

    def __init__(self, driver_path: Path):
        """
        Initializes the web driver for Selenium.

        Args:
            driver_path (Path): Path to the ChromeDriver executable.
        """
        self.driver_path = driver_path
        self.driver = None

    def start_driver(self) -> None:
        """
        Starts the Selenium web driver with the provided service path.
        """
        service = Service(str(self.driver_path))
        self.driver = webdriver.Chrome(service=service)

    def open_page(self, url: str) -> None:
        """
        Opens the specified URL in the web browser.

        Args:
            url (str): The URL to open.
        """
        if self.driver:
            self.driver.get(url)
            time.sleep(1)
        else:
            raise RuntimeError("Driver not started. Call start_driver() first.")

    def apply_filters(self) -> None:
        """
        Applies filters on the Smartprix website by interacting with checkboxes.
        """
        self.driver.find_element(
            by=By.XPATH,
            value='//*[@id="app"]/main/aside/div/div[5]/div[2]/label[1]/input',
        ).click()
        time.sleep(1)
        self.driver.find_element(
            by=By.XPATH,
            value='//*[@id="app"]/main/aside/div/div[5]/div[2]/label[2]/input',
        ).click()
        time.sleep(2)

    def scroll_page(self) -> None:
        """
        Scrolls the webpage to load dynamic content until no new content is loaded.
        """
        old_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.find_element(
                by=By.XPATH, value='//*[@id="app"]/main/div[1]/div[2]/div[3]'
            ).click()
            time.sleep(1)

            new_height = self.driver.execute_script(
                "return document.body.scrollHeight"
            )

            print(f"Old height: {old_height}, New height: {new_height}")

            if new_height == old_height:
                break

            old_height = new_height

    def save_page_source(self, output_path: Path) -> None:
        """
        Saves the HTML page source to a specified file.

        Args:
            output_path (Path): Path to save the HTML file.
        """
        html = self.driver.page_source
        output_path.parent.mkdir(parents=True, exist_ok=True)  
        with output_path.open("w", encoding="utf-8") as file:
            file.write(html)

    def close_driver(self) -> None:
        """
        Closes the web driver.
        """
        if self.driver:
            self.driver.quit()

    def run(self, url: str, output_path: Path) -> None:
        """
        Executes the entire scraping process.

        Args:
            url (str): The URL to scrape.
            output_path (Path): Path to save the HTML file.
        """
        try:
            self.start_driver()
            self.open_page(url)
            self.apply_filters()
            self.scroll_page()
            self.save_page_source(output_path)
        finally:
            self.close_driver()


if __name__ == "__main__":

    driver_path = Path("C:/Users/PRAMIT DE/Desktop/chromedriver.exe")
    output_path = Path("scrape/smartprix.html")
    scraper = SmartprixScraper(driver_path=driver_path)
    scraper.run("https://www.smartprix.com/mobiles", output_path)
