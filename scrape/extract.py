from pathlib import Path
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


class SmartprixDataExtract:
    """
    A class to parse and extract data from the saved HTML of the Smartprix website.
    """

    def __init__(self, input_html_path: Path):
        """
        Initializes the data scraper with the path to the input HTML file.

        Args:
            input_html_path (Path): Path to the saved HTML file.
        """
        self.input_html_path = input_html_path
        self.soup = None
        self.data = {
            'model': [],
            'price': [],
            'rating': [],
            'sim': [],
            'processor': [],
            'ram': [],
            'battery': [],
            'display': [],
            'camera': [],
            'card': [],
            'os': []
        }

    def load_html(self) -> None:
        """
        Loads the HTML content from the specified file and initializes BeautifulSoup.
        """
        with self.input_html_path.open("r", encoding="utf-8") as file:
            html = file.read()
        self.soup = BeautifulSoup(html, "lxml")

    def scrape_data(self) -> None:
        """
        Scrapes the required data from the loaded HTML and populates the data dictionary.
        """
        containers = self.soup.find_all(
            "div", {"class": "sm-product has-tag has-features has-actions"}
        )

        for container in containers:
            try:
                self.data["model"].append(container.find("h2").text)
            except:
                self.data["model"].append(np.nan)
            try:
                self.data["price"].append(container.find("span", {"class": "price"}).text)
            except:
                self.data["price"].append(np.nan)
            try:
                self.data["rating"].append(
                    container.find("div", {"class": "score rank-2-bg"}).find("b").text
                )
            except:
                self.data["rating"].append(np.nan)

            features = container.find("ul", {"class": "sm-feat specs"}).find_all("li")
            self._extract_features(features)

    def _extract_features(self, features: list) -> None:
        """
        Extracts additional features such as SIM, processor, RAM, etc., from the feature list.

        Args:
            features (list): A list of HTML elements containing the feature details.
        """
        keys = ["sim", "processor", "ram", "battery", "display", "camera", "card", "os"]
        for idx, key in enumerate(keys):
            try:
                self.data[key].append(features[idx].text)
            except:
                self.data[key].append(np.nan)

    def export_to_csv(self, output_path: Path) -> None:
        """
        Exports the scraped data to a CSV file.

        Args:
            output_path (Path): Path to save the CSV file.
        """
        df = pd.DataFrame(self.data)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        df.to_csv(output_path, index=False, encoding="utf-8")

    def run(self, output_path: Path) -> None:
        """
        Executes the complete scraping process and exports the data to a CSV file.

        Args:
            output_path (Path): Path to save the exported CSV file.
        """
        self.load_html()
        self.scrape_data()
        self.export_to_csv(output_path)


if __name__ == "__main__":

    input_html_path = Path("scrape/smartprix.html") 
    output_csv_path = Path("data/raw/smartphones.csv")  

    scraper = SmartprixDataExtract(input_html_path=input_html_path)
    scraper.run(output_csv_path)
