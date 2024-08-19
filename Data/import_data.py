import os
import requests
import zipfile


class DatasetDownloader:
    """
    This class is responsible for downloading and extracting a dataset from a given URL.
    """

    def __init__(self, url, dataset_path):
        """
        Initialize the DatasetDownloader with the URL of the dataset and the path to save it.
        """
        self.url = url
        self.dataset_path = dataset_path

    def download_and_extract(self):
        """
        Download the dataset from the URL and extract it to the specified path.
        """
        if not os.path.exists(self.dataset_path):
            print("Downloading code and datasets...")
            r = requests.get(self.url, allow_redirects=True)
            open(f"{self.dataset_path}.zip", "wb").write(r.content)

            with zipfile.ZipFile(f"{self.dataset_path}.zip", "r") as zip_ref:
                zip_ref.extractall("./")

            if os.path.exists("./v0.6.0.zip"):
                os.remove("./v0.6.0.zip")

            if os.path.exists(self.dataset_path):
                print("Download datasets success")
            else:
                print(
                    "Download datasets failed, please check the download url is valid or not."
                )
        else:
            print(f"{self.dataset_path} already exists")


if __name__ == "__main__":
    downloader = DatasetDownloader(
        "https://cnnorth4-modelhub-datasets-obsfs-sfnua.obs.cn-north-4.myhuaweicloud.com/content/c2c1853f-d6a6-4c9d-ac0e-203d4c304c88/NkxX5K/dataset/rebar_count_datasets.zip",
        "./rebar_count_datasets",)
    downloader.download_and_extract()
