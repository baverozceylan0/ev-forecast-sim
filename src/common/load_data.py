import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

import pandas as pd
import requests
import zipfile
from io import BytesIO

from src.constants import Paths
from src.common.utils import validate_ev_charging_dataframe

import os

logger = logging.getLogger(__name__)
RAW_DATAPATH = Path.cwd() / Paths["RAW_DATA"]


class DataLoader(ABC):
    """Abstract base class for dataset loading and validation."""

    def __init__(self, dataset_id: str, force_download: bool = False):
        self.dataset_id = dataset_id
        self.force_download = force_download

    def load(self) -> pd.DataFrame:
        """Loads the dataset and validates structure and types."""
        df = self._load()
        return validate_ev_charging_dataframe(df)

    @abstractmethod
    def _load(self) -> pd.DataFrame:
        pass

    @staticmethod
    def _download_and_extract(url: str, target_file_in_zip: str, output_file: Path) -> None:      
        """Download a ZIP file and extract the target to the output path."""
        os.makedirs(RAW_DATAPATH, exist_ok=True)
        logger.info(f"Downloading dataset from: {url}")

        # Download
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download file (status code: {response.status_code})")

        # Extract
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            if target_file_in_zip not in zf.namelist():
                raise FileNotFoundError(f"'{target_file_in_zip}' not found in ZIP archive.")
            zf.extract(target_file_in_zip, path=RAW_DATAPATH)
            extracted_path = RAW_DATAPATH / target_file_in_zip
            extracted_path.rename(output_file)
            logger.info(f"Extracted and renamed '{target_file_in_zip}' → '{output_file.name}'")


class ASRDataLoader(DataLoader): 
    """Loader for the ASR dataset."""

    def _load(self) -> pd.DataFrame:
        # URL pointing to the zipped dataset hosted on 4TU repository
        url = "https://data.4tu.nl/ndownloader/items/80ef3824-3f5d-4e45-8794-3b8791efbd13/versions/1"        
        target_file_in_zip = "202410DatasetEVOfficeParking_v0.csv"
        final_path = RAW_DATAPATH / f"{self.dataset_id}.csv"

        if final_path.exists() and not self.force_download:
            logger.info(f"The dataset '{self.dataset_id}' already exists in '{RAW_DATAPATH}' and will not be re-downloaded.")
        else:
            logger.info(f"Preparing to download: {self.dataset_id}")
            tmp_path = RAW_DATAPATH / "tmp.csv"
            self._download_and_extract(url, target_file_in_zip, tmp_path)

            # Select only the required columns, and save them to the final CSV
            df = pd.read_csv(tmp_path, delimiter=";")
            df[["EV_id_x", "start_datetime", "end_datetime", "total_energy"]].to_csv(final_path, index=False)            
            logger.info(f"The dataset '{self.dataset_id}' is saved as: {final_path.name}")
            tmp_path.unlink(missing_ok=True)
            logger.info(f"Deleted: {tmp_path.name}")

        df = pd.read_csv(final_path)
        df["EV_id_x"] = df["EV_id_x"].astype("string")
        df["start_datetime"] = pd.to_datetime(df["start_datetime"])
        df["end_datetime"] = pd.to_datetime(df["end_datetime"])
        df["total_energy"] = df["total_energy"].astype("float")                   

        return df    


class DataLoaderFactory:
    """Factory for instantiating dataset-specific DataLoader objects."""

    @staticmethod
    def get_loader(dataset_id: str, force_download: bool = False) -> DataLoader:
        loaders: dict[str, Type[DataLoader]] = {
            "ASR": ASRDataLoader,
        }

        if dataset_id not in loaders:
            raise ValueError(f"No DataLoader defined for dataset: {dataset_id}")

        return loaders[dataset_id](dataset_id, force_download=force_download)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    loader = DataLoaderFactory.get_loader("ASR", force_download=True)
    df = loader.load()
    print(df.info())