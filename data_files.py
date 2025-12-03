import os
from enum import Enum


class DataFiles(Enum):
    """Constants for dataset filenames stored in the data directory."""

    __IMAGE_FILE_PATTERN__ = "*.png"
    __MODEL_FILE_PATTERN__ = "*.pt"
    __MAPPING_FILE_PATTERN__ = "*.npz"

    FACIES = "facies"
    WELLS = "wells"
    SEISMIC = "seismic"

    def __init__(self,  value: str, data_dir: str = "./data") -> None:
        self.filename = value
        self.data_dir = data_dir

    def as_data_path(self, data_dir: str | None = None) -> str:
        """Return the full filesystem path for this data file inside `data_dir`.

        Parameters
        ----------
        data_dir : str
            Directory where the dataset files are stored.

        Returns
        -------
        str
            Full path to the file represented by this enum member.
        """
        return os.path.join(data_dir or self.data_dir, self.filename)
    
    @property
    def image_file_pattern(self) -> str:
        """Get the image file pattern for this data type.

        Returns
        -------
        str
            File pattern for image files of this data type.
        """
        return self.__IMAGE_FILE_PATTERN__

    @property
    def model_file_pattern(self) -> str:
        """Get the model file pattern for this data type.

        Returns
        -------
        str
            File pattern for model files of this data type.
        """
        return self.__MODEL_FILE_PATTERN__

    @property
    def mapping_file_pattern(self) -> str:
        """Get the mapping file pattern for this data type.

        Returns
        -------
        str
            File pattern for mapping files of this data type.
        """
        return self.__MAPPING_FILE_PATTERN__
