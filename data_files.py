"""Data file management for facies, seismic, and wells datasets."""

from enum import Enum


class DataFiles(Enum):
    """Enumeration for different types of data files with their paths and patterns."""

    FACIES = "facies"
    SEISMIC = "seismic"
    WELLS = "well"

    def as_data_path(self) -> str:
        """Return the directory path for this data file type.

        Returns
        -------
        str
            Path to the data directory (e.g., 'data/dataset/facies').
        """
        return f"data/dataset/{self.value}"

    @property
    def image_file_pattern(self) -> str:
        """Return the glob pattern for image files of this type.

        Returns
        -------
        str
            Glob pattern for matching image files (e.g., '*.png').
        """
        match self:
            case DataFiles.FACIES:
                return "xz_crossline_*_highres.png"
            case DataFiles.SEISMIC:
                return "xz_crossline_*.png"
            case DataFiles.WELLS:
                return "xz_crossline_*_highres.png"

    @property
    def model_file_pattern(self) -> str:
        """Return the glob pattern for model checkpoint files of this type.

        Returns
        -------
        str
            Glob pattern for matching model files (e.g., '*.pt').
        """
        return "xz_crossline_*.pt"

    @property
    def mapping_file_pattern(self) -> str:
        """Return the glob pattern for mapping files (wells only).

        Returns
        -------
        str
            Glob pattern for matching mapping files (e.g., '*.npz').
        """
        return "*.npz"
