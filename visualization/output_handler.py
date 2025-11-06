"""
Output handler for saving visualizations with timestamp organization
"""
import os
from datetime import datetime
from pathlib import Path


class OutputHandler:
    """
    Manages output directory structure for visualizations.
    Creates timestamped folders to organize visualization outputs.
    """

    _current_session_dir = None
    _base_dir = "images"

    @classmethod
    def initialize_session(cls, base_dir="images"):
        """
        Initialize a new session with a timestamped directory.

        Parameters:
        -----------
        base_dir : str
            Base directory for all images (default: 'images')
        """
        cls._base_dir = base_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls._current_session_dir = os.path.join(base_dir, f"run_{timestamp}")

        # Create the directory
        Path(cls._current_session_dir).mkdir(parents=True, exist_ok=True)

        return cls._current_session_dir

    @classmethod
    def get_output_path(cls, filename):
        """
        Get the full path for saving a file.
        Automatically initializes session if not already done.

        Parameters:
        -----------
        filename : str
            Name of the file to save

        Returns:
        --------
        str
            Full path including session directory
        """
        if cls._current_session_dir is None:
            cls.initialize_session()

        return os.path.join(cls._current_session_dir, filename)

    @classmethod
    def get_session_dir(cls):
        """
        Get the current session directory path.

        Returns:
        --------
        str
            Path to current session directory
        """
        if cls._current_session_dir is None:
            cls.initialize_session()

        return cls._current_session_dir

    @classmethod
    def reset_session(cls):
        """
        Reset the current session (for testing or multiple runs).
        """
        cls._current_session_dir = None

