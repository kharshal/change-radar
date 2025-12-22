# ============================================================================
# FILE: services/file_handling.py
# ============================================================================
"""
File handling service for uploads and storage.
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import streamlit as st


class FileHandling:
    """Handle file uploads and storage operations."""
    
    @staticmethod
    def create_session_folder() -> str:
        """
        Create a timestamped folder for the current session.
        
        Returns:
            str: Path to the created folder
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"change_radar_{timestamp}"
        folder_path = "files" / Path(folder_name)
        
        # Create main folder and subfolders
#         data_folder = folder_path / "data"
#         docs_folder = folder_path / "docs"
#         output_folder = folder_path / "output"
        
#         data_folder.mkdir(parents=True, exist_ok=True)
#         docs_folder.mkdir(parents=True, exist_ok=True)
#         output_folder.mkdir(parents=True, exist_ok=True)
        
        return str(folder_path)
    
    @staticmethod
    def save_uploaded_files(files, folder_path: str, subfolder: str = "data") -> List[str]:
        """
        Save uploaded files to specified folder.
        
        Args:
            files: Streamlit uploaded files
            folder_path: Base folder path
            subfolder: Subfolder name (data/docs)
            
        Returns:
            List[str]: Paths of saved files
        """
        saved_files = []
        target_folder = Path(folder_path) / subfolder
        target_folder.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            file_path = target_folder / file.name
            
            # Write file to disk
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            saved_files.append(str(file_path))
        
        return saved_files
    
    @staticmethod
    def get_file_info(files) -> Tuple[int, float]:
        """
        Get information about uploaded files.
        
        Args:
            files: Streamlit uploaded files
            
        Returns:
            Tuple[int, float]: Number of files and total size in MB
        """
        if not files:
            return 0, 0.0
        
        total_size = sum(file.size for file in files)
        total_size_mb = total_size / (1024 * 1024)
        
        return len(files), total_size_mb

