import pandas as pd
from pathlib import Path

class FileReader:

    def __init__(self, file_name: str):
        self.file_name = file_name


    def read_file(self) -> pd.DataFrame:
        file_path = Path('file', self.file_name)
        data=pd.read_csv(file_path)
        return data