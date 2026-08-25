import yaml
from pathlib import Path
from pydantic import BaseModel, RootModel


class genericSettings(BaseModel):
    @classmethod
    def load_from_yaml(cls, file_path: Path):
        """Loads settings from a YAML file and returns an instance of the class"""
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return(cls(**config_data))
