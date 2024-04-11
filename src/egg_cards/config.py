from pathlib import Path
import logging
import os
from dotenv import load_dotenv

from pathlib import Path
import logging
import os


load_dotenv() 


ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

DATA_DIR = ROOT_DIR / 'data'

RAW_DATA_DIR = DATA_DIR / 'raw'

INTERMEDIATE_DATA_DIR = DATA_DIR / 'intermediate'
INTERMEDIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = DATA_DIR / '.cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')