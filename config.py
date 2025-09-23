import os

class Config:
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

# ? __init__.py ? `from .config import config` ???
config = Config()
