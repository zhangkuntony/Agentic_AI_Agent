# config.py - 项目配置文件（从 .env 加载配置）

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# AI API配置
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")