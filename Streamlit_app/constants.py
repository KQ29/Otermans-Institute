# constants.py

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

PRIMARY_JSON_PATH = DATA_DIR / "fake_data.json"
SECONDARY_JSON_PATH = DATA_DIR / "four_students.json"
DEFAULT_JSON_PATH = PRIMARY_JSON_PATH

# Personalisation keys seen across datasets
AVATAR_KEYS = ["avatar", "avatar_name", "selected_avatar", "active_avatar", "avatarId", "avatar_id"]
FONT_KEYS = ["font", "font_name", "selected_font", "text_font"]
BACKGROUND_KEYS = ["background", "background_name", "background_theme", "bg_theme", "bg", "selected_background"]
