# language_utils.py
import json
import os
import streamlit as st

@st.cache_data
def load_translations():
    translations = {}
    locales_dir = "locales"
    for lang_file in os.listdir(locales_dir):
        if lang_file.endswith(".json"):
            lang_code = lang_file.split(".")[0]
            with open(os.path.join(locales_dir, lang_file), "r", encoding="utf-8") as f:
                translations[lang_code] = json.load(f)
    return translations

def get_translator(translations, lang):
    def translate(key):
        return translations.get(lang, {}).get(key, translations.get("en", {}).get(key, key))
    return translate