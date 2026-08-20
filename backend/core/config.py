from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SLUG_")

    app_name: str = "speak-local-understand-global"
    debug: bool = True
    api_prefix: str = "/api/v1"

    database_url: str = "postgresql+asyncpg://localhost:5432/speak_local"

    whisper_model: str = "small"
    whisper_model_telugu: str = "large-v3"
    translation_backend: str = "indic2"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    device: str = "auto"


settings = Settings()