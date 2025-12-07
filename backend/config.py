from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_hostname: str = "localhost"
    database_port: str = "5432"
    database_password: str
    database_name: str
    database_username: str

    secret_key: str
    algorithm: str
    access_token_expire_minutes: int

    gemini_api_key: str | None = None
    tavily_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid"   # keeps strict validation
    )


settings = Settings()
