from pydantic_settings import BaseSettings
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # If the user forgets to set this, it defaults to 1000
    # If they pass "500" in a .env file, Pydantic converts it to an int automatically
    chunk_size: int = Field(default=1000, gt=0, description="Size of the text chunks")
    chunk_overlap: int = Field(default=100, ge=0, description="Overlap between chunks")
    chunk_threshold: float = Field(default=0.35, ge=0, description  ="threshold value to evaluate chunk with proxy matrics")
    
    # It will automatically look for OPENAI_API_KEY in your environment
    GROQ_API_KEY: str = Field(default="",description="Required API key")
    HF_TOKEN: str = Field(default="", description="Required API key")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

def get_settings() -> Settings:
    return Settings()