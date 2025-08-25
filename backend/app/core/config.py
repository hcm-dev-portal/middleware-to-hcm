from pydantic import BaseModel

class Settings(BaseModel):
    app_name: str = "AI DB Assistant"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # picked up by boto3 and our services
    AWS_REGION: str = "ap-southeast-2"
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    OPENAI_API_KEY: str | None = None

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

