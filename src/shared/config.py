"""
Eva Live Configuration Management

This module handles loading and managing configuration from YAML files,
environment variables, and provides typed access to all settings.
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field
from pydantic_settings import SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    
    # PostgreSQL
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_database: str = Field(default="eva_live", alias="POSTGRES_DATABASE")
    postgres_username: str = Field(default="eva_user", alias="POSTGRES_USERNAME")
    postgres_password: str = Field(alias="POSTGRES_PASSWORD")
    postgres_pool_size: int = Field(default=20, alias="POSTGRES_POOL_SIZE")
    
    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_database: int = Field(default=0, alias="REDIS_DATABASE")
    redis_password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=100, alias="REDIS_MAX_CONNECTIONS")
    
    # Vector Database
    vector_db_provider: str = Field(default="pinecone", alias="VECTOR_DB_PROVIDER")
    pinecone_api_key: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-west1-gcp", alias="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="eva-live-knowledge", alias="PINECONE_INDEX_NAME")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class AIServiceConfig(BaseSettings):
    """AI/ML service configuration"""
    
    # OpenAI
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", alias="OPENAI_EMBEDDING_MODEL")
    openai_max_tokens: int = Field(default=4096, alias="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, alias="OPENAI_TEMPERATURE")
    
    # Google Cloud
    google_cloud_api_key: Optional[str] = Field(default=None, alias="GOOGLE_CLOUD_API_KEY")
    google_cloud_project_id: Optional[str] = Field(default=None, alias="GOOGLE_CLOUD_PROJECT_ID")
    
    # ElevenLabs
    elevenlabs_api_key: Optional[str] = Field(default=None, alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", alias="ELEVENLABS_VOICE_ID")
    elevenlabs_model_id: str = Field(default="eleven_monolingual_v1", alias="ELEVENLABS_MODEL_ID")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class SecurityConfig(BaseSettings):
    """Security and authentication configuration"""
    
    jwt_secret_key: str = Field(alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(default=7, alias="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    
    encryption_key: str = Field(alias="ENCRYPTION_KEY")
    encryption_algorithm: str = Field(default="AES-256-GCM", alias="ENCRYPTION_ALGORITHM")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class IntegrationConfig(BaseSettings):
    """Platform integration configuration"""
    
    # Zoom
    zoom_api_key: Optional[str] = Field(default=None, alias="ZOOM_API_KEY")
    zoom_api_secret: Optional[str] = Field(default=None, alias="ZOOM_API_SECRET")
    zoom_webhook_secret: Optional[str] = Field(default=None, alias="ZOOM_WEBHOOK_SECRET")
    
    # Microsoft Teams
    teams_tenant_id: Optional[str] = Field(default=None, alias="TEAMS_TENANT_ID")
    teams_client_id: Optional[str] = Field(default=None, alias="TEAMS_CLIENT_ID")
    teams_client_secret: Optional[str] = Field(default=None, alias="TEAMS_CLIENT_SECRET")
    
    # Google Meet
    google_client_id: Optional[str] = Field(default=None, alias="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[str] = Field(default=None, alias="GOOGLE_CLIENT_SECRET")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class EvaLiveConfig:
    """Main Eva Live configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from YAML file and environment variables"""
        self.config_path = config_path or self._find_config_file()
        self._config_data = self._load_yaml_config()
        
        # Initialize typed configuration sections
        self.database = DatabaseConfig()
        self.ai_services = AIServiceConfig()
        self.security = SecurityConfig()
        self.integrations = IntegrationConfig()
        
        # Load additional configuration from YAML
        self._load_additional_config()
    
    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations"""
        possible_paths = [
            "configs/config.yaml",
            "../configs/config.yaml",
            "../../configs/config.yaml",
            "/etc/eva-live/config.yaml",
            os.path.expanduser("~/.eva-live/config.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Configuration file not found in standard locations")
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config_content = file.read()
                # Replace environment variables in the YAML content
                config_content = self._substitute_env_vars(config_content)
                return yaml.safe_load(config_content)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in configuration content"""
        import re
        
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))  # Return original if env var not found
        
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, content)
    
    def _load_additional_config(self):
        """Load additional configuration sections from YAML"""
        # App settings
        app_config = self._config_data.get('app', {})
        self.app_name = app_config.get('name', 'Eva Live')
        self.app_version = app_config.get('version', '1.0.0')
        self.environment = app_config.get('environment', 'development')
        self.debug = app_config.get('debug', True)
        self.host = app_config.get('host', '0.0.0.0')
        self.port = app_config.get('port', 8000)
        
        # API settings
        api_config = self._config_data.get('api', {})
        self.api_base_url = api_config.get('base_url', 'https://api.eva-live.com/v1')
        self.websocket_url = api_config.get('websocket_url', 'wss://ws.eva-live.com/v1')
        self.rate_limits = api_config.get('rate_limits', {})
        
        # Performance targets
        performance_config = self._config_data.get('performance', {})
        self.latency_targets = performance_config.get('latency_targets', {})
        self.quality_targets = performance_config.get('quality_targets', {})
        
        # Avatar configuration
        avatar_config = self._config_data.get('avatar', {})
        self.avatar_default_config = avatar_config.get('default_config', {})
        self.avatar_rendering = avatar_config.get('rendering', {})
        
        # Virtual devices
        virtual_devices_config = self._config_data.get('virtual_devices', {})
        self.virtual_camera = virtual_devices_config.get('camera', {})
        self.virtual_microphone = virtual_devices_config.get('microphone', {})
        
        # Monitoring
        monitoring_config = self._config_data.get('monitoring', {})
        self.monitoring_metrics = monitoring_config.get('metrics', {})
        self.monitoring_alerts = monitoring_config.get('alerts', {})
        self.monitoring_logging = monitoring_config.get('logging', {})
        
        # Content processing
        content_config = self._config_data.get('content', {})
        self.content_supported_formats = content_config.get('supported_formats', {})
        self.content_processing = content_config.get('processing', {})
        
        # Session management
        session_config = self._config_data.get('session', {})
        self.session_default_duration = session_config.get('default_duration', 180)
        self.session_max_concurrent_per_user = session_config.get('max_concurrent_per_user', 5)
        self.session_cleanup_interval = session_config.get('cleanup_interval', 300)
        self.session_heartbeat_interval = session_config.get('heartbeat_interval', 30)
        self.session_fallback = session_config.get('fallback', {})
        
        # Feature flags
        features_config = self._config_data.get('features', {})
        self.features = features_config
        
        # Development settings
        development_config = self._config_data.get('development', {})
        self.development = development_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key"""
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_database_url(self) -> str:
        """Get PostgreSQL database URL"""
        return (
            f"postgresql://{self.database.postgres_username}:"
            f"{self.database.postgres_password}@"
            f"{self.database.postgres_host}:{self.database.postgres_port}/"
            f"{self.database.postgres_database}"
        )
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        auth_part = f":{self.database.redis_password}@" if self.database.redis_password else ""
        return (
            f"redis://{auth_part}{self.database.redis_host}:"
            f"{self.database.redis_port}/{self.database.redis_database}"
        )
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        return self.features.get(feature, False)
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == 'production'


# Global configuration instance
config: Optional[EvaLiveConfig] = None


def get_config() -> EvaLiveConfig:
    """Get the global configuration instance"""
    global config
    if config is None:
        config = EvaLiveConfig()
    return config


def init_config(config_path: Optional[str] = None) -> EvaLiveConfig:
    """Initialize the global configuration instance"""
    global config
    config = EvaLiveConfig(config_path)
    return config


def reload_config() -> EvaLiveConfig:
    """Reload the configuration from file"""
    global config
    if config is not None:
        config = EvaLiveConfig(config.config_path)
    else:
        config = EvaLiveConfig()
    return config
