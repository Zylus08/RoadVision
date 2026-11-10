"""Configuration management for RoadVision."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


class Config:
    """Configuration loader with environment variable overrides."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self._load_config()

    def _load_config(self):
        """Load configuration from YAML file with env var overrides."""
        config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"

        if config_path.exists():
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides for cache directories."""
        # HF_HOME override - only set from config if the path is valid/writable
        if "HF_HOME" in os.environ:
            self._config.setdefault("model", {})["cache_dir"] = os.environ["HF_HOME"]
        elif "HF_HOME" not in os.environ and "model" in self._config and "cache_dir" in self._config["model"]:
            # Validate the config path is usable before setting as env var
            cache_dir = self._config["model"]["cache_dir"]
            if self._is_path_usable(cache_dir):
                os.environ["HF_HOME"] = cache_dir

        # TORCH_HOME override - only set from config if the path is valid/writable
        if "TORCH_HOME" in os.environ:
            self._config.setdefault("model", {})["torch_cache_dir"] = os.environ["TORCH_HOME"]
        elif "TORCH_HOME" not in os.environ and "model" in self._config and "torch_cache_dir" in self._config["model"]:
            torch_cache_dir = self._config["model"]["torch_cache_dir"]
            if self._is_path_usable(torch_cache_dir):
                os.environ["TORCH_HOME"] = torch_cache_dir

    def _is_path_usable(self, path: str) -> bool:
        """Check if a path is usable (exists or can be created)."""
        return _is_path_usable_impl(path)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with dot notation (e.g., 'model.cache_dir')."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self._config.get("model", {})

    def get_corruption_config(self) -> Dict[str, Any]:
        """Get corruption configuration section."""
        return self._config.get("corruptions", {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration section."""
        return self._config.get("evaluation", {})

    def get_video_config(self) -> Dict[str, Any]:
        """Get video configuration section."""
        return self._config.get("video", {})

    def get_classes(self) -> Dict[int, str]:
        """Get class names mapping."""
        classes = self._config.get("classes", {})
        return {int(k): v for k, v in classes.items()}

    def get_palette(self) -> Dict[int, tuple]:
        """Get color palette mapping."""
        palette = self._config.get("palette", {})
        return {int(k): tuple(v) for k, v in palette.items()}


# Lazy config instance (singleton)
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global Config instance (lazy initialization)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def get_cache_dir() -> str:
    """Get cache directory with fallback priority:
    1. HF_HOME environment variable
    2. Config file model.cache_dir (if usable)
    3. Default ~/.cache/huggingface
    """
    if "HF_HOME" in os.environ:
        return os.environ["HF_HOME"]

    cache_dir = get_config().get("model.cache_dir")
    if cache_dir:
        # Expand ~ to home directory
        cache_dir = os.path.expanduser(cache_dir)
        if _is_path_usable_global(cache_dir):
            return cache_dir

    # Default to user cache directory
    default_cache = str(Path.home() / ".cache" / "huggingface")
    return default_cache


def _is_path_usable_impl(path: str) -> bool:
    """Check if a path is usable (exists or can be created)."""
    try:
        p = Path(path)
        # Check if parent directory exists and is writable
        parent = p.parent if p.suffix else p
        parent.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = parent / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True
    except (OSError, PermissionError):
        return False


def _is_path_usable_global(path: str) -> bool:
    """Check if a path is usable (exists or can be created)."""
    return _is_path_usable_impl(path)


def get_torch_cache_dir() -> str:
    """Get torch cache directory with fallback priority."""
    if "TORCH_HOME" in os.environ:
        return os.environ["TORCH_HOME"]

    cache_dir = get_config().get("model.torch_cache_dir")
    if cache_dir:
        # Expand ~ to home directory
        cache_dir = os.path.expanduser(cache_dir)
        if _is_path_usable_global(cache_dir):
            return cache_dir

    # Default to user cache directory
    default_cache = str(Path.home() / ".cache" / "torch")
    return default_cache


def setup_cache_dirs():
    """Setup cache directories and environment variables."""
    hf_cache = get_cache_dir()
    torch_cache = get_torch_cache_dir()

    os.environ["HF_HOME"] = hf_cache
    os.environ["TORCH_HOME"] = torch_cache

    # Create directories if they don't exist
    Path(hf_cache).mkdir(parents=True, exist_ok=True)
    Path(torch_cache).mkdir(parents=True, exist_ok=True)

    return hf_cache, torch_cache


# Convenience function for lazy config access
def get_model_config() -> Dict[str, Any]:
    """Get model configuration section."""
    return get_config().get_model_config()


def get_corruption_config() -> Dict[str, Any]:
    """Get corruption configuration section."""
    return get_config().get_corruption_config()


def get_evaluation_config() -> Dict[str, Any]:
    """Get evaluation configuration section."""
    return get_config().get_evaluation_config()


def get_video_config() -> Dict[str, Any]:
    """Get video configuration section."""
    return get_config().get_video_config()