from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProfilerConfig:
    data_dir: Path = Path("./data")
    db_path: Path = Path("./data/profiler.db")
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:0.5b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: int = 60
    embedding_model: str = "all-MiniLM-L6-v2"
    project_summary_questionnaire_path: Path = Path("./data/project_summary_questionnaire.txt")
    max_text_chars_for_llm: int = 12000
    default_chunk_size: int = 1200
    default_chunk_overlap: int = 150

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
