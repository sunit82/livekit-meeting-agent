# model_factory.py
from abc import ABC, abstractmethod
import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings


class BaseModel(ABC):
    """Abstract base class for model providers."""

    @abstractmethod
    def fetch_llm(self):
        """Return an LLM instance."""
        pass

    @abstractmethod
    def fetch_embeddings(self):
        """Return an embeddings instance."""
        pass


class GoogleModel(BaseModel):
    """Google Generative AI model provider."""

    def fetch_llm(self):
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.8,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    def fetch_embeddings(self):
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )


class OllamaModel(BaseModel):
    """Ollama local model provider."""

    def fetch_llm(self):
        return ChatOllama(
            model="qwen2.5",
            temperature=0.8,
            base_url="http://localhost:11434",
        )

    def fetch_embeddings(self):
        return OllamaEmbeddings(model="embeddinggemma")


class ModelFactory:
    """Factory that creates the appropriate model provider based on SELECTED_MODEL env variable."""

    @staticmethod
    def create() -> BaseModel:
        selected = os.getenv("SELECTED_MODEL", "ollama").lower()
        if selected == "google":
            return GoogleModel()
        elif selected == "ollama":
            return OllamaModel()
        else:
            raise ValueError(
                f"Unknown SELECTED_MODEL: '{selected}'. Supported values: 'google', 'ollama'."
            )
