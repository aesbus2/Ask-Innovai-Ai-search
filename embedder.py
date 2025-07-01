"""
embedder.py - Transformer-Based Text Embedding Service
PERFORMANCE OPTIMIZED: Fixes slow initial search with model preloading and better caching
Endpoint structure: {id, url, program, name, lob, updated, content}
"""

import os
import logging
import hashlib
import pickle
import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import time
import json
import threading
from functools import lru_cache

# Configuration with SAFE DEFAULTS
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "./embedding_cache")
EMBEDDING_CACHE_ENABLED = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
EMBEDDING_MAX_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))  # Increased for better performance
EMBEDDING_LRU_CACHE_SIZE = int(os.getenv("EMBEDDING_LRU_CACHE_SIZE", "10000"))  # In-memory cache

# Default fallback vector for blank or invalid inputs
EMPTY_VECTOR = [0.0] * EMBEDDING_DIMENSION

# Setup logging
logger = logging.getLogger(__name__)

class TransformerEmbeddingService:
    """
    PERFORMANCE OPTIMIZED: Transformer-based text embedding service with preloading and better caching
    """
    
    def __init__(self, cache_dir: str = None, enable_cache: bool = None, 
                 provider: str = None, model_name: str = None):
        self.cache_dir = Path(cache_dir or EMBEDDING_CACHE_DIR)
        self.enable_cache = enable_cache if enable_cache is not None else EMBEDDING_CACHE_ENABLED
        self.embedding_dimension = EMBEDDING_DIMENSION
        self.provider = provider or EMBEDDING_PROVIDER
        self.model_name = model_name or EMBEDDING_MODEL
        self.max_length = EMBEDDING_MAX_LENGTH
        self.batch_size = EMBEDDING_BATCH_SIZE
        
        # PERFORMANCE: Thread-safe model loading
        self.model = None
        self.model_loaded = False
        self.model_loading = False
        self.load_lock = threading.Lock()
        
        # Cache management settings
        self.cache_clear_interval = int(os.getenv("CACHE_CLEAR_INTERVAL", "2000"))
        self.max_cache_size_mb = int(os.getenv("MAX_CACHE_SIZE_MB", "1000"))
        self.documents_processed = 0
        
        # PERFORMANCE: Enhanced statistics tracking
        self.embedding_stats = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "lru_cache_hits": 0,
            "lru_cache_misses": 0,
            "file_cache_hits": 0,
            "file_cache_misses": 0,
            "total_processing_time": 0.0,
            "model_load_time": 0.0,
            "avg_embedding_time": 0.0
        }
        
        # Validate critical settings
        if not self.provider:
            logger.warning("No embedding provider set, defaulting to 'local'")
            self.provider = "local"
        
        if not self.model_name:
            logger.warning("No embedding model set, defaulting to sentence-transformers/all-MiniLM-L6-v2")
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Initialize the embedding backend
        self.backend = self.provider
        
        # Create cache directory if caching is enabled
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache enabled at: {self.cache_dir}")
        
        # PERFORMANCE: Don't initialize backend in __init__ to avoid blocking
        # Will be initialized on first use or explicit preload
        logger.info(f"Embedding service configured (provider: {self.provider}, model: {self.model_name})")
    
    def preload(self):
        """PERFORMANCE: Preload the model during app startup to avoid delays"""
        logger.info("🔥 Preloading embedding model for faster searches...")
        load_start = time.time()
        
        try:
            self._ensure_model_loaded()
            
            # Warm up with test embeddings
            warmup_texts = [
                "customer service question",
                "account management",
                "technical support",
                "billing inquiry"
            ]
            
            logger.info("🔥 Warming up with test embeddings...")
            for text in warmup_texts:
                self.embed_single(text)
            
            load_time = time.time() - load_start
            self.embedding_stats["model_load_time"] = load_time
            
            logger.info(f"✅ Embedding model preloaded successfully in {load_time:.2f}s")
            logger.info(f"📊 Model ready for fast searches (dimension: {self.embedding_dimension})")
            
        except Exception as e:
            logger.error(f"❌ Model preloading failed: {e}")
            raise
    
    def _ensure_model_loaded(self):
        """PERFORMANCE: Thread-safe model loading with proper error handling"""
        if self.model_loaded:
            return
        
        with self.load_lock:
            if self.model_loaded:  # Double-check pattern
                return
            
            if self.model_loading:
                # Another thread is loading, wait for it
                while self.model_loading and not self.model_loaded:
                    time.sleep(0.1)
                return
            
            self.model_loading = True
            
            try:
                logger.info(f"🔄 Loading embedding model: {self.model_name}")
                load_start = time.time()
                
                if self.provider == "openai":
                    self._initialize_openai()
                else:
                    self._initialize_local()
                
                load_time = time.time() - load_start
                self.embedding_stats["model_load_time"] = load_time
                self.model_loaded = True
                
                logger.info(f"✅ Model loaded in {load_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                # Fallback to local if OpenAI fails
                if self.provider == "openai":
                    logger.info("🔄 Falling back to local sentence-transformers")
                    self.provider = "local"
                    self.backend = "local"
                    self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    self._initialize_local()
                    self.model_loaded = True
                else:
                    raise
            finally:
                self.model_loading = False
    
    def _initialize_openai(self):
        """Initialize OpenAI embeddings backend"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable required for OpenAI provider")
        
        try:
            import openai
            self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            # Test the client
            test_response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input="test",
                dimensions=self.embedding_dimension
            )
            
            logger.info(f"OpenAI embeddings initialized (model: text-embedding-3-small, dimensions: {self.embedding_dimension})")
            self.model_name = "text-embedding-3-small"
            
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {e}")
    
    def _initialize_local(self):
        """Initialize local sentence-transformers backend"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load the model
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Check if we need to adjust dimensions
            model_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model native dimension: {model_dim}, target dimension: {self.embedding_dimension}")
            
            if model_dim != self.embedding_dimension:
                logger.warning(f"Model dimension ({model_dim}) != target dimension ({self.embedding_dimension}). Embeddings will be truncated/padded.")
            
            logger.info(f"Local sentence-transformers model loaded: {self.model_name}")
            
        except ImportError:
            raise ImportError("sentence-transformers package not installed. Run: pip install sentence-transformers")
        except Exception as e:
            raise Exception(f"Failed to load model {self.model_name}: {e}")
    
    @lru_cache(maxsize=EMBEDDING_LRU_CACHE_SIZE)
    def _cached_embed_single(self, text: str) -> tuple:
        """PERFORMANCE: LRU cached single embedding for frequently accessed texts"""
        if not text or not text.strip():
            return tuple([0.0] * self.embedding_dimension)
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        try:
            # Generate embedding based on provider
            if self.provider == "openai":
                embeddings = self._embed_with_openai([text])
                embedding = embeddings[0]
            else:
                embeddings = self._embed_with_local([text])
                embedding = embeddings[0]
            
            return tuple(embedding.astype(np.float32))
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {text[:50]}...: {e}")
            return tuple([0.0] * self.embedding_dimension)
    
    def _get_cache_path(self, text: str) -> Path:
        """Generate cache file path for given text"""
        cache_key = f"{self.provider}_{self.model_name}_{text}_{self.embedding_dimension}"
        text_hash = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{text_hash}.pkl"
    
    def _load_from_file_cache(self, text: str) -> Optional[np.ndarray]:
        """Load embedding from file cache if available"""
        if not self.enable_cache:
            return None
        
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    self.embedding_stats["file_cache_hits"] += 1
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache {cache_path}: {e}")
                cache_path.unlink(missing_ok=True)
        
        self.embedding_stats["file_cache_misses"] += 1
        return None
    
    def _save_to_file_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to file cache"""
        if not self.enable_cache:
            return
        
        cache_path = self._get_cache_path(text)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache {cache_path}: {e}")
    
    def _preprocess_text(self, text: Union[str, dict]) -> str:
        """Preprocess text with improved content handling for endpoint data"""
        if isinstance(text, dict):
            # Handle structured data from endpoint - use correct field names
            if 'content' in text:  # Correct endpoint field name (lowercase)
                text = text['content']
            elif 'name' in text:   # Correct endpoint field name (lowercase)
                text = text['name']
            else:
                text = json.dumps(text)
        elif not isinstance(text, str):
            text = str(text)

        # Skip meaningless content
        if text.strip() in ["[empty document]", "", "null", "undefined"]:
            return ""

        # Clean and normalize text
        text = text.strip()
        
        # More accurate token counting for different models
        if self.model_name == "sentence-transformers/all-MiniLM-L6-v2":
            # Rough approximation: 1 token ≈ 4 characters for English text
            estimated_tokens = len(text) // 4
            if estimated_tokens > self.max_length:
                logger.debug(f"Text has ~{estimated_tokens} estimated tokens, may exceed {self.max_length} token limit")
                # Optionally truncate to approximate token limit
                if estimated_tokens > self.max_length * 1.5:  # 50% over limit
                    max_chars = self.max_length * 4  # Convert back to characters
                    text = text[:max_chars]
                    logger.debug(f"Text truncated to ~{self.max_length} tokens ({max_chars} characters)")
        
        elif "openai" in self.provider.lower():
            # OpenAI models generally handle longer contexts
            estimated_tokens = len(text) // 4
            openai_limit = 8192  # Most OpenAI embedding models
            if estimated_tokens > openai_limit:
                logger.warning(f"Text has ~{estimated_tokens} estimated tokens, which may exceed OpenAI embedding limits")

        return text
    
    def _adjust_embedding_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """Adjust embedding to target dimension"""
        current_dim = len(embedding)
        
        if current_dim == self.embedding_dimension:
            return embedding
        elif current_dim > self.embedding_dimension:
            # Truncate to target dimension
            return embedding[:self.embedding_dimension]
        else:
            # Pad with zeros to reach target dimension
            padding = np.zeros(self.embedding_dimension - current_dim, dtype=embedding.dtype)
            return np.concatenate([embedding, padding])
    
    def _embed_with_openai(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
                dimensions=self.embedding_dimension
            )
            
            embeddings = []
            for data in response.data:
                embedding = np.array(data.embedding, dtype=np.float32)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    def _embed_with_local(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using local sentence-transformers"""
        try:
            # Generate embeddings with optimized settings
            embeddings = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 50,  # Only show progress for large batches
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity
            )
            
            # Adjust dimensions if needed
            adjusted_embeddings = []
            for embedding in embeddings:
                if len(embedding.shape) == 1:  # 1D embedding
                    adjusted = self._adjust_embedding_dimension(embedding.astype(np.float32))
                    adjusted_embeddings.append(adjusted)
                else:  # Handle batch dimension
                    for emb in embedding:
                        adjusted = self._adjust_embedding_dimension(emb.astype(np.float32))
                        adjusted_embeddings.append(adjusted)
            
            return adjusted_embeddings[:len(texts)]  # Ensure we return the right number
            
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """
        PERFORMANCE OPTIMIZED: Generate embedding for a single text with multi-level caching
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        start_time = time.time()
        text = self._preprocess_text(text)
        
        if not text:
            logger.debug("Empty text provided, returning zero vector")
            return [0.0] * self.embedding_dimension
        
        # PERFORMANCE: Try LRU cache first (fastest)
        try:
            result = self._cached_embed_single(text)
            self.embedding_stats["lru_cache_hits"] += 1
            self.embedding_stats["cache_hits"] += 1
            
            processing_time = time.time() - start_time
            self.embedding_stats["total_processing_time"] += processing_time
            
            return list(result)
            
        except Exception:
            self.embedding_stats["lru_cache_misses"] += 1
            self.embedding_stats["cache_misses"] += 1
        
        # PERFORMANCE: Try file cache second
        cached_embedding = self._load_from_file_cache(text)
        if cached_embedding is not None:
            processing_time = time.time() - start_time
            self.embedding_stats["total_processing_time"] += processing_time
            
            # Also cache in LRU for next time
            try:
                self._cached_embed_single.__wrapped__(self, text)  # Cache the result
            except:
                pass
            
            return cached_embedding.tolist()
        
        # Generate new embedding
        self._ensure_model_loaded()
        
        try:
            # Generate embedding based on provider
            if self.provider == "openai":
                embeddings = self._embed_with_openai([text])
                embedding = embeddings[0]
            else:
                embeddings = self._embed_with_local([text])
                embedding = embeddings[0]
            
            # Cache the result in file cache
            self._save_to_file_cache(text, embedding)
            
            # Update stats
            self.embedding_stats["total_embeddings"] += 1
            processing_time = time.time() - start_time
            self.embedding_stats["total_processing_time"] += processing_time
            
            # Calculate running average
            if self.embedding_stats["total_embeddings"] > 0:
                self.embedding_stats["avg_embedding_time"] = (
                    self.embedding_stats["total_processing_time"] / 
                    self.embedding_stats["total_embeddings"]
                )
            
            # Increment document counter for cache management
            self._increment_document_counter()
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {text[:50]}...: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        PERFORMANCE OPTIMIZED: Generate embeddings for multiple texts efficiently with caching
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        start_time = time.time()
        
        # Preprocess all texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # PERFORMANCE: Check both LRU and file cache for each text
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(processed_texts):
            if not text:
                results.append([0.0] * self.embedding_dimension)
                continue
            
            # Try LRU cache first
            try:
                cached_result = self._cached_embed_single(text)
                results.append(list(cached_result))
                self.embedding_stats["lru_cache_hits"] += 1
                continue
            except:
                self.embedding_stats["lru_cache_misses"] += 1
            
            # Try file cache
            cached = self._load_from_file_cache(text)
            if cached is not None:
                results.append(cached.tolist())
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts in batches
        if uncached_texts:
            self._ensure_model_loaded()
            
            try:
                # Process in batches for better memory management
                batch_size = self.batch_size
                for batch_start in range(0, len(uncached_texts), batch_size):
                    batch_end = min(batch_start + batch_size, len(uncached_texts))
                    batch_texts = uncached_texts[batch_start:batch_end]
                    batch_indices = uncached_indices[batch_start:batch_end]
                    
                    if self.provider == "openai":
                        new_embeddings = self._embed_with_openai(batch_texts)
                    else:
                        new_embeddings = self._embed_with_local(batch_texts)
                    
                    # Fill in results and cache
                    for idx, embedding in zip(batch_indices, new_embeddings):
                        results[idx] = embedding.tolist()
                        self._save_to_file_cache(processed_texts[idx], embedding)
                        
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Fill failed embeddings with zero vectors
                for idx in uncached_indices:
                    if results[idx] is None:
                        results[idx] = [0.0] * self.embedding_dimension
        
        # Update stats
        self.embedding_stats["total_embeddings"] += len(texts)
        processing_time = time.time() - start_time
        self.embedding_stats["total_processing_time"] += processing_time
        
        # Calculate running average
        if self.embedding_stats["total_embeddings"] > 0:
            self.embedding_stats["avg_embedding_time"] = (
                self.embedding_stats["total_processing_time"] / 
                self.embedding_stats["total_embeddings"]
            )
        
        # Increment document counter for cache management
        self._increment_document_counter()
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.embedding_dimension
    
    def get_stats(self) -> dict:
        """PERFORMANCE: Get comprehensive embedding service statistics"""
        total_cache_requests = self.embedding_stats["cache_hits"] + self.embedding_stats["cache_misses"]
        cache_hit_rate = 0.0
        if total_cache_requests > 0:
            cache_hit_rate = self.embedding_stats["cache_hits"] / total_cache_requests
        
        lru_requests = self.embedding_stats["lru_cache_hits"] + self.embedding_stats["lru_cache_misses"]
        lru_hit_rate = 0.0
        if lru_requests > 0:
            lru_hit_rate = self.embedding_stats["lru_cache_hits"] / lru_requests
        
        file_requests = self.embedding_stats["file_cache_hits"] + self.embedding_stats["file_cache_misses"]
        file_hit_rate = 0.0
        if file_requests > 0:
            file_hit_rate = self.embedding_stats["file_cache_hits"] / file_requests
        
        # Get LRU cache info
        lru_cache_info = {}
        try:
            cache_info = self._cached_embed_single.cache_info()
            lru_cache_info = {
                "size": cache_info.currsize,
                "maxsize": cache_info.maxsize,
                "hits": cache_info.hits,
                "misses": cache_info.misses
            }
        except:
            pass
        
        return {
            **self.embedding_stats,
            "model_loaded": self.model_loaded,
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "lru_cache_hit_rate": f"{lru_hit_rate:.2%}",
            "file_cache_hit_rate": f"{file_hit_rate:.2%}",
            "avg_processing_time_seconds": self.embedding_stats.get("avg_embedding_time", 0),
            "documents_processed": self.documents_processed,
            "cache_size_mb": self.get_cache_size_mb(),
            "lru_cache_info": lru_cache_info,
            "provider": self.provider,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension
        }
    
    def _check_cache_limits(self):
        """Check and manage cache size and document count limits"""
        if not self.enable_cache:
            return
            
        # Check document count limit
        if self.documents_processed >= self.cache_clear_interval:
            logger.info(f"Processed {self.documents_processed} documents, clearing cache...")
            self.clear_cache()
            self.documents_processed = 0
            return
            
        # Check cache size limit
        try:
            cache_size_mb = self.get_cache_size_mb()
            if cache_size_mb > self.max_cache_size_mb:
                logger.info(f"Cache size ({cache_size_mb:.1f}MB) exceeds limit ({self.max_cache_size_mb}MB), clearing cache...")
                self.clear_cache()
                self.documents_processed = 0
        except Exception as e:
            logger.warning(f"Failed to check cache size: {e}")
    
    def get_cache_size_mb(self) -> float:
        """Get current cache size in MB"""
        if not self.cache_dir.exists():
            return 0.0
            
        total_size = 0
        for file_path in self.cache_dir.rglob("*.pkl"):
            try:
                total_size += file_path.stat().st_size
            except (OSError, FileNotFoundError):
                continue
                
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _increment_document_counter(self):
        """Increment document counter and check limits"""
        self.documents_processed += 1
        
        # Check limits every 100 documents to avoid too frequent checks
        if self.documents_processed % 100 == 0:
            self._check_cache_limits()
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        # Clear LRU cache
        try:
            self._cached_embed_single.cache_clear()
            logger.info("LRU cache cleared")
        except:
            pass
        
        # Clear file cache
        if not self.enable_cache:
            logger.info("File caching is disabled, nothing to clear")
            return
        
        try:
            import shutil
            if self.cache_dir.exists():
                # Get cache size before clearing
                cache_size = self.get_cache_size_mb()
                
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"File cache cleared ({cache_size:.1f}MB freed)")
                
                # Reset counter and stats
                self.documents_processed = 0
                self.embedding_stats["cache_hits"] = 0
                self.embedding_stats["cache_misses"] = 0
                self.embedding_stats["file_cache_hits"] = 0
                self.embedding_stats["file_cache_misses"] = 0
        except Exception as e:
            logger.error(f"Failed to clear file cache: {e}")

# Maintain backward compatibility with existing interface
SimpleEmbeddingService = TransformerEmbeddingService

# Global embedding service instance
_embedding_service = None

def get_embedding_service() -> TransformerEmbeddingService:
    """Get or create the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = TransformerEmbeddingService()
    return _embedding_service

def preload_embedding_model():
    """PERFORMANCE: Preload the embedding model for faster first searches"""
    service = get_embedding_service()
    service.preload()

def embed_text(text: str) -> List[float]:
    """
    Convenience function to embed a single text
    
    Args:
        text: Input text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    service = get_embedding_service()
    return service.embed_single(text)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convenience function to embed multiple texts
    
    Args:
        texts: List of input texts to embed
        
    Returns:
        List of embedding vectors
    """
    service = get_embedding_service()
    return service.embed_batch(texts)

def get_embedding_dimension() -> int:
    """Get the dimension of embeddings"""
    service = get_embedding_service()
    return service.get_embedding_dimension()

def get_empty_vector() -> List[float]:
    """Return an all-zero fallback embedding"""
    return EMPTY_VECTOR

def get_embedding_stats() -> dict:
    """Get embedding service statistics"""
    service = get_embedding_service()
    return service.get_stats()

def health_check() -> dict:
    """
    Health check for the embedding service
    
    Returns:
        Dictionary with health status and info
    """
    try:
        service = get_embedding_service()
        test_embedding = service.embed_single("health check test")
        stats = service.get_stats()
        
        return {
            "status": "healthy",
            "model": service.model_name,
            "backend": service.backend,
            "provider": service.provider,
            "dimension": service.embedding_dimension,
            "cache_enabled": service.enable_cache,
            "test_embedding_length": len(test_embedding),
            "model_loaded": service.model_loaded,
            "performance_stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    # Test the embedding service
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    print(" Testing PERFORMANCE OPTIMIZED Embedding Service...")
    print("Expected endpoint fields: id, url, program, name, lob, updated, content")
    
    # Test preloading
    print("\nTesting model preloading...")
    start_time = time.time()
    preload_embedding_model()
    preload_time = time.time() - start_time
    print(f" Preloading completed in {preload_time:.2f}s")
    
    # Health check
    health = health_check()
    print(f"\n Health check: {health['status']}")
    
    if health["status"] == "healthy":
        print(f"   Model: {health['model']}")
        print(f"   Provider: {health['provider']}")
        print(f"   Dimensions: {health['dimension']}")
        print(f"   Model Loaded: {health['model_loaded']}")
        
        # Test single embedding (should be fast now)
        print("\n Testing single embedding performance...")
        start_time = time.time()
        test_content = "This is a test document about Metro incentive programs for customer retention."
        embedding = embed_text(test_content)
        single_time = time.time() - start_time
        
        print(f"   Single embedding: {len(embedding)} dimensions in {single_time:.4f}s")
        print(f"   First 5 values: {[round(x, 4) for x in embedding[:5]]}")
        
        # Test cached embedding (should be even faster)
        start_time = time.time()
        embedding_cached = embed_text(test_content)  # Same text
        cached_time = time.time() - start_time
        
        print(f"   Cached embedding: {cached_time:.4f}s (speedup: {single_time/cached_time:.1f}x)")
        
        # Test batch embedding
        test_documents = [
            "customer service inquiry",
            "technical support request", 
            "billing question",
            "account management",
            test_content  # This should hit LRU cache
        ]
        
        print(f"\n Testing batch embedding ({len(test_documents)} texts)...")
        start_time = time.time()
        batch_embeddings = embed_texts(test_documents)
        batch_time = time.time() - start_time
        
        print(f"   Batch embedding: {len(batch_embeddings)} texts in {batch_time:.4f}s")
        print(f"   Average time per text: {batch_time / len(test_documents):.4f}s")
        
        # Show comprehensive performance stats
        stats = get_embedding_stats()
        print(f"\n Performance Statistics:")
        print(f"   Model loaded: {stats.get('model_loaded', 'Unknown')}")
        print(f"   Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"   Cache hit rate: {stats.get('cache_hit_rate', '0%')}")
        print(f"   LRU cache hit rate: {stats.get('lru_cache_hit_rate', '0%')}")
        print(f"   File cache hit rate: {stats.get('file_cache_hit_rate', '0%')}")
        print(f"   Average embedding time: {stats.get('avg_processing_time_seconds', 0):.4f}s")
        print(f"   Model load time: {stats.get('model_load_time', 0):.2f}s")
        
        if 'lru_cache_info' in stats and stats['lru_cache_info']:
            cache_info = stats['lru_cache_info']
            print(f"   LRU cache size: {cache_info.get('size', 0)}/{cache_info.get('maxsize', 0)}")
        
        print("\n PERFORMANCE OPTIMIZED embedding service is working correctly!")
        print(" Initial searches should now be fast thanks to model preloading!")
        
    else:
        print(f" Health check failed: {health.get('error', 'Unknown error')}")
        print("Fix the embedder before running main.py!")
    
    print("\n🏁 Testing complete!")