import google.generativeai as genai
import json
import os
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import hashlib
import pickle
from sentence_transformers import SentenceTransformer
import backoff

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedInsuranceEmbedder:
    """Advanced embedding system for insurance documents with hybrid embedding support"""
    
    def __init__(self):
        self.configure_gemini()
        self.local_embedder = self._initialize_local_embedder()
        self.embedding_cache = {}
        self._load_cache()
        
    def configure_gemini(self):
        """Configure Gemini API with enhanced error handling"""
        env_path = Path(__file__).parent.parent / '.env'
        
        if not env_path.exists():
            raise FileNotFoundError(f".env file not found at: {env_path}")
            
        load_dotenv(env_path)
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
            
        genai.configure(api_key=api_key)
        logger.info("Gemini configured successfully")
        
    def _initialize_local_embedder(self):
        """Initialize local embedding model as fallback"""
        try:
            model = SentenceTransformer('all-mpnet-base-v2')
            logger.info("Local embedding model loaded")
            return model
        except Exception as e:
            logger.warning(f"Could not load local embedder: {str(e)}")
            return None
    
    def _load_cache(self, cache_path: str = "embedding_cache.pkl"):
        """Load embedding cache from disk"""
        try:
            with open(cache_path, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            logger.info(f"Loaded embedding cache with {len(self.embedding_cache)} entries")
        except (FileNotFoundError, EOFError):
            self.embedding_cache = {}
    
    def _save_cache(self, cache_path: str = "embedding_cache.pkl"):
        """Save embedding cache to disk"""
        with open(cache_path, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
        logger.info(f"Saved embedding cache with {len(self.embedding_cache)} entries")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate consistent cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _get_gemini_embedding(self, text: str) -> List[float]:
        """Get embedding from Gemini API with retry logic"""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Gemini embedding failed: {str(e)}")
            raise
    
    def _get_local_embedding(self, text: str) -> List[float]:
        """Get embedding from local model"""
        if self.local_embedder:
            return self.local_embedder.encode(text).tolist()
        raise RuntimeError("Local embedder not available")
    
    def get_embedding(self, text: str, force_local: bool = False) -> List[float]:
        """Get embedding with hybrid fallback strategy"""
        cache_key = self._get_cache_key(text)
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        try:
            if not force_local:
                embedding = self._get_gemini_embedding(text)
            else:
                raise RuntimeError("Forcing local embedding")
        except Exception as e:
            logger.warning(f"Using local embedder: {str(e)}")
            embedding = self._get_local_embedding(text)
            
        self.embedding_cache[cache_key] = embedding
        return embedding
    
    def batch_embed(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Process batch of texts with parallel execution"""
        embeddings = []
        
        # First check cache
        cached_results = []
        uncached_texts = []
        for text in texts:
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                cached_results.append(self.embedding_cache[cache_key])
            else:
                uncached_texts.append(text)
        
        # Process uncached texts in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i+batch_size]
                futures.append(executor.submit(self._process_batch, batch))
            
            for future in tqdm(futures, desc="Embedding batches"):
                try:
                    embeddings.extend(future.result())
                except Exception as e:
                    logger.error(f"Batch failed: {str(e)}")
                    embeddings.extend([None] * batch_size)
        
        # Combine results maintaining original order
        final_embeddings = []
        cached_idx = uncached_idx = 0
        for text in texts:
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                final_embeddings.append(self.embedding_cache[cache_key])
                cached_idx += 1
            else:
                final_embeddings.append(embeddings[uncached_idx])
                uncached_idx += 1
        
        return final_embeddings
    
    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """Process a single batch with error handling"""
        try:
            if len(batch) > 10:  # Use Gemini for larger batches
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=batch,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                embeddings = result['embedding']
            else:  # Use local for small batches
                embeddings = self.local_embedder.encode(batch).tolist()
            
            # Update cache
            for text, emb in zip(batch, embeddings):
                self.embedding_cache[self._get_cache_key(text)] = emb
            
            return embeddings
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return [self._get_local_embedding(text) for text in batch]
    
    def process_document(self, input_path: Path, output_dir: Path) -> bool:
        """Process a single document with enhanced features"""
        try:
            with open(input_path) as f:
                data = json.load(f)
            
            # Extract and clean clauses
            clauses = []
            for clause in data.get("clauses", []):
                text = clause["text"].strip()
                if len(text) > 20:  # Minimum length threshold
                    clauses.append({
                        "text": text,
                        "type": clause.get("type", "UNKNOWN"),
                        "keywords": clause.get("keywords", []),  # Ensure keywords exist
                        "metadata": clause.get("metadata", {})
                    })
            
            if not clauses:
                logger.warning(f"No valid clauses found in {input_path.name}")
                return False
            
            # Batch embed with progress
            texts = [c["text"] for c in clauses]
            embeddings = []
            for i in tqdm(range(0, len(texts), 100), desc="Processing batches"):
                batch = texts[i:i+100]
                embeddings.extend(self.batch_embed(batch))
                time.sleep(1)  # Rate limiting
                
            # Add embeddings to clauses
            for clause, emb in zip(clauses, embeddings):
                clause["embedding"] = emb
                clause["embedding_model"] = "gemini-001" if not isinstance(emb, type(None)) else "local-mpnet"
            
            # Save results with all required fields
            output_path = output_dir / f"{input_path.stem}_embedded.json"
            with open(output_path, 'w') as f:
                json.dump({
                    "metadata": data.get("metadata", {}),
                    "clauses": clauses,
                    "embedding_stats": {
                        "total_clauses": len(clauses),
                        "gemini_embeddings": sum(1 for c in clauses if c["embedding_model"] == "gemini-001"),
                        "local_embeddings": sum(1 for c in clauses if c["embedding_model"] == "local-mpnet")
                    }
                }, f, indent=2)
            
            logger.info(f"Successfully processed {input_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {input_path.name}: {str(e)}")
            return False

def main():
    try:
        embedder = AdvancedInsuranceEmbedder()
        
        # Configure paths
        BASE_DIR = Path(__file__).parent.parent
        INPUT_DIR = BASE_DIR / "processed_clauses"
        OUTPUT_DIR = BASE_DIR / "embedded_docs"
        
        OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Process files
        json_files = list(INPUT_DIR.glob("*.json"))
        if not json_files:
            logger.error(f"No JSON files found in {INPUT_DIR}")
            return
            
        success_count = 0
        for input_file in tqdm(json_files, desc="Processing files"):
            if embedder.process_document(input_file, OUTPUT_DIR):
                success_count += 1
                
        logger.info(f"\nProcessing complete! Success: {success_count}/{len(json_files)}")
        embedder._save_cache()  # Persist cache
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()