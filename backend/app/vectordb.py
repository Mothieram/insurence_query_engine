import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, Tuple, Set
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import hashlib
import pickle
from pymongo import MongoClient
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

class InsuranceVectorDB:
    """
    Enhanced vector database system for insurance documents with:
    - MongoDB integration for embedding storage
    - Parallel processing
    - Intelligent caching
    - Advanced metadata handling
    - Hybrid search capabilities
    """
    
    def __init__(self, embedding_model: Embeddings):
        """Initialize with an embedding model and enhanced features."""
        self.embedding_model = embedding_model
        self.db: Optional[FAISS] = None
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # MongoDB setup
        load_dotenv()
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MONGO_URI not found in environment variables")
        self.mongo_client = MongoClient(mongo_uri)
        self.embeddings_col = self.mongo_client["insurance_documents"]["embeddings"]
        
        self._initialize_logging()
        self._load_cache()

    def __len__(self) -> int:
        """Return the number of clauses in the database"""
        if not self.db:
            return 0
        return len(self.db.index_to_docstore_id)

    def _initialize_logging(self):
        """Configure advanced logging for the vector database"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('insurance_vectordb.log'),
                logging.StreamHandler()
            ]
        )

    def _load_cache(self, cache_path: str = "embedding_cache.pkl") -> None:
        """Load embedding cache from disk"""
        try:
            with open(cache_path, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            logger.info(f"Loaded embedding cache with {len(self.embedding_cache)} entries")
        except (FileNotFoundError, EOFError, pickle.PickleError) as e:
            logger.warning(f"Could not load cache: {str(e)}")
            self.embedding_cache = {}

    def _save_cache(self, cache_path: str = "embedding_cache.pkl") -> None:
        """Save embedding cache to disk"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved embedding cache with {len(self.embedding_cache)} entries")
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to save cache: {str(e)}")

    def _get_cache_key(self, text: str) -> str:
        """Generate consistent cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _process_file(self, file: Path) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        """Process a single file using embeddings from MongoDB"""
        texts: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "clauses" not in data:
                logger.warning(f"No clauses found in {file.name}, skipping")
                return texts, embeddings, metadatas

            for clause in data["clauses"]:
                text = clause.get("text", "").strip()
                if not text:
                    continue
                    
                # Get embedding from cache or MongoDB
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    embedding = self.embedding_cache[cache_key]
                else:
                    # Try to find embedding in MongoDB
                    emb_record = self.embeddings_col.find_one({"clause_hash": cache_key})
                    if emb_record:
                        embedding = emb_record["embedding"]
                        self.embedding_cache[cache_key] = embedding
                    else:
                        continue

                texts.append(text)
                embeddings.append(embedding)
                metadatas.append({
                    "source": data.get("metadata", {}).get("source_file", file.name),
                    "section": clause.get("type", "GENERAL").upper(),
                    "keywords": ", ".join(clause.get("keywords", [])),
                    "policy_number": data.get("metadata", {}).get("policy_number", ""),
                    "effective_date": data.get("metadata", {}).get("effective_date", ""),
                    "original_file": file.name,
                    "clause_hash": cache_key
                })

            logger.info(f"Processed {file.name}")
            return texts, embeddings, metadatas

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            return texts, embeddings, metadatas

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.db:
            return {
                "status": "not_initialized",
                "clause_count": 0,
                "policy_count": 0,
                "sections": {}
            }
        
        return {
            "status": "active",
            "clause_count": len(self),
            "policy_count": len(self.get_all_policies()),
            "sections": self._get_section_distribution()
        }

    def _get_section_distribution(self) -> Dict[str, int]:
        """Get distribution of clause types"""
        if not self.db:
            return {}
            
        sections: Dict[str, int] = {}
        for doc in self.db.docstore._dict.values():
            section = doc.metadata.get("section", "UNKNOWN")
            sections[section] = sections.get(section, 0) + 1
        return sections

    def create_from_embedded_files(self, embedded_dir: str, parallel: bool = True) -> None:
        """
        Create vector database from embedded JSON files with parallel processing.
        
        Args:
            embedded_dir: Directory containing _embedded.json files
            parallel: Whether to use parallel processing
            
        Raises:
            FileNotFoundError: If no embedded files found
            ValueError: If invalid data structure in files
        """
        embedded_files = list(Path(embedded_dir).glob("*_embedded.json"))
        
        if not embedded_files:
            error_msg = f"No embedded files found in {embedded_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        all_texts: List[str] = []
        all_embeddings: List[List[float]] = []
        all_metadatas: List[Dict[str, Any]] = []

        if parallel:
            with ThreadPoolExecutor() as executor:
                results = list(tqdm(
                    executor.map(self._process_file, embedded_files),
                    total=len(embedded_files),
                    desc="Processing files"
                ))
                for texts, embeddings, metadatas in results:
                    all_texts.extend(texts)
                    all_embeddings.extend(embeddings)
                    all_metadatas.extend(metadatas)
        else:
            for file in tqdm(embedded_files, desc="Processing files"):
                texts, embeddings, metadatas = self._process_file(file)
                all_texts.extend(texts)
                all_embeddings.extend(embeddings)
                all_metadatas.extend(metadatas)

        if not all_texts:
            error_msg = "No valid clauses found in any files"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Processed {len(embedded_files)} files with {len(all_texts)} clauses")

        try:
            self.db = FAISS.from_embeddings(
                text_embeddings=list(zip(all_texts, all_embeddings)),
                embedding=self.embedding_model,
                metadatas=all_metadatas
            )
            logger.info("Vector database created successfully")
            self._save_cache()
        except Exception as e:
            logger.error(f"Failed to create vector database: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """
        Save vector database to disk with validation.
        
        Args:
            path: Directory path to save the database
            
        Raises:
            ValueError: If database not initialized
            IOError: If save operation fails
        """
        if not self.db:
            error_msg = "Database not initialized"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.db.save_local(folder_path=path)
            logger.info(f"Database saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save database: {str(e)}")
            raise IOError(f"Save failed: {str(e)}")

    def load(self, path: str) -> None:
        """
        Load existing vector database from disk with validation.
        
        Args:
            path: Directory path containing the saved database
            
        Raises:
            IOError: If load operation fails
        """
        required_files = ['index.faiss', 'index.pkl']
        if not all((Path(path) / f).exists() for f in required_files):
            raise IOError(f"Missing required database files in {path}")

        try:
            self.db = FAISS.load_local(
                folder_path=path,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Database loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load database: {str(e)}")
            raise IOError(f"Load failed: {str(e)}")

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        hybrid: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Enhanced semantic search with multiple features.
        
        Args:
            query: Search query text
            k: Number of results to return
            filters: Metadata filters
            threshold: Minimum similarity score (0-1)
            hybrid: Whether to use hybrid keyword/semantic search
            
        Returns:
            List of results with text, metadata, and score
        """
        if not self.db:
            raise ValueError("Database not initialized")

        try:
            filter_func = self._create_filter_func(filters) if filters else None

            if threshold:
                results = self.db.similarity_search_with_relevance_scores(
                    query=query,
                    k=k*3 if hybrid else k,
                    filter=filter_func
                )
                results = self._process_results(results, threshold, hybrid)
            else:
                results = self.db.similarity_search(
                    query=query,
                    k=k*3 if hybrid else k,
                    filter=filter_func
                )
                results = self._process_results([(r, None) for r in results], None, hybrid)[:k]

            # Enrich with MongoDB data
            enriched_results = []
            for result in results:
                if 'clause_hash' in result['metadata']:
                    mongo_data = self.embeddings_col.find_one(
                        {"clause_hash": result['metadata']['clause_hash']},
                        {"document_id": 1, "clause_id": 1}
                    )
                    if mongo_data:
                        result['mongo_ref'] = {
                            "document_id": mongo_data.get("document_id"),
                            "clause_id": mongo_data.get("clause_id")
                        }
                enriched_results.append(result)
            
            return enriched_results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise ValueError(f"Search error: {str(e)}")

    def _create_filter_func(self, filters: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
        """Create filter function from metadata filters"""
        def filter_func(metadata: Dict[str, Any]) -> bool:
            for key, value in filters.items():
                if key not in metadata:
                    return False
                if isinstance(value, list):
                    if str(metadata[key]).lower() not in [str(v).lower() for v in value]:
                        return False
                else:
                    if str(metadata[key]).lower() != str(value).lower():
                        return False
            return True
        return filter_func

    def _process_results(
        self,
        results: List[Tuple[Document, Optional[float]]],
        threshold: Optional[float],
        hybrid: bool
    ) -> List[Dict[str, Any]]:
        """Process and optionally re-rank search results"""
        processed: List[Dict[str, Any]] = []
        for doc, score in results:
            if threshold is not None and score is not None and score < threshold:
                continue
                
            processed.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "keywords": doc.metadata.get("keywords", "").split(", ")
            })

        if hybrid:
            processed = self._hybrid_rerank(processed)

        return processed[:len(results)]

    def _hybrid_rerank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple hybrid re-ranking based on keyword matches"""
        if not results or 'keywords' not in results[0]:
            return results
            
        for result in results:
            if result['score'] is not None:
                keyword_boost = min(0.1, len(result['keywords']) * 0.01)
                result['score'] += keyword_boost
                
        return sorted(results, key=lambda x: x['score'] or 0, reverse=True)

    def get_all_policies(self) -> List[str]:
        """Get unique policy numbers with caching"""
        if not self.db:
            logger.warning("Database not initialized")
            return []

        try:
            policies: Set[str] = set()
            for doc in self.db.docstore._dict.values():
                if policy_num := doc.metadata.get("policy_number"):
                    policies.add(str(policy_num).strip())
            return sorted(list(policies))
        except Exception as e:
            logger.error(f"Failed to get policies: {str(e)}")
            return []