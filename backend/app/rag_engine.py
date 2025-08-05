from typing import List, Dict, Tuple, Optional, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from vectordb import InsuranceVectorDB
import os
from dotenv import load_dotenv
import shutil
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsuranceVectorSystem:
    """Enhanced insurance vector database management system with MongoDB integration"""
    
    def __init__(self):
        """Initialize with embedding model and vector DB"""
        self.embedder = self._initialize_embedder()
        self.vector_db = InsuranceVectorDB(self.embedder)
    
    def _initialize_embedder(self) -> GoogleGenerativeAIEmbeddings:
        """Initialize the Gemini embeddings model with robust error handling"""
        load_dotenv()
        
        if not os.getenv("GEMINI_API_KEY"):
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("Missing API key - please set GEMINI_API_KEY in .env")
        
        try:
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                task_type="retrieval_document"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {str(e)}")
            raise
    
    def _validate_db_directory(self, path: str) -> bool:
        """Enhanced directory validation with detailed checks"""
        required_files = ['index.faiss', 'index.pkl']
        path = Path(path)
        
        if not path.exists():
            return False
            
        missing_files = [f for f in required_files if not (path / f).exists()]
        if missing_files:
            logger.warning(f"Missing database files: {missing_files}")
            return False
            
        return True
    
    def create_or_update_database(self, 
                               db_name: str, 
                               embedded_dirs: List[str],
                               parallel: bool = True) -> Dict[str, Any]:
        """
        Enhanced database creation/update with progress tracking
        
        Args:
            db_name: Name/path of the database
            embedded_dirs: List of directories to process
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary with operation statistics
        """
        stats: Dict[str, Any] = {
            "total_processed": 0,
            "failed_directories": [],
            "clauses_added": 0,
            "existing_clauses": 0
        }
        
        db_path = Path(db_name)
        
        # Handle existing database
        if db_path.exists():
            if self._validate_db_directory(db_name):
                logger.info(f"Loading existing database from {db_name}")
                try:
                    self.vector_db.load(db_name)
                    stats["existing_clauses"] = len(self.vector_db)
                    logger.info(f"Loaded {stats['existing_clauses']} existing clauses")
                    
                    # Update with new documents
                    for dir_path in embedded_dirs:
                        try:
                            logger.info(f"Processing directory: {dir_path}")
                            self.vector_db.create_from_embedded_files(dir_path, parallel)
                            stats["total_processed"] += 1
                            stats["clauses_added"] = len(self.vector_db) - stats["existing_clauses"]
                        except Exception as e:
                            logger.warning(f"Failed to process {dir_path}: {str(e)}")
                            stats["failed_directories"].append(dir_path)
                            continue
                except Exception as e:
                    logger.error(f"Database load failed: {str(e)} - creating new")
                    shutil.rmtree(db_name)
                    db_path.mkdir()
            else:
                logger.warning("Invalid database - recreating")
                shutil.rmtree(db_name)
                db_path.mkdir()
        else:
            db_path.mkdir(parents=True)
        
        # Create new database if needed
        if len(self.vector_db) == 0:
            for dir_path in tqdm(embedded_dirs, desc="Processing directories"):
                try:
                    self.vector_db.create_from_embedded_files(dir_path, parallel)
                    stats["total_processed"] += 1
                    stats["clauses_added"] = len(self.vector_db)
                    break  # Successfully created from first valid directory
                except Exception as e:
                    logger.warning(f"Failed to create from {dir_path}: {str(e)}")
                    stats["failed_directories"].append(dir_path)
                    continue
        
        # Save and return results
        self.vector_db.save(db_name)
        stats["total_clauses"] = len(self.vector_db)
        logger.info(f"Database saved with {stats['total_clauses']} clauses")
        return stats
    
    def search_policies(self, 
                       query: str, 
                       policy_filter: Optional[str] = None,
                       section_filter: Optional[str] = None,
                       threshold: float = 0.7,
                       k: int = 5) -> List[Dict[str, Any]]:
        """
        Enhanced policy search with combined filters
        
        Args:
            query: Search query
            policy_filter: Specific policy number
            section_filter: Clause section type
            threshold: Similarity threshold
            k: Number of results
            
        Returns:
            List of matching clauses with MongoDB references
        """
        filters: Dict[str, Any] = {}
        if policy_filter:
            filters["policy_number"] = policy_filter
        if section_filter:
            filters["section"] = section_filter.upper()
            
        return self.vector_db.search(
            query=query,
            k=k,
            filters=filters if filters else None,
            threshold=threshold,
            hybrid=True
        )
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics with error handling"""
        stats: Dict[str, Any] = {
            "status": "unknown",
            "error": None,
            "clause_count": 0,
            "policy_count": 0,
            "sections": {}
        }
        
        try:
            if hasattr(self.vector_db, 'get_stats'):
                stats.update(self.vector_db.get_stats())
            else:
                stats["error"] = "get_stats_method_not_available"
                
            if hasattr(self.vector_db, 'db') and self.vector_db.db:
                stats["index_size"] = self._get_directory_size("insurance_vector_db")
                stats["cache_hit_rate"] = self._calculate_cache_hit_rate()
                
            stats["status"] = "success"
            return stats
            
        except Exception as e:
            stats["error"] = str(e)
            stats["status"] = "error"
            logger.error(f"Failed to get database stats: {str(e)}")
            return stats
    
    def _get_directory_size(self, path: str) -> str:
        """Calculate directory size in human-readable format"""
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
                
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total < 1024:
                return f"{total:.2f} {unit}"
            total /= 1024
        return f"{total:.2f} TB"
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate embedding cache effectiveness"""
        if not hasattr(self.vector_db, 'embedding_cache'):
            return 0.0
            
        cache = self.vector_db.embedding_cache
        if not cache:
            return 0.0
            
        total_searches = len(self.vector_db.db.docstore._dict)
        if total_searches == 0:
            return 0.0
            
        return len(cache) / total_searches

def main():
    try:
        # Initialize system
        system = InsuranceVectorSystem()
        
        # Configuration
        config: Dict[str, Any] = {
            "db_name": "insurance_vector_db",
            "source_directories": [
                "embedded_docs",
                "additional_policies",
                "supplementary_clauses"
            ],
            "test_queries": [
                ("knee surgery coverage", "COVERAGE"),
                ("policy exclusions", "EXCLUSION"),
                ("claim submission process", "PROCEDURE")
            ]
        }
        
        # Verify directories
        missing_dirs = [d for d in config["source_directories"] if not Path(d).exists()]
        if missing_dirs:
            logger.warning(f"Missing directories: {missing_dirs}")
        
        # Create/update database
        logger.info("Starting database creation/update")
        stats = system.create_or_update_database(
            db_name=config["db_name"],
            embedded_dirs=config["source_directories"],
            parallel=True
        )
        logger.info(f"Database stats: {stats}")
        
        # Run test queries
        logger.info("\nRunning test queries:")
        for query, section in config["test_queries"]:
            results = system.search_policies(
                query=query,
                section_filter=section,
                threshold=0.65
            )
            
            if results:
                print(f"\nResults for '{query}' ({section}):")
                for i, res in enumerate(results[:3], 1):
                    print(f"{i}. {res['text'][:150]}... (Score: {res.get('score', 'N/A')})")
                    print(f"   Source: {res['metadata']['source']}")
                    if 'mongo_ref' in res:
                        print(f"   MongoDB Ref: {res['mongo_ref']}")
            else:
                print(f"\nNo results found for '{query}'")
        
        # Show final stats
        full_stats = system.get_database_stats()
        print("\nFinal Database Statistics:")
        for k, v in full_stats.items():
            print(f"{k.replace('_', ' ').title()}: {v}")
        
    except Exception as e:
        logger.error(f"System error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()