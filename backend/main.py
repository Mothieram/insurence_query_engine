import os
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
import logging

# Import your existing modules
from app.parser import InsuranceClauseDetector
from app.embedding import AdvancedInsuranceEmbedder
from app.query import InsuranceQueryProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Insurance Document Processing Pipeline",
    description="End-to-end insurance document processing with NLP and vector search",
    version="2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("Missing MONGO_URI in environment variables")

client = MongoClient(mongo_uri)
try:
    # Verify connection
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

db = client["insurance_documents"]

# Collections
documents_col = db["documents"]
clauses_col = db["clauses"]
embeddings_col = db["embeddings"]

# Initialize processors
parser = InsuranceClauseDetector()
embedder = AdvancedInsuranceEmbedder()
query_processor = InsuranceQueryProcessor()

# Create indexes
def create_indexes():
    documents_col.create_index([("metadata.policy_number", ASCENDING)])
    documents_col.create_index([("metadata.coverage_types", ASCENDING)])
    clauses_col.create_index([("document_id", ASCENDING)])
    clauses_col.create_index([("type", ASCENDING)])
    embeddings_col.create_index([("clause_id", ASCENDING)])
    embeddings_col.create_index([("document_id", ASCENDING)])
    
    # Text search index for clauses
    clauses_col.create_index([("text", "text")])
    
    # Vector search index (MongoDB 7.0+)
    try:
        embeddings_col.create_index([
            ("embedding", "vector"),
        ], {
            "name": "vector_search_index",
            "vectorOptions": {
                "dimensions": 768,  # Match your embedding model
                "similarity": "cosine"
            }
        })
    except Exception as e:
        logger.warning(f"Could not create vector index: {str(e)}")

create_indexes()

@app.post("/upload/")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    store_original: bool = False
):
    """Upload and process an insurance document"""
    try:
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Read file content once
        file_content = await file.read()
        
        # Save basic document info
        doc_record = {
            "_id": doc_id,
            "filename": file.filename,
            "file_type": file.content_type,
            "upload_date": datetime.utcnow(),
            "processing_status": {
                "parsed": False,
                "embedded": False,
                "error": None
            }
        }
        
        # Store original file if requested
        if store_original:
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(file_content)
            doc_record["original_path"] = str(file_path)
        
        # Insert initial document record
        documents_col.insert_one(doc_record)
        
        # Add background processing task with file content
        background_tasks.add_task(
            process_document_content,
            file_content,
            file.filename,
            doc_id,
            store_original
        )
        
        return {"status": "processing", "doc_id": doc_id}
    
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_document_content(
    file_content: bytes,
    filename: str,
    doc_id: str,
    store_original: bool
):
    """Background task to process a document from its content"""
    temp_path = None
    try:
        # Step 1: Save to temp file
        temp_path = f"temp_{filename}"
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        parsed_data = parser.process_document(temp_path)
        
        if not parsed_data:
            raise ValueError("No data parsed from document")
        
        # Step 2: Save parsed data to MongoDB
        update_doc = {
            "$set": {
                "metadata": parsed_data.get("metadata", {}),
                "analysis": parsed_data.get("analysis", {}),
                "processing_status.parsed": True
            }
        }
        documents_col.update_one({"_id": doc_id}, update_doc)
        
        # Insert clauses
        clauses = []
        for idx, clause in enumerate(parsed_data.get("clauses", [])):
            clause_id = str(uuid.uuid4())
            clause_record = {
                "_id": clause_id,
                "document_id": doc_id,
                **clause,
                "metadata": {
                    "position_in_doc": idx,
                    **clause.get("metadata", {})
                }
            }
            clauses.append(clause_record)
        
        if clauses:
            clauses_col.insert_many(clauses)
        
        # Step 3: Generate embeddings
        clause_texts = [c["text"] for c in clauses]
        embeddings = embedder.batch_embed(clause_texts)
        
        # Save embeddings
        embedding_records = []
        for clause, embedding in zip(clauses, embeddings):
            if not embedding:
                continue
                
            embedding_records.append({
                "clause_id": clause["_id"],
                "document_id": doc_id,
                "embedding": embedding,
                "embedding_model": "gemini-001",
                "vector_index": {
                    "type": "knnVector",
                    "version": "1.0"
                },
                "created_at": datetime.utcnow(),
                "clause_hash": hashlib.md5(clause["text"].encode('utf-8')).hexdigest()
            })
        
        if embedding_records:
            embeddings_col.insert_many(embedding_records)
        
        # Mark document as complete
        documents_col.update_one(
            {"_id": doc_id},
            {"$set": {"processing_status.embedded": True}}
        )
        
        logger.info(f"Successfully processed document {doc_id}")
        
    except Exception as e:
        logger.error(f"Failed to process document {doc_id}: {str(e)}")
        documents_col.update_one(
            {"_id": doc_id},
            {"$set": {
                "processing_status.error": str(e),
                "processing_status.parsed": False,
                "processing_status.embedded": False
            }}
        )
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str, include_clauses: bool = False):
    """Retrieve a processed document"""
    document = documents_col.find_one({"_id": doc_id})
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    result = {"document": document}
    
    if include_clauses:
        clauses = list(clauses_col.find({"document_id": doc_id}))
        result["clauses"] = clauses
        
        # Get embeddings for clauses
        clause_ids = [c["_id"] for c in clauses]
        embeddings = list(embeddings_col.find({
            "clause_id": {"$in": clause_ids}
        }))
        result["embeddings"] = embeddings
    
    return result

@app.post("/analyze/query")
async def analyze_insurance_query(query: Union[str, Dict[str, Any]]):
    """
    Comprehensive insurance query analysis endpoint
    Accepts both text queries and structured JSON inputs
    """
    try:
        # Handle both string and JSON input
        if isinstance(query, dict):
            query_str = json.dumps(query)
        else:
            query_str = str(query)
        
        # Process with full pipeline
        result = query_processor.process_query(query_str)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Query analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/document")
async def analyze_insurance_document(file: UploadFile):
    """
    End-to-end document analysis combining:
    - Document parsing
    - Clause extraction
    - Embedding generation
    - Immediate analysis
    """
    try:
        # Process document through existing pipeline
        doc_id = str(uuid.uuid4())
        file_content = await file.read()
        
        # Temporary processing (modify as needed)
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        # Use query processor's document analysis
        doc_text = query_processor._extract_text_from_pdf(temp_path) if file.filename.endswith('.pdf') else file_content.decode()
        analysis = query_processor._analyze_document(doc_text)
        
        # Generate potential questions
        questions = analysis.get("processing_recommendations", {}).get("potential_questions", [])
        if questions:
            batch_result = query_processor._process_question_batch(doc_text, questions[:5])  # Limit to 5 questions
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "document_analysis": analysis,
            "auto_generated_qa": batch_result if questions else None,
            "document_id": doc_id
        }
    
    except Exception as e:
        logger.error(f"Document analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/semantic")
async def semantic_search(
    query: str, 
    limit: int = 5,
    expand_query: bool = True
):
    """Enhanced semantic search with query expansion"""
    try:
        # Use query processor for expansion and search
        classification = query_processor._classify_input(query)
        expanded = [query]
        
        if expand_query and classification["input_type"] == "claim":
            claim_details = query_processor._extract_claim_details(query)
            expanded = query_processor._expand_claim_query(query, claim_details).get("expanded_queries", [query])
        
        # Get results for each expanded query
        all_results = []
        for q in expanded:
            clauses = query_processor._retrieve_clauses([q], {})
            all_results.extend(clauses)
        
        # Deduplicate and sort
        unique_results = {r["text"]: r for r in all_results}.values()
        sorted_results = sorted(unique_results, key=lambda x: x.get("score", 0), reverse=True)[:limit]
        
        return {"query": query, "results": sorted_results}
    
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/keyword")
async def keyword_search(query: str, limit: int = 5):
    """Traditional keyword search"""
    try:
        # Search in clauses
        clauses = list(clauses_col.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit))
        
        # Get related documents and embeddings
        results = []
        for clause in clauses:
            document = documents_col.find_one({"_id": clause["document_id"]})
            embedding = embeddings_col.find_one({"clause_id": clause["_id"]})
            
            results.append({
                "clause": clause,
                "document": document,
                "embedding": embedding
            })
        
        return {"query": query, "results": results}
    
    except Exception as e:
        logger.error(f"Keyword search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{doc_id}")
async def get_processing_status(doc_id: str):
    """Check document processing status"""
    document = documents_col.find_one(
        {"_id": doc_id},
        {"processing_status": 1}
    )
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document["processing_status"]

@app.get("/clear_cache")
async def clear_cache():
    """Clear the query cache"""
    query_processor.query_cache = {}
    return {"status": "cache cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)