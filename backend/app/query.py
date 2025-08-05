import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import json
import re
import time
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import spacy
from PyPDF2 import PdfReader
import requests
from io import BytesIO
from pymongo import MongoClient

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('insurance_query_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InsuranceQueryProcessor:
    def __init__(self):
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.mongo_uri = os.getenv("MONGO_URI")

        if not self.gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment variables")
        if not self.mongo_uri:
            raise ValueError("Missing MONGO_URI in environment variables")

        try:
            self.nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

            self.embedder = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.gemini_api_key,
                task_type="retrieval_document"
            )

            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=0,
                google_api_key=self.gemini_api_key,
                max_output_tokens=2048
            )

            self.mongo_client = MongoClient(self.mongo_uri)
            self.vector_collection = self.mongo_client["insurance_documents"]["embeddings"]

            self.parser = JsonOutputParser()
            self._initialize_all_prompts()

            self.query_cache = {}
            self.document_cache = {}

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def _initialize_all_prompts(self):
        """Initialize all prompt templates with enhanced instructions"""
        # Main processing prompts
        self.query_classification_prompt = ChatPromptTemplate.from_template("""
        Classify this insurance-related input and determine processing requirements:
        
        Input: {input}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "input_type": "claim|question|document_qa|mixed",
            "processing_steps": ["list", "of", "required", "processing", "steps"],
            "needs_additional_info": boolean,
            "suggested_questions": ["list", "of", "clarifying", "questions"],
            "confidence": "high|medium|low"
        }}
        """)
        
        # Claim processing prompts
        self.claim_extraction_prompt = ChatPromptTemplate.from_template("""
        Extract detailed claim information from this input:
        
        Input: {input}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "demographics": {{
                "age": number,
                "gender": "male|female|other",
                "location": "string"
            }},
            "medical_details": {{
                "procedure": "string",
                "diagnosis": "string",
                "treatment_type": "surgery|hospitalization|therapy|other"
            }},
            "financial_details": {{
                "claimed_amount": number,
                "currency": "INR|USD|etc",
                "policy_tenure": "string"
            }},
            "extraction_confidence": "high|medium|low",
            "missing_information": ["list", "of", "missing", "items"]
        }}
        """)
        
        self.query_expansion_prompt = ChatPromptTemplate.from_template("""
        As an insurance expert, comprehensively expand this query for optimal processing:
        
        Original: {query}
        Extracted Claim Details: {claim_details}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "expanded_queries": ["list", "of", "expanded", "queries"],
            "key_terms": ["list", "of", "important", "terms"],
            "potential_amount_ranges": {{
                "min": number,
                "max": number,
                "currency": "INR",
                "calculation_basis": "string"
            }},
            "expected_documents": ["list", "of", "relevant", "document", "types"],
            "coverage_analysis": {{
                "likely_covered": boolean,
                "common_exclusions": ["list", "of", "potential", "exclusions"],
                "special_conditions": ["list", "of", "special", "considerations"]
            }}
        }}
        """)
        
        self.claim_decision_prompt = ChatPromptTemplate.from_template("""
        Perform comprehensive insurance claim analysis:
        
        Claim Details:
        {claim_details}
        
        Relevant Policy Clauses:
        {clauses}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "decision_analysis": {{
                "coverage_decision": "approved|rejected|review",
                "decision_reason": "detailed explanation",
                "confidence": "high|medium|low",
                "key_factors": ["list", "of", "decision", "factors"]
            }},
            "financial_analysis": {{
                "approved_amount": {{
                    "value": number,
                    "currency": "INR",
                    "calculation_basis": "string"
                }},
                "potential_adjustments": ["list", "of", "possible", "adjustments"],
                "deductibles_applied": {{
                    "amount": number,
                    "reason": "string"
                }}
            }},
            "documentation_analysis": {{
                "required_documents": ["list", "of", "needed", "documents"],
                "missing_documents": ["list", "of", "missing", "items"]
            }},
            "next_steps": {{
                "immediate_actions": ["list", "of", "actions"],
                "long_term_actions": ["list", "of", "followups"],
                "customer_communications": ["list", "of", "communication", "points"]
            }},
            "risk_analysis": {{
                "fraud_risk": "high|medium|low",
                "risk_factors": ["list", "of", "risk", "factors"],
                "mitigation_strategies": ["list", "of", "strategies"]
            }}
        }}
        """)
        
        # Document Q&A prompts
        self.document_processing_prompt = ChatPromptTemplate.from_template("""
        Analyze this document and prepare it for question answering:
        
        Document Metadata:
        {metadata}
        
        Document Text (excerpt):
        {text_excerpt}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "document_analysis": {{
                "document_type": "policy|claim_form|medical_report|other",
                "key_sections": ["list", "of", "document", "sections"],
                "coverage_period": "string",
                "notable_exclusions": ["list", "of", "exclusions"],
                "special_conditions": ["list", "of", "special", "clauses"]
            }},
            "processing_recommendations": {{
                "extraction_method": "full_text|sectional|key_points",
                "priority_sections": ["list", "of", "important", "sections"],
                "potential_questions": ["list", "of", "likely", "questions"]
            }}
        }}
        """)
        
        self.document_qa_prompt = ChatPromptTemplate.from_template("""
        Perform comprehensive document-based question answering:
        
        Document Context:
        {context}
        
        Question: {question}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "answer_analysis": {{
                "direct_answer": "string",
                "answer_confidence": "high|medium|low",
                "answer_source": "exact_text|inferred|composite",
                "supporting_evidence": ["list", "of", "supporting", "excerpts"]
            }},
            "document_references": {{
                "page_numbers": ["list", "of", "pages"],
                "section_names": ["list", "of", "sections"],
                "relevant_clauses": ["list", "of", "clause", "references"]
            }},
            "validation_info": {{
                "cross_references": ["list", "of", "related", "sections"],
                "potential_contradictions": ["list", "of", "contradictory", "info"],
                "consistency_check": "consistent|partial|conflicting"
            }},
            "additional_insights": {{
                "related_topics": ["list", "of", "related", "topics"],
                "implied_conditions": ["list", "of", "implied", "conditions"],
                "policy_implications": ["list", "of", "policy", "impacts"]
            }}
        }}
        """)
        
        self.batch_qa_prompt = ChatPromptTemplate.from_template("""
        Process this batch of questions against the document comprehensively:
        
        Document Summary:
        {document_summary}
        
        Questions:
        {questions}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "batch_summary": {{
                "total_questions": number,
                "questions_answered": number,
                "confidence_distribution": {{
                    "high": number,
                    "medium": number,
                    "low": number
                }},
                "coverage_analysis": {{
                    "fully_covered": number,
                    "partially_covered": number,
                    "unanswered": number
                }}
            }},
            "question_results": [
                {{
                    "question": "string",
                    "answer_summary": {{
                        "status": "answered|partial|unanswered",
                        "primary_answer": "string",
                        "alternative_interpretations": ["list", "of", "alternatives"]
                    }},
                    "detail_analysis": {{
                        "source_reliability": "high|medium|low",
                        "evidence_quality": "direct|indirect|inferred",
                        "consistency": "consistent|partial|conflicting"
                    }},
                    "followup_recommendations": {{
                        "needed": boolean,
                        "suggested_questions": ["list", "of", "followups"],
                        "recommended_sources": ["list", "of", "sources"]
                    }}
                }}
            ]
        }}
        """)
        
        # General question prompts
        self.general_question_prompt = ChatPromptTemplate.from_template("""
        Analyze and respond to this general insurance question thoroughly:
        
        Question: {question}
        
        Context: {context}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "question_analysis": {{
                "question_type": "coverage|claims|premium|eligibility|other",
                "complexity": "simple|moderate|complex",
                "information_needs": ["list", "of", "required", "information"]
            }},
            "answer_components": {{
                "direct_answer": "string",
                "qualifications": ["list", "of", "qualifying", "statements"],
                "examples": ["list", "of", "relevant", "examples"],
                "common_misconceptions": ["list", "of", "misconceptions"]
            }},
            "source_analysis": {{
                "primary_sources": ["list", "of", "primary", "sources"],
                "secondary_sources": ["list", "of", "secondary", "sources"],
                "source_reliability": "high|medium|low"
            }},
            "recommendations": {{
                "next_questions": ["list", "of", "followup", "questions"],
                "related_topics": ["list", "of", "related", "topics"],
                "action_items": ["list", "of", "suggested", "actions"]
            }}
        }}
        """)

    def _retrieve_clauses(self, queries: List[str], claim_details: Dict) -> List[Dict]:
        clauses = []
        for q in queries:
            try:
                procedure = claim_details.get('medical_details', {}).get('procedure', '')
                query_embedding = self.embedder.embed_query(f"{q} {procedure}")

                results = self.vector_collection.aggregate([
                    {
                        "$vectorSearch": {
                            "queryVector": query_embedding,
                            "path": "embedding",
                            "numCandidates": 100,
                            "limit": 3,
                            "index": "default"
                        }
                    }
                ])
                clauses.extend(list(results))
            except Exception as e:
                logger.warning(f"MongoDB vector search failed for '{q}': {str(e)}")
        return clauses

    def process_query(self, query: str) -> Dict[str, Any]:
        """Comprehensive query processing using all relevant prompts"""
        try:
            # Initial classification
            classification = self._classify_input(query)
            
            # Route processing based on classification
            if classification["input_type"] == "claim":
                return self._process_claim(query, classification)
            elif classification["input_type"] == "document_qa":
                return self._process_document_qa(query)
            elif classification["input_type"] == "question":
                return self._process_general_question(query)
            else:
                return self._process_mixed_query(query, classification)
                
        except Exception as e:
            logger.error(f"Comprehensive processing failed: {str(e)}", exc_info=True)
            return {
                "error": "Processing failed",
                "details": str(e),
                "timestamp": datetime.now().isoformat(),
                "recovery_suggestion": "Please rephrase your query or try again later"
            }

    def _classify_input(self, query: str) -> Dict[str, Any]:
        """Classify the input type using the classification prompt"""
        try:
            chain = self.query_classification_prompt | self.llm | self.parser
            return chain.invoke({"input": query})
        except Exception as e:
            logger.warning(f"Classification failed, defaulting to general question: {str(e)}")
            return {
                "input_type": "question",
                "processing_steps": ["general_question_processing"],
                "needs_additional_info": False,
                "suggested_questions": [],
                "confidence": "medium"
            }

    def _process_claim(self, query: str, classification: Dict) -> Dict[str, Any]:
        """Comprehensive claim processing using all claim-related prompts"""
        # Step 1: Extract claim details
        claim_details = self._extract_claim_details(query)
        
        # Step 2: Expand query for comprehensive analysis
        expanded = self._expand_claim_query(query, claim_details)
        
        # Step 3: Retrieve relevant clauses
        clauses = self._retrieve_clauses(
            expanded.get("expanded_queries", [query]),
            claim_details
        )
        
        # Step 4: Make comprehensive decision
        decision = self._make_claim_decision(claim_details, clauses)
        
        # Step 5: Compile comprehensive response
        return {
            "processing_type": "claim",
            "processing_steps": classification.get("processing_steps", []),
            "claim_details": claim_details,
            "query_expansion": expanded,
            "decision_analysis": decision,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "processing_time_ms": int(time.time() * 1000),
                "system_version": "2.1",
                "confidence": classification.get("confidence", "medium")
            }
        }

    def _extract_claim_details(self, query: str) -> Dict[str, Any]:
        """Enhanced claim detail extraction using dedicated prompt"""
        try:
            chain = self.claim_extraction_prompt | self.llm | self.parser
            return chain.invoke({"input": query})
        except Exception as e:
            logger.warning(f"Claim extraction failed, using fallback: {str(e)}")
            return self._fallback_claim_extraction(query)

    def _fallback_claim_extraction(self, query: str) -> Dict[str, Any]:
        """Fallback claim extraction when primary method fails"""
        doc = self.nlp(query.lower())
        
        # Basic demographic extraction
        age_gender = re.search(r'(\d+)\s*(year old|yo|yr)?\s*(male|female|M|F)\b', query, re.I)
        location = re.search(r'\b(delhi|mumbai|bangalore|pune|hyderabad|chennai|kolkata)\b', query, re.I)
        
        # Medical procedure extraction
        procedure = None
        diagnosis = None
        procedure_patterns = [
            r'(knee|heart|hip|shoulder|spine|brain)\s+(surgery|replacement|operation|procedure)',
            r'[\w\s]+ (surgery|treatment|therapy|procedure|bypass|transplant|implant)',
            r'(chemotherapy|radiation|dialysis|transplant|biopsy|endoscopy)'
        ]
        
        for pattern in procedure_patterns:
            if match := re.search(pattern, query, re.I):
                procedure = match.group(0).strip()
                break
        
        # Financial extraction
        amount = re.search(r'(?:Rs\.?|INR)\s*(\d+[,\.]\d+)', query, re.I)
        policy_age = re.search(r'(\d+)\s*(month|year|day)s?\s*old', query, re.I)
        
        return {
            "demographics": {
                "age": int(age_gender.group(1)) if age_gender else None,
                "gender": age_gender.group(3).lower() if age_gender else None,
                "location": location.group(0) if location else None
            },
            "medical_details": {
                "procedure": procedure,
                "diagnosis": diagnosis,
                "treatment_type": self._determine_treatment_type(procedure) if procedure else None
            },
            "financial_details": {
                "claimed_amount": float(amount.group(1).replace(',', '')) if amount else None,
                "currency": "INR",
                "policy_tenure": policy_age.group(0) if policy_age else None
            },
            "extraction_confidence": "medium",
            "missing_information": ["diagnosis"] if not diagnosis else []
        }
    def _determine_treatment_type(self, procedure: str) -> str:
        """Determine treatment type from procedure description"""
        if not procedure:
            return "other"
        
        procedure = procedure.lower()
        if any(t in procedure for t in ["surgery", "operation", "transplant"]):
            return "surgery"
        if any(t in procedure for t in ["therapy", "treatment", "chemotherapy"]):
            return "therapy"
        if "hospitalization" in procedure:
            return "hospitalization"
        return "other"

    def _expand_claim_query(self, query: str, claim_details: Dict) -> Dict[str, Any]:
        """Comprehensive query expansion using dedicated prompt"""
        try:
            chain = self.query_expansion_prompt | self.llm | self.parser
            return chain.invoke({
                "query": query,
                "claim_details": claim_details
            })
        except Exception as e:
            logger.warning(f"Query expansion failed: {str(e)}")
            return {
                "expanded_queries": [query],
                "key_terms": [],
                "potential_amount_ranges": {
                    "min": 0,
                    "max": 0,
                    "currency": "INR"
                }
            }
        
    def _retrieve_clauses(self, queries: List[str], claim_details: Dict) -> List[Dict]:
        """Enhanced clause retrieval with claim context"""
        clauses = []
        for query in queries:
            try:
                # Augment query with claim details
                augmented_query = f"{query} {claim_details.get('medical_details', {}).get('procedure', '')}"
                results = self.db.search(query=augmented_query, k=3)
                clauses.extend(results)
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {str(e)}")
        return clauses
    

    def _make_claim_decision(self, claim_details: Dict, clauses: List[Dict]) -> Dict:
        """Comprehensive claim decision using all relevant information"""
        try:
            chain = self.claim_decision_prompt | self.llm | self.parser
            return chain.invoke({
                "claim_details": claim_details,
                "clauses": clauses
            })
        except Exception as e:
            logger.error(f"Decision failed: {str(e)}")
            return {
                "decision_analysis": {
                    "coverage_decision": "review",
                    "decision_reason": "Decision process failed",
                    "confidence": "low",
                    "key_factors": []
                },
                "error_details": str(e)
            }
    
    def _process_document_qa(self, query: str) -> Dict[str, Any]:
        """Comprehensive document Q&A processing"""
        try:
            # Parse the document Q&A request
            request = json.loads(query)
            document_url = request.get("documents")
            questions = request.get("questions")
            
            if not document_url or not questions:
                raise ValueError("Invalid document Q&A request format")
            
            # Process the document and questions
            document_text = self._extract_text_from_pdf(document_url)
            if not document_text:
                return {"error": "Failed to extract document text"}
            
            # First analyze the document
            doc_analysis = self._analyze_document(document_text)
            
            # Then process questions in batches
            batch_results = []
            for i in range(0, len(questions), 5):  # Process 5 questions at a time
                batch = questions[i:i+5]
                result = self._process_question_batch(document_text, batch)
                batch_results.extend(result["question_results"])
            
            return {
                "processing_type": "document_qa",
                "document_analysis": doc_analysis,
                "question_results": batch_results,
                "summary": {
                    "total_questions": len(questions),
                    "answered_questions": sum(1 for r in batch_results if r["answer_summary"]["status"] == "answered"),
                    "confidence": self._calculate_overall_confidence(batch_results)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Document Q&A processing failed: {str(e)}")
            return {
                "error": "Document processing failed",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_document(self, document_text: str) -> Dict[str, Any]:
        """Comprehensive document analysis"""
        try:
            # Use first 2000 chars for analysis (adjust as needed)
            excerpt = document_text[:2000]
            chain = self.document_processing_prompt | self.llm | self.parser
            return chain.invoke({
                "metadata": {"length": len(document_text)},
                "text_excerpt": excerpt
            })
        except Exception as e:
            logger.warning(f"Document analysis failed: {str(e)}")
            return {
                "document_type": "unknown",
                "key_sections": [],
                "processing_recommendations": {
                    "extraction_method": "full_text"
                }
            }

    def _process_question_batch(self, document_text: str, questions: List[str]) -> Dict:
        """Comprehensive batch question processing"""
        try:
            # Create document summary for context
            summary = f"Document length: {len(document_text)} characters. Key sections identified."
            
            chain = self.batch_qa_prompt | self.llm | self.parser
            return chain.invoke({
                "document_summary": summary,
                "questions": questions
            })
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return {
                "batch_summary": {
                    "total_questions": len(questions),
                    "questions_answered": 0,
                    "confidence_distribution": {"high": 0, "medium": 0, "low": len(questions)}
                },
                "question_results": [{
                    "question": q,
                    "answer_summary": {
                        "status": "unanswered",
                        "primary_answer": "Processing failed",
                        "alternative_interpretations": []
                    }
                } for q in questions]
            }

    def _calculate_overall_confidence(self, results: List[Dict]) -> str:
        """Calculate overall confidence from multiple results"""
        if not results:
            return "low"
        
        high = sum(1 for r in results if r["detail_analysis"]["source_reliability"] == "high")
        medium = sum(1 for r in results if r["detail_analysis"]["source_reliability"] == "medium")
        
        if high / len(results) > 0.7:
            return "high"
        if (high + medium) / len(results) > 0.7:
            return "medium"
        return "low"

    def _process_general_question(self, query: str) -> Dict[str, Any]:
        """Comprehensive general question processing"""
        try:
            # First try to find direct answers in policy clauses
            clauses = self._retrieve_clauses([query], {})
            
            if clauses:
                context = " ".join(c.get("text", "")[:500] for c in clauses[:3])
            else:
                context = "No specific policy clauses found"
            
            # Use the general question prompt for comprehensive answer
            chain = self.general_question_prompt | self.llm | self.parser
            response = chain.invoke({
                "question": query,
                "context": context
            })
            
            return {
                "processing_type": "general_question",
                "question_analysis": response.get("question_analysis", {}),
                "answer_components": response.get("answer_components", {}),
                "source_analysis": response.get("source_analysis", {}),
                "recommendations": response.get("recommendations", {}),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "policy_clauses_used": bool(clauses),
                    "clauses_count": len(clauses)
                }
            }
            
        except Exception as e:
            logger.error(f"General question processing failed: {str(e)}")
            return {
                "error": "Question processing failed",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _process_mixed_query(self, query: str, classification: Dict) -> Dict[str, Any]:
        """Process queries with mixed requirements"""
        try:
            # Handle JSON input for document Q&A
            if query.strip().startswith('{') and query.strip().endswith('}'):
                try:
                    request = json.loads(query)
                    if "documents" in request and "questions" in request:
                        return self._process_document_qa(query)
                except json.JSONDecodeError:
                    pass
            
            # Default to general question processing with enhanced context
            return self._process_general_question(query)
            
        except Exception as e:
            logger.error(f"Mixed query processing failed: {str(e)}")
            return {
                "error": "Processing failed",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
    def _format_claim_response(self, data: Dict) -> Dict:
        return {
            "processing_type": "claim",
            "processing_steps": data.get("processing_steps", []),
            "claim_details": data.get("claim_details", {}),
            "query_expansion": data.get("query_expansion", {}),
            "decision_analysis": data.get("decision_analysis", {}),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "processing_time_ms": int(time.time() * 1000),
                "system_version": "2.1",
                "confidence": data.get("confidence", "medium")
            }
        }
    

    def _format_general_response(self, data: Dict) -> Dict:
        return {
            "processing_type": "general_question",
            "question_analysis": {
                "question_type": data.get("question_type", "unknown"),
                "complexity": data.get("complexity", "unknown"),
                "information_needs": data.get("information_needs", [])
            },
            "answer_components": {
                "direct_answer": data.get("direct_answer", ""),
                "qualifications": data.get("qualifications", []),
                "examples": data.get("examples", []),
                "common_misconceptions": data.get("common_misconceptions", [])
            },
            "source_analysis": {
                "primary_sources": data.get("primary_sources", []),
                "secondary_sources": data.get("secondary_sources", []),
                "source_reliability": data.get("source_reliability", "medium")
            },
            "recommendations": {
                "next_questions": data.get("next_questions", []),
                "related_topics": data.get("related_topics", []),
                "action_items": data.get("action_items", [])
            },
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "processing_time_ms": int(time.time() * 1000),
                "system_version": "2.1",
                "confidence": data.get("confidence", "medium"),
                "policy_clauses_used": data.get("policy_clauses_used", False),
                "clauses_count": data.get("clauses_count", 0)
            }
        }

    def _format_error_response(self, error: str) -> Dict:
        return {
            "error": "Processing failed",
            "details": error,
            "timestamp": datetime.now().isoformat(),
            "recovery_suggestion": "Please rephrase your query or try again later"
        }

    def _process_fallback(self, query: str) -> Dict:
        try:
            return self._process_general_question(query)
        except Exception as e:
            return self._format_error_response(str(e))

if __name__ == "__main__":
    try:
        processor = InsuranceQueryProcessor()

        test_queries = [
            "46M, knee surgery (Rs. 2,50,000), Pune, 3-month policy",
            "What is the difference between floater and individual health policies?",
            {
                "documents": "https://example.com/policy.pdf",
                "questions": ["What is covered under accidental injuries?"]
            }
        ]

        results = []
        for query in test_queries:
            result = processor.process_query(query)
            results.append(result)
            print(json.dumps(result, indent=2))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/results_{timestamp}.json", "w") as f:
            json.dump({
                "test_queries": test_queries,
                "results": results,
                "timestamp": timestamp
            }, f, indent=2)

    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        print(f"Application error: {str(e)}")
