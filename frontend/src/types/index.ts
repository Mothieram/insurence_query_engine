export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  queryType?: string;
  isStreaming?: boolean;
}

export interface QueryResponse {
  query?: string;
  results?: any[];
  expanded_queries?: string[];
  analysis?: any;
  vector_search_results?: any[];
}

export type QueryType = 'semantic' | 'vector' | 'keyword' | 'analyze';

export interface UploadResponse {
  status: string;
  doc_id: string;
}

export interface ProcessingStatus {
  parsed: boolean;
  embedded: boolean;
  error?: string;
}

export interface DocumentInfo {
  _id: string;
  filename: string;
  file_type: string;
  upload_date: string;
  processing_status: ProcessingStatus;
  metadata?: any;
  analysis?: any;
}