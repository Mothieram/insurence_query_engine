import { QueryType, QueryResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000';

class ChatService {
  async sendQuery(query: string, queryType: QueryType): Promise<QueryResponse> {
    try {
      let endpoint = '';
      let payload: any = {};

      switch (queryType) {
        case 'semantic':
          endpoint = '/search/semantic';
          const semanticUrl = new URL(`${API_BASE_URL}${endpoint}`);
          semanticUrl.searchParams.append('query', query);
          semanticUrl.searchParams.append('limit', '5');
          semanticUrl.searchParams.append('expand_query', 'true');
          
          const semanticResponse = await fetch(semanticUrl.toString(), {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          });
          
          if (!semanticResponse.ok) {
            throw new Error(`HTTP error! status: ${semanticResponse.status}`);
          }
          
          return await semanticResponse.json();

        case 'vector':
          endpoint = '/analyze/query';
          payload = query; // Send as string directly, not as object
          break;

        case 'keyword':
          endpoint = '/search/keyword';
          const keywordUrl = new URL(`${API_BASE_URL}${endpoint}`);
          keywordUrl.searchParams.append('query', query);
          keywordUrl.searchParams.append('limit', '5');
          
          const keywordResponse = await fetch(keywordUrl.toString(), {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          });
          
          if (!keywordResponse.ok) {
            throw new Error(`HTTP error! status: ${keywordResponse.status}`);
          }
          
          return await keywordResponse.json();

        case 'analyze':
          endpoint = '/analyze/query';
          payload = query; // Send as string directly, not as object
          break;

        default:
          throw new Error(`Unknown query type: ${queryType}`);
      }

      // For POST requests (vector and analyze)
      if (queryType === 'vector' || queryType === 'analyze') {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload), // This will now send the query string directly
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const result = await response.json();
        console.log('API Response:', result); // Debug log
        return result;
      }

      throw new Error('Invalid request type');
    } catch (error) {
      console.error('Chat service error:', error);
      throw error;
    }
  }

  async checkServerHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/`, {
        method: 'GET',
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async uploadDocument(file: File, storeOriginal: boolean = true): Promise<{status: string, doc_id: string}> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('store_original', storeOriginal.toString());

      const response = await fetch(`${API_BASE_URL}/upload/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Upload service error:', error);
      throw error;
    }
  }

  async getProcessingStatus(docId: string): Promise<{parsed: boolean, embedded: boolean, error?: string}> {
    try {
      const response = await fetch(`${API_BASE_URL}/status/${docId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Status check service error:', error);
      throw error;
    }
  }

  async getDocument(docId: string, includeClauses: boolean = false): Promise<any> {
    try {
      const url = new URL(`${API_BASE_URL}/documents/${docId}`);
      if (includeClauses) {
        url.searchParams.append('include_clauses', 'true');
      }

      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Document retrieval service error:', error);
      throw error;
    }
  }

  async analyzeDocument(file: File): Promise<any> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/analyze/document`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Document analysis service error:', error);
      throw error;
    }
  }

  async clearCache(): Promise<{status: string}> {
    try {
      const response = await fetch(`${API_BASE_URL}/clear_cache`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Clear cache service error:', error);
      throw error;
    }
  }
}

export const chatService = new ChatService();