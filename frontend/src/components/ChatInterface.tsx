import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles, Search, Database, Brain, Upload, FileText } from 'lucide-react';
import { Message, QueryType } from '../types';
import MessageBubble from './MessageBubble';
import QueryTypeSelector from './QueryTypeSelector';
import FileUpload from './FileUpload';
import { chatService } from '../services/chatService';

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: 'Hello! I\'m your Insurance Document Assistant. I can help you search and analyze insurance documents using advanced AI. You can:\n\n• Upload documents for full processing and embedding\n• Get instant document analysis\n• Search using semantic, keyword, or deep analysis\n• Ask questions about policies, claims, and coverage\n\nWhat would you like to do?',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedQueryType, setSelectedQueryType] = useState<QueryType>('semantic');
  const [showUpload, setShowUpload] = useState(false);
  const [uploadedDocuments, setUploadedDocuments] = useState<Array<{docId: string, filename: string}>>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
      queryType: selectedQueryType,
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputValue.trim();
    setInputValue('');
    setIsLoading(true);

    try {
      console.log('Sending query:', currentInput, 'Type:', selectedQueryType);
      const response = await chatService.sendQuery(currentInput, selectedQueryType);
      console.log('Received response:', response);
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: typeof response === 'string' ? response : JSON.stringify(response, null, 2),
        timestamp: new Date(),
        isStreaming: true,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleUploadComplete = (docId: string, filename: string) => {
    setUploadedDocuments(prev => [...prev, { docId, filename }]);
    
    const successMessage: Message = {
      id: Date.now().toString(),
      type: 'assistant',
      content: `✅ Document "${filename}" has been successfully processed and embedded! You can now ask questions about this document using any of the search methods.`,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, successMessage]);
    setShowUpload(false);
  };

  const handleUploadError = (error: string) => {
    const errorMessage: Message = {
      id: Date.now().toString(),
      type: 'assistant',
      content: `❌ Upload failed: ${error}`,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, errorMessage]);
  };

  const handleAnalyzeComplete = (analysis: any, filename: string) => {
    const analysisMessage: Message = {
      id: Date.now().toString(),
      type: 'assistant',
      content: JSON.stringify(analysis, null, 2),
      timestamp: new Date(),
      isStreaming: true,
    };
    
    setMessages(prev => [...prev, analysisMessage]);
    setShowUpload(false);
  };

  const handleClearCache = async () => {
    try {
      await chatService.clearCache();
      const successMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: '🧹 Query cache has been cleared successfully!',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, successMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: `❌ Failed to clear cache: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };
  const getQueryTypeIcon = (type: QueryType) => {
    switch (type) {
      case 'semantic': return <Search className="w-4 h-4" />;
      case 'vector': return <Database className="w-4 h-4" />;
      case 'keyword': return <Sparkles className="w-4 h-4" />;
      case 'analyze': return <Brain className="w-4 h-4" />;
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      {/* Header */}
      <div className="bg-white/10 backdrop-blur-lg border-b border-white/20 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Insurance AI Assistant</h1>
              <p className="text-purple-200 text-sm">Powered by Advanced NLP & Vector Search</p>
              {uploadedDocuments.length > 0 && (
                <p className="text-green-300 text-xs mt-1">
                  📄 {uploadedDocuments.length} document{uploadedDocuments.length > 1 ? 's' : ''} processed
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={handleClearCache}
              className="flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-200 bg-white/10 hover:bg-white/20 text-white border border-white/20"
              title="Clear query cache"
            >
              <span className="text-sm font-medium">Clear Cache</span>
            </button>
            <button
              onClick={() => setShowUpload(!showUpload)}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-200 ${
                showUpload 
                  ? 'bg-purple-500 text-white' 
                  : 'bg-white/10 hover:bg-white/20 text-white border border-white/20'
              }`}
            >
              <Upload className="w-4 h-4" />
              <span className="text-sm font-medium">Upload</span>
            </button>
            <QueryTypeSelector
              selectedType={selectedQueryType}
              onTypeChange={setSelectedQueryType}
            />
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {/* Upload Section */}
          {showUpload && (
            <div className="animate-fadeIn">
              <FileUpload
                onUploadComplete={handleUploadComplete}
                onUploadError={handleUploadError}
                onAnalyzeComplete={handleAnalyzeComplete}
              />
            </div>
          )}
          
          {/* Uploaded Documents Info */}
          {uploadedDocuments.length > 0 && !showUpload && (
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10">
              <div className="flex items-center space-x-2 mb-2">
                <FileText className="w-4 h-4 text-green-400" />
                <span className="text-green-300 text-sm font-medium">Processed Documents</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {uploadedDocuments.map((doc, index) => (
                  <span
                    key={index}
                    className="bg-green-500/20 text-green-300 px-2 py-1 rounded-full text-xs border border-green-500/30"
                  >
                    {doc.filename}
                  </span>
                ))}
              </div>
            </div>
          )}
          
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          {isLoading && (
            <div className="flex items-center space-x-2 text-purple-200">
              <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center animate-pulse">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Container */}
      <div className="bg-white/10 backdrop-blur-lg border-t border-white/20 px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-4 bg-white/20 backdrop-blur-sm rounded-2xl px-4 py-3 border border-white/30">
            <div className="flex items-center space-x-2 text-purple-200">
              {getQueryTypeIcon(selectedQueryType)}
              <span className="text-sm font-medium capitalize">{selectedQueryType}</span>
            </div>
            <div className="w-px h-6 bg-white/30"></div>
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={uploadedDocuments.length > 0 
                ? `Ask about your uploaded documents... (${selectedQueryType} search)`
                : `Ask about insurance policies, claims, coverage... (${selectedQueryType} search)`
              }
              className="flex-1 bg-transparent text-white placeholder-purple-200 focus:outline-none"
              disabled={isLoading}
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center hover:from-purple-600 hover:to-pink-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105"
            >
              <Send className="w-5 h-5 text-white" />
            </button>
          </div>
          <div className="flex items-center justify-center mt-3 space-x-6 text-xs text-purple-200">
            <span className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span>AI-Powered</span>
            </span>
            <span className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
              <span>Vector Search</span>
            </span>
            <span className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              <span>Semantic Analysis</span>
            </span>
            {uploadedDocuments.length > 0 && (
              <span className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span>{uploadedDocuments.length} Doc{uploadedDocuments.length > 1 ? 's' : ''} Ready</span>
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;