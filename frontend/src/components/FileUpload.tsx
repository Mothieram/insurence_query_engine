import React, { useRef, useState } from 'react';
import { Upload, File, CheckCircle, AlertCircle, Loader2, X } from 'lucide-react';
import { chatService } from '../services/chatService';

interface FileUploadProps {
  onUploadComplete: (docId: string, filename: string) => void;
  onUploadError: (error: string) => void;
  onAnalyzeComplete?: (analysis: any, filename: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUploadComplete, onUploadError, onAnalyzeComplete }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadMode, setUploadMode] = useState<'upload' | 'analyze'>('upload');
  const [uploadProgress, setUploadProgress] = useState<{
    filename: string;
    status: 'uploading' | 'processing' | 'analyzing' | 'complete' | 'error';
    progress: number;
    docId?: string;
    error?: string;
    analysis?: any;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileUpload = async (file: File) => {
    // Validate file type
    const allowedTypes = ['application/pdf', 'text/plain', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    if (!allowedTypes.includes(file.type)) {
      onUploadError('Please upload a PDF, DOC, DOCX, or TXT file');
      return;
    }

    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      onUploadError('File size must be less than 10MB');
      return;
    }

    setUploadProgress({
      filename: file.name,
      status: 'uploading',
      progress: 0
    });

    try {
      if (uploadMode === 'analyze') {
        // Direct document analysis
        setUploadProgress(prev => prev ? {
          ...prev,
          status: 'analyzing',
          progress: 50
        } : null);

        const analysis = await chatService.analyzeDocument(file);
        
        setUploadProgress(prev => prev ? {
          ...prev,
          status: 'complete',
          progress: 100,
          analysis
        } : null);

        if (onAnalyzeComplete) {
          onAnalyzeComplete(analysis, file.name);
        }

        // Clear progress after 3 seconds
        setTimeout(() => {
          setUploadProgress(null);
        }, 3000);
        return;
      }

      // Regular upload and processing
      const formData = new FormData();
      formData.append('file', file);
      formData.append('store_original', 'true');

      const response = await fetch('http://localhost:8000/upload/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      setUploadProgress(prev => prev ? {
        ...prev,
        status: 'processing',
        progress: 50,
        docId: result.doc_id
      } : null);

      // Poll for processing status
      await pollProcessingStatus(result.doc_id, file.name);

    } catch (error) {
      setUploadProgress(prev => prev ? {
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Upload failed'
      } : null);
      onUploadError(error instanceof Error ? error.message : 'Upload failed');
    }
  };

  const pollProcessingStatus = async (docId: string, filename: string) => {
    const maxAttempts = 30; // 30 seconds timeout
    let attempts = 0;

    const poll = async () => {
      try {
        const response = await fetch(`http://localhost:8000/status/${docId}`);
        if (!response.ok) {
          throw new Error('Failed to check processing status');
        }

        const status = await response.json();
        
        if (status.error) {
          setUploadProgress(prev => prev ? {
            ...prev,
            status: 'error',
            error: status.error
          } : null);
          onUploadError(`Processing failed: ${status.error}`);
          return;
        }

        if (status.parsed && status.embedded) {
          setUploadProgress(prev => prev ? {
            ...prev,
            status: 'complete',
            progress: 100
          } : null);
          onUploadComplete(docId, filename);
          
          // Clear progress after 3 seconds
          setTimeout(() => {
            setUploadProgress(null);
          }, 3000);
          return;
        }

        // Update progress based on completion
        const progress = status.parsed ? 75 : 50;
        setUploadProgress(prev => prev ? {
          ...prev,
          progress
        } : null);

        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 1000); // Poll every second
        } else {
          throw new Error('Processing timeout');
        }
      } catch (error) {
        setUploadProgress(prev => prev ? {
          ...prev,
          status: 'error',
          error: error instanceof Error ? error.message : 'Processing failed'
        } : null);
        onUploadError(error instanceof Error ? error.message : 'Processing failed');
      }
    };

    poll();
  };

  const clearUpload = () => {
    setUploadProgress(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full">
      {/* Upload Mode Selector */}
      <div className="flex items-center justify-center space-x-4 mb-4">
        <button
          onClick={() => setUploadMode('upload')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
            uploadMode === 'upload'
              ? 'bg-purple-500 text-white'
              : 'bg-white/10 text-purple-200 hover:bg-white/20'
          }`}
        >
          Upload & Process
        </button>
        <button
          onClick={() => setUploadMode('analyze')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
            uploadMode === 'analyze'
              ? 'bg-purple-500 text-white'
              : 'bg-white/10 text-purple-200 hover:bg-white/20'
          }`}
        >
          Quick Analysis
        </button>
      </div>

      {!uploadProgress ? (
        <div
          className={`relative border-2 border-dashed rounded-xl p-6 transition-all duration-200 cursor-pointer ${
            isDragging
              ? 'border-purple-400 bg-purple-500/10 scale-105'
              : 'border-white/30 bg-white/5 hover:border-purple-400 hover:bg-white/10'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.doc,.docx,.txt"
            onChange={handleFileSelect}
            className="hidden"
          />
          
          <div className="flex flex-col items-center space-y-3">
            <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200 ${
              isDragging ? 'bg-purple-500 scale-110' : 'bg-white/20'
            }`}>
              <Upload className={`w-6 h-6 transition-colors duration-200 ${
                isDragging ? 'text-white' : 'text-purple-200'
              }`} />
            </div>
            
            <div className="text-center">
              <p className="text-white font-medium">
                {isDragging ? 'Drop your file here' : 
                 uploadMode === 'upload' ? 'Upload & Process Document' : 'Quick Document Analysis'}
              </p>
              <p className="text-purple-200 text-sm mt-1">
                {uploadMode === 'upload' 
                  ? 'Full processing with embedding for search • PDF, DOC, DOCX, TXT • Max 10MB'
                  : 'Instant analysis without storage • PDF, DOC, DOCX, TXT • Max 10MB'
                }
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4 border border-white/20">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-3">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                uploadProgress.status === 'complete' ? 'bg-green-500' :
                uploadProgress.status === 'error' ? 'bg-red-500' :
                'bg-blue-500'
              }`}>
                {uploadProgress.status === 'uploading' || uploadProgress.status === 'processing' ? (
                  <Loader2 className="w-4 h-4 text-white animate-spin" />
                ) : uploadProgress.status === 'complete' ? (
                  <CheckCircle className="w-4 h-4 text-white" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-white" />
                )}
              </div>
              
              <div>
                <div className="flex items-center space-x-2">
                  <File className="w-4 h-4 text-purple-200" />
                  <span className="text-white text-sm font-medium truncate max-w-48">
                    {uploadProgress.filename}
                  </span>
                </div>
                <p className="text-purple-200 text-xs">
                  {uploadProgress.status === 'uploading' && 'Uploading...'}
                  {uploadProgress.status === 'processing' && 'Processing document...'}
                  {uploadProgress.status === 'analyzing' && 'Analyzing document...'}
                  {uploadProgress.status === 'complete' && 'Processing complete!'}
                  {uploadProgress.status === 'error' && `Error: ${uploadProgress.error}`}
                </p>
              </div>
            </div>
            
            <button
              onClick={clearUpload}
              className="w-6 h-6 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition-colors duration-200"
            >
              <X className="w-3 h-3 text-white" />
            </button>
          </div>
          
          {uploadProgress.status !== 'error' && (
            <div className="w-full bg-white/20 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  uploadProgress.status === 'complete' ? 'bg-green-500' : 'bg-blue-500'
                }`}
                style={{ width: `${uploadProgress.progress}%` }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FileUpload;