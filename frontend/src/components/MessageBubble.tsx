import React, { useState, useEffect } from 'react';
import { User, Bot, Copy, Check } from 'lucide-react';
import { Message } from '../types';
import { convertJsonToText } from '../utils/JsonToText';

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const [displayedContent, setDisplayedContent] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [copied, setCopied] = useState(false);
  const [processedContent, setProcessedContent] = useState('');

  useEffect(() => {
    // Process content first (convert JSON to readable text if needed)
    let contentToDisplay = message.content;
    
    if (message.type === 'assistant') {
      try {
        // Try to parse as JSON first
        const parsed = JSON.parse(message.content);
        contentToDisplay = convertJsonToText(parsed);
      } catch (e) {
        // If it's not JSON, use as-is
        contentToDisplay = message.content;
      }
    }
    
    setProcessedContent(contentToDisplay);
    
    if (message.isStreaming && message.type === 'assistant') {
      setIsTyping(true);
      setDisplayedContent('');
      
      let index = 0;
      const content = contentToDisplay;
      
      const timer = setInterval(() => {
        if (index < content.length) {
          setDisplayedContent(content.slice(0, index + 1));
          index += 3; // Speed up the typing effect
        } else {
          setIsTyping(false);
          clearInterval(timer);
        }
      }, 5); // Faster typing

      return () => clearInterval(timer);
    } else {
      setDisplayedContent(contentToDisplay);
    }
  }, [message.content, message.isStreaming, message.type]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const formatTime = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const isJsonResponse = (content: string) => {
    try {
      JSON.parse(content);
      return true;
    } catch {
      return false;
    }
  };

  if (message.type === 'user') {
    return (
      <div className="flex justify-end space-x-3 animate-fadeIn">
        <div className="max-w-xs md:max-w-2xl">
          <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-lg">
            <p className="text-sm md:text-base whitespace-pre-wrap">{displayedContent}</p>
          </div>
          <div className="flex items-center justify-end mt-1 space-x-2 text-xs text-purple-200">
            {message.queryType && (
              <span className="capitalize bg-purple-500/30 px-2 py-1 rounded-full">
                {message.queryType}
              </span>
            )}
            <span>{formatTime(message.timestamp)}</span>
          </div>
        </div>
        <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center flex-shrink-0">
          <User className="w-4 h-4 text-white" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start space-x-3 animate-fadeIn">
      <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-teal-500 rounded-full flex items-center justify-center flex-shrink-0">
        <Bot className="w-4 h-4 text-white" />
      </div>
      <div className="max-w-xs md:max-w-3xl">
        <div className="bg-white/20 backdrop-blur-sm text-white rounded-2xl rounded-tl-sm px-4 py-3 shadow-lg border border-white/10">
          <div className="relative">
            <div className="text-sm md:text-base whitespace-pre-wrap leading-relaxed">
              {displayedContent.split('\n').map((line, index) => {
                if (line.startsWith('**') && line.endsWith('**')) {
                  // Bold headers
                  return (
                    <div key={index} className="font-bold text-yellow-300 mt-3 mb-1">
                      {line.replace(/\*\*/g, '')}
                    </div>
                  );
                } else if (line.startsWith('• ')) {
                  // Bullet points
                  return (
                    <div key={index} className="ml-4 mb-1 text-gray-100">
                      <span className="text-blue-300">•</span> {line.substring(2)}
                    </div>
                  );
                } else if (line.match(/^\d+\./)) {
                  // Numbered lists
                  return (
                    <div key={index} className="ml-4 mb-1 text-gray-100">
                      <span className="text-green-300 font-medium">{line.match(/^\d+\./)?.[0]}</span> {line.replace(/^\d+\./, '')}
                    </div>
                  );
                } else if (line.trim().startsWith('🔍') || line.trim().startsWith('📝') || line.trim().startsWith('📋') || line.trim().startsWith('🧠') || line.trim().startsWith('🎯') || line.trim().startsWith('📄') || line.trim().startsWith('❓') || line.trim().startsWith('✅') || line.trim().startsWith('📊')) {
                  // Emoji headers
                  return (
                    <div key={index} className="font-bold text-blue-200 mt-4 mb-2 text-base">
                      {line}
                    </div>
                  );
                } else if (line.trim() === '') {
                  // Empty lines
                  return <div key={index} className="mb-2"></div>;
                } else {
                  // Regular text
                  return (
                    <div key={index} className="mb-1 text-gray-100">
                      {line}
                    </div>
                  );
                }
              })}
            </div>
            {isJsonResponse(message.content) && (
              <button
                onClick={handleCopy}
                className="absolute top-2 right-2 p-1 bg-white/10 hover:bg-white/20 rounded transition-colors duration-200"
                title="Copy original response"
              >
                {copied ? (
                  <Check className="w-4 h-4 text-green-400" />
                ) : (
                  <Copy className="w-4 h-4 text-gray-300" />
                )}
              </button>
            )}
          </div>
          {isTyping && (
            <div className="flex items-center space-x-1 mt-2">
              <div className="w-1 h-1 bg-white/60 rounded-full animate-pulse"></div>
              <div className="w-1 h-1 bg-white/60 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-1 h-1 bg-white/60 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
            </div>
          )}
        </div>
        <div className="flex items-center justify-start mt-1 text-xs text-purple-200">
          <span>{formatTime(message.timestamp)}</span>
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;