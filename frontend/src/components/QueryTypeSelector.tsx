import React from 'react';
import { Search, Database, Sparkles, Brain, ChevronDown } from 'lucide-react';
import { QueryType } from '../types';

interface QueryTypeSelectorProps {
  selectedType: QueryType;
  onTypeChange: (type: QueryType) => void;
}

const QueryTypeSelector: React.FC<QueryTypeSelectorProps> = ({ selectedType, onTypeChange }) => {
  const queryTypes: { type: QueryType; label: string; icon: React.ReactNode; description: string }[] = [
    {
      type: 'semantic',
      label: 'Semantic Search',
      icon: <Search className="w-4 h-4" />,
      description: 'Enhanced semantic search with query expansion'
    },
    {
      type: 'vector',
      label: 'Query Analysis',
      icon: <Database className="w-4 h-4" />,
      description: 'Comprehensive insurance query analysis'
    },
    {
      type: 'keyword',
      label: 'Keyword Search',
      icon: <Sparkles className="w-4 h-4" />,
      description: 'Traditional text matching'
    },
    {
      type: 'analyze',
      label: 'Deep Analysis',
      icon: <Brain className="w-4 h-4" />,
      description: 'Comprehensive AI analysis'
    }
  ];

  const [isOpen, setIsOpen] = React.useState(false);
  const selectedQuery = queryTypes.find(q => q.type === selectedType);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 bg-white/10 hover:bg-white/20 backdrop-blur-sm rounded-lg px-3 py-2 text-white transition-all duration-200 border border-white/20"
      >
        {selectedQuery?.icon}
        <span className="text-sm font-medium">{selectedQuery?.label}</span>
        <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute right-0 top-12 w-64 bg-white/95 backdrop-blur-lg rounded-xl shadow-xl border border-white/20 z-10 overflow-hidden">
          {queryTypes.map((query) => (
            <button
              key={query.type}
              onClick={() => {
                onTypeChange(query.type);
                setIsOpen(false);
              }}
              className={`w-full flex items-center space-x-3 px-4 py-3 text-left hover:bg-purple-50 transition-colors duration-200 ${
                selectedType === query.type ? 'bg-purple-100 border-r-2 border-purple-500' : ''
              }`}
            >
              <div className={`p-2 rounded-lg ${
                selectedType === query.type 
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white' 
                  : 'bg-gray-100 text-gray-600'
              }`}>
                {query.icon}
              </div>
              <div>
                <div className={`font-medium ${selectedType === query.type ? 'text-purple-700' : 'text-gray-800'}`}>
                  {query.label}
                </div>
                <div className="text-xs text-gray-500">{query.description}</div>
              </div>
            </button>
          ))}
        </div>
      )}

      {isOpen && (
        <div
          className="fixed inset-0 z-0"
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  );
};

export default QueryTypeSelector;