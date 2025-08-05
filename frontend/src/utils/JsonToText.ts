import { formatNumber } from "../utils/utlis";

interface JsonValue {
  [key: string]: any;
}

export class JsonToTextConverter {
  private static formatValue(value: any, depth: number = 0): string {
    const indent = '  '.repeat(depth);
    
    if (value === null || value === undefined) {
      return 'Not specified';
    }
    
    if (typeof value === 'string') {
      return value;
    }
    
    if (typeof value === 'number') {
      return value.toString();
    }
    
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }
    
    if (Array.isArray(value)) {
      if (value.length === 0) return 'None';
      
      return value.map((item, index) => {
        if (typeof item === 'string' || typeof item === 'number') {
          return `${index + 1}. ${item}`;
        }
        return `${index + 1}. ${this.formatValue(item, depth + 1)}`;
      }).join('\n');
    }
    
    if (typeof value === 'object') {
      return this.formatObject(value, depth);
    }
    
    return String(value);
  }
  
  private static formatObject(obj: JsonValue, depth: number = 0): string {
    const entries = Object.entries(obj);
    if (entries.length === 0) return 'No data available';
    
    return entries.map(([key, value]) => {
      const formattedKey = this.formatKey(key);
      const formattedValue = this.formatValue(value, depth + 1);
      
      if (typeof value === 'object' && !Array.isArray(value) && value !== null) {
        return `**${formattedKey}:**\n${formattedValue}`;
      }
      
      return `**${formattedKey}:** ${formattedValue}`;
    }).join('\n\n');
  }
  
  private static formatKey(key: string): string {
    // Convert snake_case and camelCase to readable format
    return key
      .replace(/_/g, ' ')
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase())
      .trim();
  }
  
  public static convertToReadableText(jsonData: any): string {
    if (typeof jsonData === 'string') {
      try {
        jsonData = JSON.parse(jsonData);
      } catch {
        return jsonData; // Return as-is if not valid JSON
      }
    }
    
    if (typeof jsonData !== 'object' || jsonData === null) {
      return String(jsonData);
    }
    
    // Handle specific insurance query response formats
    if (this.isInsuranceQueryResponse(jsonData)) {
      return this.formatInsuranceResponse(jsonData);
    }
    
    // Handle search results
    if (this.isSearchResponse(jsonData)) {
      return this.formatSearchResponse(jsonData);
    }
    
    // Handle document analysis
    if (this.isDocumentAnalysis(jsonData)) {
      return this.formatDocumentAnalysis(jsonData);
    }
    
    // Generic object formatting
    return this.formatObject(jsonData);
  }
  
  private static isInsuranceQueryResponse(data: any): boolean {
    return data.hasOwnProperty('query') || 
           data.hasOwnProperty('results') || 
           data.hasOwnProperty('expanded_queries') ||
           data.hasOwnProperty('analysis') ||
           data.hasOwnProperty('processing_type') ||
           data.hasOwnProperty('claim_details') ||
           data.hasOwnProperty('decision_analysis');
  }
  
  private static isSearchResponse(data: any): boolean {
    return data.hasOwnProperty('query') && data.hasOwnProperty('results');
  }
  
  private static isDocumentAnalysis(data: any): boolean {
    return data.hasOwnProperty('document_analysis') || 
           data.hasOwnProperty('auto_generated_qa');
  }
  
  private static formatInsuranceResponse(data: any): string {
    let result = '';
    
    // Query section
    if (data.original_query || data.query) {
      result += `🔍 **Your Query:** ${data.original_query || data.query}\n\n`;
    }
    
    // Processing type and steps
    if (data.processing_type) {
      result += `⚙️ **Processing Type:** ${this.formatKey(data.processing_type)}\n\n`;
    }
    
    if (data.processing_steps && Array.isArray(data.processing_steps)) {
      result += `📋 **Processing Steps:**\n`;
      data.processing_steps.forEach((step: string, index: number) => {
        result += `${index + 1}. ${step}\n`;
      });
      result += '\n';
    }
    
    // Claim details section
    if (data.claim_details) {
      result += `📊 **Claim Details:**\n`;
      const details = data.claim_details;
      
      if (details.demographics) {
        result += `• **Demographics:**\n`;
        if (details.demographics.age) result += `  - Age: ${details.demographics.age}\n`;
        if (details.demographics.gender) result += `  - Gender: ${this.formatKey(details.demographics.gender)}\n`;
        if (details.demographics.location) result += `  - Location: ${details.demographics.location}\n`;
      }
      
      if (details.medical_details) {
        result += `• **Medical Details:**\n`;
        if (details.medical_details.procedure) result += `  - Procedure: ${details.medical_details.procedure}\n`;
        if (details.medical_details.diagnosis) {
          result += `  - Diagnosis: ${details.medical_details.diagnosis}\n`;
        } else {
          result += `  - Diagnosis: Not provided\n`;
        }
        if (details.medical_details.treatment_type) result += `  - Treatment Type: ${this.formatKey(details.medical_details.treatment_type)}\n`;
      }
      
      if (details.financial_details) {
        result += `• **Financial Details:**\n`;
        if (details.financial_details.claimed_amount) {
          const amount = details.financial_details.claimed_amount;
          const currency = details.financial_details.currency || 'INR';
          result += `  - Claimed Amount: ${currency} ${amount.toLocaleString()}\n`;
        }
        if (details.financial_details.policy_tenure) result += `  - Policy Tenure: ${details.financial_details.policy_tenure}\n`;
      }
      
      if (details.extraction_confidence) {
        result += `• **Extraction Confidence:** ${this.formatKey(details.extraction_confidence)}\n`;
      }
      
      if (details.missing_information && Array.isArray(details.missing_information)) {
        result += `• **Missing Information:** ${details.missing_information.length > 0 ? details.missing_information.join(', ') : 'None'}\n`;
      }
      
      result += '\n';
    }
    
    // Expanded queries
    if (data.expanded_queries && Array.isArray(data.expanded_queries) && data.expanded_queries.length > 0) {
      result += `📝 **Related Search Terms:**\n`;
      data.expanded_queries.forEach((query: string, index: number) => {
        result += `${index + 1}. ${query}\n`;
      });
      result += '\n';
    }
    
    // Query expansion section
    if (data.query_expansion) {
      const expansion = data.query_expansion;
      
      if (expansion.key_terms && Array.isArray(expansion.key_terms)) {
        result += `🔑 **Key Terms:** ${expansion.key_terms.join(', ')}\n\n`;
      }
      
      if (expansion.potential_amount_ranges) {
        const ranges = expansion.potential_amount_ranges;
        result += `💰 **Potential Amount Ranges:**\n`;
        if (ranges.min && ranges.max) {
          const currency = ranges.currency || 'INR';
          result += `• Range: ${currency} ${ranges.min.toLocaleString()} - ${currency} ${ranges.max.toLocaleString()}\n`;
        }
        if (ranges.calculation_basis) {
          result += `• Basis: ${ranges.calculation_basis}\n`;
        }
        result += '\n';
      }
      
      if (expansion.expected_documents && Array.isArray(expansion.expected_documents)) {
        result += `📄 **Expected Documents:**\n`;
        expansion.expected_documents.forEach((doc: string, index: number) => {
          result += `${index + 1}. ${doc}\n`;
        });
        result += '\n';
      }
      
      if (expansion.coverage_analysis) {
        const coverage = expansion.coverage_analysis;
        result += `🛡️ **Coverage Analysis:**\n`;
        if (coverage.likely_covered !== undefined) {
          result += `• **Likely Covered:** ${coverage.likely_covered ? 'Yes' : 'No'}\n`;
        }
        
        if (coverage.common_exclusions && Array.isArray(coverage.common_exclusions)) {
          result += `• **Common Exclusions:**\n`;
          coverage.common_exclusions.forEach((exclusion: string) => {
            result += `  - ${exclusion}\n`;
          });
        }
        
        if (coverage.special_conditions && Array.isArray(coverage.special_conditions)) {
          result += `• **Special Conditions:**\n`;
          coverage.special_conditions.forEach((condition: string) => {
            result += `  - ${condition}\n`;
          });
        }
        result += '\n';
      }
    }
    
    // Analysis section
    if (data.analysis) {
      result += `🧠 **Query Analysis:**\n`;
      if (data.analysis.intent) {
        result += `• **Intent:** ${data.analysis.intent}\n`;
      }
      if (data.analysis.entities && Array.isArray(data.analysis.entities)) {
        result += `• **Key Terms:** ${data.analysis.entities.join(', ')}\n`;
      }
      if (data.analysis.confidence) {
        result += `• **Confidence:** ${(data.analysis.confidence * 100).toFixed(1)}%\n`;
      }
      if (data.analysis.query_type) {
        result += `• **Query Type:** ${data.analysis.query_type}\n`;
      }
      result += '\n';
    }
    
    // Decision analysis section
    if (data.decision_analysis) {
      const decision = data.decision_analysis;
      
      result += `⚖️ **Decision Analysis:**\n`;
      if (decision.coverage_decision) {
        result += `• **Coverage Decision:** ${this.formatKey(decision.coverage_decision)}\n`;
      }
      if (decision.decision_reason) {
        result += `• **Reason:** ${decision.decision_reason}\n`;
      }
      if (decision.confidence) {
        result += `• **Confidence:** ${this.formatKey(decision.confidence)}\n`;
      }
      if (decision.key_factors && Array.isArray(decision.key_factors)) {
        result += `• **Key Factors:** ${decision.key_factors.join(', ')}\n`;
      }
      result += '\n';
      
      if (decision.financial_analysis) {
        result += `💵 **Financial Analysis:**\n`;
        const financial = decision.financial_analysis;
        
        if (financial.approved_amount) {
          const amount = financial.approved_amount;
          if (amount.value !== null && amount.value !== undefined) {
            const currency = amount.currency || 'INR';
            result += `• **Approved Amount:** ${currency} ${amount.value.toLocaleString()}\n`;
          } else {
            result += `• **Approved Amount:** Pending review\n`;
          }
          if (amount.calculation_basis) {
            result += `• **Calculation Basis:** ${amount.calculation_basis}\n`;
          }
        }
        
        if (financial.potential_adjustments && Array.isArray(financial.potential_adjustments)) {
          result += `• **Potential Adjustments:**\n`;
          financial.potential_adjustments.forEach((adjustment: string) => {
            result += `  - ${adjustment}\n`;
          });
        }
        
        if (financial.deductibles_applied) {
          const deductible = financial.deductibles_applied;
          if (deductible.amount !== null && deductible.amount !== undefined) {
            result += `• **Deductibles Applied:** ${deductible.amount}\n`;
          } else {
            result += `• **Deductibles Applied:** To be determined\n`;
          }
          if (deductible.reason) {
            result += `• **Deductible Reason:** ${deductible.reason}\n`;
          }
        }
        result += '\n';
      }
      
      if (decision.documentation_analysis) {
        result += `📋 **Documentation Analysis:**\n`;
        const docs = decision.documentation_analysis;
        
        if (docs.required_documents && Array.isArray(docs.required_documents)) {
          result += `• **Required Documents:**\n`;
          docs.required_documents.forEach((doc: string) => {
            result += `  - ${doc}\n`;
          });
        }
        
        if (docs.missing_documents && Array.isArray(docs.missing_documents)) {
          result += `• **Missing Documents:**\n`;
          docs.missing_documents.forEach((doc: string) => {
            result += `  - ${doc}\n`;
          });
        }
        result += '\n';
      }
      
      if (decision.next_steps) {
        result += `🚀 **Next Steps:**\n`;
        const steps = decision.next_steps;
        
        if (steps.immediate_actions && Array.isArray(steps.immediate_actions)) {
          result += `• **Immediate Actions:**\n`;
          steps.immediate_actions.forEach((action: string) => {
            result += `  - ${action}\n`;
          });
        }
        
        if (steps.long_term_actions && Array.isArray(steps.long_term_actions)) {
          result += `• **Long-term Actions:**\n`;
          steps.long_term_actions.forEach((action: string) => {
            result += `  - ${action}\n`;
          });
        }
        
        if (steps.customer_communications && Array.isArray(steps.customer_communications)) {
          result += `• **Customer Communications:**\n`;
          steps.customer_communications.forEach((comm: string) => {
            result += `  - ${comm}\n`;
          });
        }
        result += '\n';
      }
      
      if (decision.risk_analysis) {
        result += `⚠️ **Risk Analysis:**\n`;
        const risk = decision.risk_analysis;
        
        if (risk.fraud_risk) {
          result += `• **Fraud Risk:** ${this.formatKey(risk.fraud_risk)}\n`;
        }
        
        if (risk.risk_factors && Array.isArray(risk.risk_factors)) {
          result += `• **Risk Factors:**\n`;
          risk.risk_factors.forEach((factor: string) => {
            result += `  - ${factor}\n`;
          });
        }
        
        if (risk.mitigation_strategies && Array.isArray(risk.mitigation_strategies)) {
          result += `• **Mitigation Strategies:**\n`;
          risk.mitigation_strategies.forEach((strategy: string) => {
            result += `  - ${strategy}\n`;
          });
        }
        result += '\n';
      }
    }
    
    // Question analysis section
    if (data.question_analysis) {
      result += `❓ **Question Analysis:**\n`;
      const qa = data.question_analysis;
      
      if (qa.question_type) {
        result += `• **Question Type:** ${this.formatKey(qa.question_type)}\n`;
      }
      if (qa.complexity) {
        result += `• **Complexity:** ${this.formatKey(qa.complexity)}\n`;
      }
      if (qa.information_needs && Array.isArray(qa.information_needs)) {
        result += `• **Information Needs:** ${qa.information_needs.join(', ')}\n`;
      }
      result += '\n';
    }
    
    // Answer components section
    if (data.answer_components) {
      result += `💡 **Answer Components:**\n`;
      const components = data.answer_components;
      
      if (components.direct_answer) {
        result += `• **Direct Answer:** ${components.direct_answer}\n\n`;
      }
      
      if (components.qualifications && Array.isArray(components.qualifications)) {
        result += `• **Qualifications:**\n`;
        components.qualifications.forEach((qual: string) => {
          result += `  - ${qual}\n`;
        });
        result += '\n';
      }
      
      if (components.examples && Array.isArray(components.examples)) {
        result += `• **Examples:**\n`;
        components.examples.forEach((example: string) => {
          result += `  - ${example}\n`;
        });
        result += '\n';
      }
      
      if (components.common_misconceptions && Array.isArray(components.common_misconceptions)) {
        result += `• **Common Misconceptions:**\n`;
        components.common_misconceptions.forEach((misconception: string) => {
          result += `  - ${misconception}\n`;
        });
        result += '\n';
      }
    }
    
    // Recommendations section
    if (data.recommendations) {
      result += `🎯 **Recommendations:**\n`;
      const rec = data.recommendations;
      
      if (rec.next_questions && Array.isArray(rec.next_questions)) {
        result += `• **Next Questions to Ask:**\n`;
        rec.next_questions.forEach((question: string) => {
          result += `  - ${question}\n`;
        });
        result += '\n';
      }
      
      if (rec.related_topics && Array.isArray(rec.related_topics)) {
        result += `• **Related Topics:** ${rec.related_topics.join(', ')}\n\n`;
      }
      
      if (rec.action_items && Array.isArray(rec.action_items)) {
        result += `• **Action Items:**\n`;
        rec.action_items.forEach((item: string) => {
          result += `  - ${item}\n`;
        });
        result += '\n';
      }
    }
    
    // Source analysis section
    if (data.source_analysis) {
      result += `📚 **Source Analysis:**\n`;
      const source = data.source_analysis;
      
      if (source.primary_sources && Array.isArray(source.primary_sources)) {
        result += `• **Primary Sources:** ${source.primary_sources.join(', ')}\n`;
      }
      
      if (source.secondary_sources && Array.isArray(source.secondary_sources)) {
        result += `• **Secondary Sources:** ${source.secondary_sources.join(', ')}\n`;
      }
      
      if (source.source_reliability) {
        result += `• **Source Reliability:** ${this.formatKey(source.source_reliability)}\n`;
      }
      result += '\n';
    }
    
    // Results section
    if (data.results && Array.isArray(data.results) && data.results.length > 0) {
      result += `📋 **Search Results:** (${data.results.length} found)\n\n`;
      data.results.forEach((item: any, index: number) => {
        result += `**Result ${index + 1}:**\n`;
        
        if (item.score) {
          result += `• **Relevance:** ${(item.score * 100).toFixed(1)}%\n`;
        }
        
        if (item.clause?.text) {
          const text = item.clause.text.length > 200 
            ? item.clause.text.substring(0, 200) + '...' 
            : item.clause.text;
          result += `• **Content:** ${text}\n`;
        }
        
        if (item.clause?.type) {
          result += `• **Type:** ${this.formatKey(item.clause.type)}\n`;
        }
        
        if (item.document?.filename) {
          result += `• **Source:** ${item.document.filename}\n`;
        }
        
        if (item.document?.metadata?.policy_number) {
          result += `• **Policy:** ${item.document.metadata.policy_number}\n`;
        }
        
        result += '\n';
      });
    }
    
    // Vector search results
    if (data.vector_search_results && Array.isArray(data.vector_search_results)) {
      result += `🎯 **Advanced Search Results:**\n\n`;
      data.vector_search_results.forEach((vectorResult: any, index: number) => {
        result += `**Search ${index + 1}:** ${vectorResult.query}\n`;
        if (vectorResult.results && Array.isArray(vectorResult.results)) {
          vectorResult.results.forEach((res: any, resIndex: number) => {
            const text = res.clause?.text?.length > 150 
              ? res.clause.text.substring(0, 150) + '...' 
              : res.clause?.text || 'No content available';
            result += `  ${resIndex + 1}. ${text}\n`;
            if (res.score) {
              result += `     Relevance: ${(res.score * 100).toFixed(1)}%\n`;
            }
          });
        }
        result += '\n';
      });
    }
    
    // Metadata section
    if (data.metadata) {
      result += `📊 **Processing Metadata:**\n`;
      const meta = data.metadata;
      
      if (meta.processing_time_ms) {
        result += `• **Processing Time:** ${meta.processing_time_ms}ms\n`;
      }
      if (meta.system_version) {
        result += `• **System Version:** ${meta.system_version}\n`;
      }
      if (meta.confidence) {
        result += `• **Overall Confidence:** ${this.formatKey(meta.confidence)}\n`;
      }
      if (meta.policy_clauses_used) {
        result += `• **Policy Clauses Used:** ${meta.policy_clauses_used ? 'Yes' : 'No'}\n`;
      }
      if (meta.clauses_count) {
        result += `• **Clauses Count:** ${meta.clauses_count}\n`;
      }
      result += '\n';
    }
    
    // Timestamp
    if (data.timestamp) {
      result += `⏰ **Processed At:** ${new Date(data.timestamp).toLocaleString()}\n\n`;
    }
    
    return result.trim();
  }
  
  private static formatSearchResponse(data: any): string {
    let result = '';
    
    if (data.query) {
      result += `🔍 **Search Query:** ${data.query}\n\n`;
    }
    
    if (data.results && Array.isArray(data.results)) {
      if (data.results.length === 0) {
        result += `📋 **Results:** No matches found for your query.\n\n`;
        result += `💡 **Suggestions:**\n`;
        result += `• Try using different keywords\n`;
        result += `• Check for spelling errors\n`;
        result += `• Use more general terms\n`;
      } else {
        result += `📋 **Found ${data.results.length} Result${data.results.length > 1 ? 's' : ''}:**\n\n`;
        
        data.results.forEach((item: any, index: number) => {
          result += `**${index + 1}. `;
          
          if (item.clause?.type) {
            result += `${this.formatKey(item.clause.type)}**\n`;
          } else {
            result += `Search Result**\n`;
          }
          
          if (item.clause?.text) {
            const text = item.clause.text.length > 300 
              ? item.clause.text.substring(0, 300) + '...' 
              : item.clause.text;
            result += `${text}\n`;
          }
          
          if (item.score) {
            result += `*Relevance: ${(item.score * 100).toFixed(1)}%*\n`;
          }
          
          if (item.document?.filename) {
            result += `*Source: ${item.document.filename}*\n`;
          }
          
          result += '\n';
        });
      }
    }
    
    return result.trim();
  }
  
  private static formatDocumentAnalysis(data: any): string {
    let result = '';
    
    if (data.document_analysis) {
      result += `📄 **Document Analysis:**\n\n`;
      const analysis = data.document_analysis;
      
      if (analysis.document_type) {
        result += `• **Document Type:** ${this.formatKey(analysis.document_type)}\n`;
      }
      
      if (analysis.key_sections && Array.isArray(analysis.key_sections)) {
        result += `• **Key Sections:** ${analysis.key_sections.map(this.formatKey).join(', ')}\n`;
      }
      
      if (analysis.summary) {
        result += `• **Summary:** ${analysis.summary}\n`;
      }
      
      if (analysis.metadata) {
        result += `• **Metadata:**\n`;
        Object.entries(analysis.metadata).forEach(([key, value]) => {
          result += `  - ${this.formatKey(key)}: ${value}\n`;
        });
      }
      
      result += '\n';
    }
    
    if (data.auto_generated_qa && Array.isArray(data.auto_generated_qa)) {
      result += `❓ **Frequently Asked Questions:**\n\n`;
      data.auto_generated_qa.forEach((qa: any, index: number) => {
        result += `**Q${index + 1}:** ${qa.question}\n`;
        result += `**A${index + 1}:** ${qa.answer}\n\n`;
      });
    }
    
    return result.trim();
  }
}


export const convertJsonToText = (data: any): string => {
  let result = '';

  if (data.processing_type === 'claim') {
    // Clause-based response
    result += '📋 **Insurance Claim Analysis**\n\n';

    if (data.processing_steps?.length) {
      result += '**Processing Steps:**\n';
      data.processing_steps.forEach((step: string, i: number) => {
        result += `• ${step}\n`;
      });
      result += '\n';
    }

    const demographics = data.claim_details?.demographics;
    const medical = data.claim_details?.medical_details;
    const financial = data.claim_details?.financial_details;

    if (demographics || medical || financial) {
      result += '**Claim Details:**\n';
      if (demographics) {
        result += `• Age: ${demographics.age}, Gender: ${demographics.gender}, Location: ${demographics.location}\n`;
      }
      if (medical) {
        result += `• Procedure: ${medical.procedure}, Diagnosis: ${medical.diagnosis || 'Not Provided'}, Treatment Type: ${medical.treatment_type}\n`;
      }
      if (financial) {
        result += `• Claimed Amount: ₹${financial.claimed_amount}, Policy Tenure: ${financial.policy_tenure}\n`;
      }
      result += '\n';
    }

    if (data.query_expansion?.expanded_queries?.length) {
      result += '**Related Questions & Insights:**\n';
      data.query_expansion.expanded_queries.forEach((q: string, i: number) => {
        result += `${i + 1}. ${q}\n`;
      });
      result += '\n';
    }

    const decision = data.decision_analysis?.decision_analysis;
    if (decision) {
      result += '🧠 **Decision Analysis:**\n';
      result += `• Coverage Decision: ${decision.coverage_decision}\n`;
      result += `• Reason: ${decision.decision_reason}\n`;
      if (decision.key_factors?.length) {
        result += `• Key Factors:\n`;
        decision.key_factors.forEach((f: string) => result += `  - ${f}\n`);
      }
      result += '\n';
    }

    const docs = data.decision_analysis?.documentation_analysis;
    if (docs) {
      result += '📄 **Documentation Review:**\n';
      if (docs.required_documents?.length) {
        result += '• Required Documents:\n';
        docs.required_documents.forEach((doc: string) => result += `  - ${doc}\n`);
      }
      if (docs.missing_documents?.length) {
        result += '• Missing Documents:\n';
        docs.missing_documents.forEach((doc: string) => result += `  - ${doc}\n`);
      }
      result += '\n';
    }

    const steps = data.decision_analysis?.next_steps;
    if (steps) {
      result += '📝 **Next Steps:**\n';
      steps.immediate_actions?.forEach((step: string) => result += `• ${step}\n`);
      result += '\n';
    }

    const risk = data.decision_analysis?.risk_analysis;
    if (risk) {
      result += '⚠️ **Risk Analysis:**\n';
      result += `• Risk Level: ${risk.fraud_risk}\n`;
      result += `• Factors:\n`;
      risk.risk_factors?.forEach((f: string) => result += `  - ${f}\n`);
      result += `• Mitigation:\n`;
      risk.mitigation_strategies?.forEach((m: string) => result += `  - ${m}\n`);
      result += '\n';
    }

  } else if (data.processing_type === 'general_question') {
    // General question response
    result += '❓ **General Query Response**\n\n';

    if (data.question_analysis) {
      result += '**Query Understanding:**\n';
      Object.entries(data.question_analysis).forEach(([key, value]) => {
        if (Array.isArray(value)) {
          value.forEach((item: any) => result += `• ${key}: ${item}\n`);
        } else {
          result += `• ${key}: ${value}\n`;
        }
      });
      result += '\n';
    }

    if (data.answer_components) {
      result += '**Answer:**\n';
      if (data.answer_components.direct_answer) {
        result += `• ${data.answer_components.direct_answer}\n`;
      }
      if (data.answer_components.qualifications?.length) {
        result += `• Notes:\n`;
        data.answer_components.qualifications.forEach((q: string) => result += `  - ${q}\n`);
      }
      if (data.answer_components.examples?.length) {
        result += `• Examples:\n`;
        data.answer_components.examples.forEach((e: string) => result += `  - ${e}\n`);
      }
      result += '\n';
    }
  } else {
    // Unknown type fallback
    result += JSON.stringify(data, null, 2);
  }

  return result.trim();
};
