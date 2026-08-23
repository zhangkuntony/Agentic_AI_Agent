from pydantic import BaseModel, Field

class DocumentRelevanceScore(BaseModel):
    """Binary relevance score for document evaluation"""
    is_relevant: bool = Field(description="Whether the document contains information relevant to the query")
    reasoning: str = Field(description="Explain for the relevance decision")

def evalute_document(document, query, llm):
    """Evaluate if a document is relevant to a query"""
    prompt = f"""You are an expert document evaluator. Your task is to determine if the following document 
    contains information relevant to the qiven query.
    Query: {query}
    
    Document content:
    {document.page_content}
    
    Analyze whether this document contains information that helps answer the query.
    """

    evaluation = llm.with_structured_output(DocumentRelevanceScore).invoke(prompt)
    return evaluation