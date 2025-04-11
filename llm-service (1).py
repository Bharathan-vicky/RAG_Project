# app/services/llm.py

import os
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

class LLMService:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM service.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.model_name = model_name
        
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]], 
                        max_tokens: int = 500) -> str:
        """
        Generate an answer based on the query and retrieved context documents.
        
        Args:
            query: User question
            context_docs: List of retrieved context documents
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated answer
        """
        # Format context for the prompt
        context_text = "\n\n".join([doc["content"] for doc in context_docs])
        
        # Create the system prompt
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the answer cannot be found in the context, acknowledge that you don't have "
            "enough information instead of making up an answer. "
            "Always cite specific parts of the context to support your answer."
        )
        
        # Create the user prompt with context and query
        user_prompt = f"""
Context information is below:
--------------------------
{context_text}
--------------------------

Based on the above context, please answer the following question:
{query}
        """
        
        try:
            # Generate completion using OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for more focused answers
                top_p=0.9
            )
            
            return response.choices[0].message["content"].strip()
            
        except Exception as e:
            # Handle API errors
            return f"Error generating answer: {str(e)}"
            
    def generate_answer_with_sources(self, query: str, context_docs: List[Dict[str, Any]], 
                                 max_tokens: int = 500) -> Dict[str, Any]:
        """
        Generate an answer with source citations.
        
        Args:
            query: User question
            context_docs: List of retrieved context documents
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with answer and sources
        """
        # Prepare document sources for citation
        sources = []
        for i, doc in enumerate(context_docs):
            sources.append({
                "id": doc.get("id", i),
                "content": doc["content"][:100] + "...",  # Preview of content
                "score": doc.get("score", 0.0)
            })
        
        # Generate the answer
        answer = self.generate_answer(query, context_docs, max_tokens)
        
        return {
            "answer": answer,
            "sources": sources
        }
