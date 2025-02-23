"""
RAG (Retrieval Augmented Generation) Implementation Module

This module implements a RAG system that:
1. Loads and processes PDF documents
2. Stores document embeddings in Qdrant vector store
3. Retrieves relevant context for questions
4. Generates answers using LLM

Author: Your Name
Date: 2024
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

from langchain import hub
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_auth0_ai import FGARetriever
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_openai import OpenAIEmbeddings
from typing_extensions import List, TypedDict
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain.chat_models import init_chat_model
from openfga_sdk.client.models import ClientBatchCheckItem
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    filename=f'rag_{datetime.now().strftime("%Y%m%d")}.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """Type definition for the RAG state dictionary."""
    question: str
    answer: str
    context: List[Document]
    user_id: str

class RAGError(Exception):
    """Base exception class for RAG-specific errors."""
    def __init__(self, message: str):
        super().__init__(message)
        logger.error(message)


class RAG:
    """Main RAG implementation class handling document processing and Q&A."""

    def __init__(self) -> None:
        """Initialize RAG system with necessary components."""
        try:
            load_dotenv()
            self.vector_store = None
            self.graph_builder = None
            self.compiled_graph = None
            
            # Initialize LLM
            self.llm = init_chat_model(
                model="gpt-4o-mini",
                model_provider="openai"
            )
            self.prompt = hub.pull("rlm/rag-prompt")
            self.openai_embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large"
            )
            
            logger.info("Initializing RAG system")
            self.build_rag()
            
        except Exception as e:
            raise RAGError(f"RAG initialization failed: {str(e)}")

    def setup_vector_store(self) -> None:
        """Set up Qdrant vector store connection and collection."""
        try:
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if not qdrant_url or not qdrant_api_key:
                raise RAGError("Missing Qdrant configuration in environment")
            
            qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
            
            # Create collection if it doesn't exist
            if not qdrant_client.collection_exists(collection_name="rag"):
                logger.info("Creating new Qdrant collection: rag")
                qdrant_client.create_collection(
                    collection_name="rag",
                    vectors_config=models.VectorParams(
                        size=3072,
                        distance=models.Distance.COSINE
                    )
                )
            
            self.vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name="rag",
                embedding=self.openai_embeddings,
            )
            logger.info("Vector store setup completed")
            
        except Exception as e:
            raise RAGError(f"Vector store initialization failed: {str(e)}")

    def read_documents(self) -> List[Document]:
        """Load documents from PDF file."""
        try:
            loader = PyPDFLoader("./knowledge/Sudheer_Talluri_Resume.pdf")
            documents = loader.load()

            # Add metadata for document tracking
            for doc in documents:
                doc.metadata["doc_id"] = "Sudheer_Talluri_Resume.pdf"  # Attach PDF filename
            logger.info(f"Loaded {len(documents)} documents")
            return documents
        except FileNotFoundError:
            raise RAGError("Required PDF document not found")
        except Exception as e:
            raise RAGError(f"Failed to load documents: {str(e)}")

    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create text splitter for document chunking."""
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    def retrieve(self, state: RAGState, authorize: bool = True) -> Dict[str, List[Document]]:
        """Retrieve relevant documents for the question."""
        try:
            if not authorize:
                retrieved_docs = self.vector_store.similarity_search(state["question"])
                logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
            else:
                retriever = FGARetriever(
                    retriever=self.vector_store.as_retriever(),
                    build_query=lambda doc: ClientBatchCheckItem(
                        user=f"user:{state['user_id']}",
                        object=f"doc:{doc.metadata.get('doc_id')}",
                        relation="viewer",
                    ),
                )
                retrieved_docs = retriever.invoke(state["question"])
                logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
            return {"context": retrieved_docs}

        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            raise RAGError(f"Failed to retrieve documents: {str(e)}")

    def generate_answer(self, state: RAGState) -> Dict[str, str]:
        """Generate answer based on retrieved context."""
        try:
            docs_content = "\n\n".join(
                [doc.page_content for doc in state["context"]]
            )
            messages = self.prompt.invoke(
                {
                    "question": state["question"],
                    "context": docs_content,
                }
            )
            response = self.llm.invoke(messages)
            logger.info("Successfully generated answer")
            return {"answer": response.content}
        except Exception as e:
            raise RAGError(f"Failed to generate answer: {str(e)}")

    def build_rag(self) -> None:
        """Build the RAG pipeline."""
        try:
            documents = self.read_documents()
            chunks = self.text_splitter().split_documents(documents)
            for chunk in chunks:
                chunk.metadata["doc_id"] = documents[0].metadata["doc_id"]
            self.setup_vector_store()
            self.vector_store.add_documents(documents=chunks)
            
            # Build the graph
            self.graph_builder = StateGraph(RAGState).add_sequence(
                [
                    self.retrieve,
                    self.generate_answer,
                ]
            )
            self.graph_builder.add_edge(START, "retrieve")
            self.compiled_graph = self.graph_builder.compile()
            logger.info("RAG pipeline built successfully")
            
        except Exception as e:
            raise RAGError(f"Failed to build RAG pipeline: {str(e)}")

    def display_graph(self) -> None:
        """Display the RAG graph visualization."""
        try:
            display(Image(
                self.compiled_graph.get_graph().draw_mermaid_png(
                    output_file_path="graph.png"
                )
            ))
            logger.info("Graph displayed successfully")
        except Exception as e:
            raise RAGError(f"Failed to display graph: {str(e)}")

    def invoke(self, question: str, user_id: str) -> str:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: The user's question
            
        Returns:
            str: Generated answer
        """
        try:
            response = self.compiled_graph.invoke({"question": question, "user_id": user_id})
            logger.info("Successfully processed question")
            return response["answer"]
        except Exception as e:
            raise RAGError(f"Failed to process question: {str(e)}")


def start_rag() -> None:
    """Main function to start the RAG interactive loop."""
    try:
        rag = RAG()
        logger.info("Starting RAG interactive session")
        user_id = input("Enter the user ID: ")
        action = input("Enter the action to perform, display, invoke or exit: ")
        while action != "exit":
            try:
                if action == "display":
                    rag.display_graph()
                elif action == "invoke":
                    question = input("Enter the question to ask: ")
                    print(rag.invoke(question, user_id=user_id))
                else:
                    print("Invalid action. Please try again.")
                    
                action = input(
                    "Enter the action to perform, display, invoke or exit: "
                )
                
            except RAGError as e:
                logger.error(f"Operation failed: {str(e)}")
                print(f"Operation failed: {str(e)}")
                action = input(
                    "Enter the action to perform, display, invoke or exit: "
                )
                
        logger.info("RAG session ended")
        
    except Exception as e:
        logger.critical(f"Critical error in RAG session: {str(e)}")
        print(f"Critical error occurred: {str(e)}")


if __name__ == "__main__":
    start_rag()

