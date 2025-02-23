import os
from langchain import hub
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_openai import OpenAIEmbeddings
from typing_extensions import List, TypedDict
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.graph_mermaid import draw_mermaid
from langchain.text_splitter import RecursiveCharacterTextSplitter


class RAGState(TypedDict):
    question: str
    answer: str
    context: List[Document]


class RAG:
    def __init__(self):
        load_dotenv()
        self.vector_store = None
        self.graph_builder = None
        self.compiled_graph = None
        self.llm = init_chat_model(
            model="gpt-4o-mini",
            model_provider="openai"
        )
        self.prompt = hub.pull("rlm/rag-prompt")
        self.openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.build_rag()
    
    def setup_vector_store(self):
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        if not qdrant_client.collection_exists(collection_name="rag"):
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
    
    def read_documents(self):
        loader = PyPDFLoader("./knowledge/Sudheer_Talluri_Resume.pdf")
        documents = loader.load()
        return documents

    def text_splitter(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return text_splitter
    
    def retrieve(self, state: RAGState):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}
    
    def generate_answer(self, state: RAGState):
        docs_content = "\n\n".join([doc.page_content for doc in state["context"]])
        messages = self.prompt.invoke(
            {
                "question": state["question"],
                "context": docs_content,
            }
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}
    
    def build_rag(self):
        documents = self.read_documents()
        chunks = self.text_splitter().split_documents(documents)
        self.setup_vector_store()
        self.vector_store.add_documents(documents=chunks)
        self.graph_builder = StateGraph(RAGState).add_sequence(
            [
                self.retrieve,
                self.generate_answer,
            ]
        )
        self.graph_builder.add_edge(START, "retrieve")
        self.compiled_graph = self.graph_builder.compile()
    
    def display_graph(self):
        display(Image(self.compiled_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")))

    def invoke(self, question: str):
        response = self.compiled_graph.invoke({"question": question})
        return response["answer"]


def start_rag():
    rag = RAG()
    action = input("Enter the action to perform, display, invoke or exit: ")
    while action != "exit":
        if action == "display":
            rag.display_graph()
        elif action == "invoke":
            question = input("Enter the question to ask: ")
            print(rag.invoke(question))
        action = input("Enter the action to perform, display, invoke or exit: ")


if __name__ == "__main__":
    start_rag()









