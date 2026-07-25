import os
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, LLMGraphTransformer, GraphCypherQAChain

def run_graph_rag(text_content, user_question):
    # 1. Connect to the Neo4j container
    graph = Neo4jGraph(
        url="bolt://localhost:7687", 
        username="neo4j", 
        password="password"
    )
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    # 2. Knowledge Extraction (Text -> Graph)
    # The transformer uses the LLM to dynamically identify entities and relationships
    transformer = LLMGraphTransformer(llm=llm)
    
    print("Extracting graph data from text...")
    documents = [Document(page_content=text_content)]
    graph_documents = transformer.convert_to_graph_documents(documents)
    
    # Write the extracted nodes and edges to Neo4j
    graph.add_graph_documents(graph_documents)
    print("Knowledge Graph built successfully!")
    
    # 3. Retrieval (Natural Language -> Cypher -> Answer)
    # The chain reads the Neo4j schema, writes a Cypher query, and executes it
    chain = GraphCypherQAChain.from_llm(
        llm=llm, 
        graph=graph, 
        verbose=True,
        # Required by the latest LangChain updates to execute generated Cypher
        allow_dangerous_requests=True 
    )
    
    print(f"User Question: {user_question}")
    response = chain.invoke({"query": user_question})
    
    return response["result"]

# ==========================================
# Example Execution
# ==========================================
if __name__ == "__main__":
    # The unstructured context
    sample_text = """
    Acme Corp was founded by Jane Doe in 2015. 
    In 2023, GlobalTech acquired Acme Corp for $500M. 
    John Smith is the current CEO of GlobalTech.
    """
    
    # A multi-hop reasoning question that standard vector RAG would struggle with
    question = "Who is the CEO of the company that acquired the startup founded by Jane Doe?"
    
    answer = run_graph_rag(sample_text, question)
    print(f"AI Answer: {answer}")
