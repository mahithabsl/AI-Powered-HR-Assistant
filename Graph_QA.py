import os
import re
import warnings
import streamlit as st
import ast

from pathlib import Path
from configparser import ConfigParser

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from pydantic import Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# Suppress specific Neo4j deprecation warnings
warnings.filterwarnings("ignore")

# Load configurations
def load_config():
    config = ConfigParser()
    config_path = Path(__file__).parent / "config.ini"
    config.read(config_path)
    return config

config = load_config()


# Initialize graph and LLM
os.environ["NEO4J_URI"] = config["Neo4j_resume"]["uri"]
os.environ["NEO4J_USERNAME"] = config["Neo4j_resume"]["username"]
os.environ["NEO4J_PASSWORD"] = config["Neo4j_resume"]["password"]
os.environ["NEO4J_DATABASE"] = config["Neo4j_resume"]["database"]



graph = Neo4jGraph()

llm = ChatGroq(
model_name="llama-3.3-70b-versatile", # can be changed to any other model
temperature=0,
api_key=config["Groq"]["api_key"]
)

if config["Embeddings"]["model"] == 'cohere':
    from langchain_cohere import CohereEmbeddings
    embeddings = CohereEmbeddings(
    model="embed-english-v3.0", cohere_api_key=config["Embeddings"]["api_key"]) # can be changed to any other model


vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    # print(input)
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    if len(words) > 1:
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
    else:
        full_text_query = f"{words[0]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting noun entities from the text. Don't include any explanation or text.",
            ),
            (
                "human",
                "Please extract all the noun entities into a list from the following "
                "input: {question}",
            ),
        ]
    )

    entity_chain = prompt | llm
    entities = entity_chain.invoke({"question": question})
    # Add missing quotes around list elements
    list_entities = entities.content.strip().split('\n')
    
    for entity in list_entities:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:5})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str):
    # print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
    {structured_data}
    Unstructured data:
    {"#Document ". join(unstructured_data)}
        """
    return final_data



def get_query_results():

    _search_query =  RunnableLambda(lambda x : x["question"])
    
    chatbot = llm

    template = """You are a recruitment specialist tasked with assisting HR departments with any queries to analyze candidates based on their resume information.
    Your goal is to deliver a clear and concise answer that aids HR in making informed decisions about potential candidates. 


    Answer the question based only on the following context:
    {context}

    Question: {question}
    
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": _search_query | retriever,
                "question": RunnablePassthrough()
            }
        )
        | prompt
        | chatbot
        | StrOutputParser()
    )
    return chain

def get_answer(query):
    try:
        chain = get_query_results()
        ans = chain.invoke({"question": query})
        return ans
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

def main():
    # Create columns for layout control, adjust the weight to align the image to the right
    col1, col2 = st.columns([0.15,0.85])  # Adjust the weights as needed to align the image

    with col1:  # Use the second column to display the image on the right
        st.image("./static/PrimaryLogo3color-768x594.jpg", width=200)

    st.title('AI Powered HR Assistant')
    user_query = st.text_input("Enter your query:")

    if st.button("Get Answer"):
        if user_query:  
            answer = get_answer(user_query)
            st.write(f"Answer: {answer}")
        else:
            st.write("Please enter a query to get an answer.")

if __name__ == "__main__":
    main()

