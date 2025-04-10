import os
import pickle
import time
import json_repair
import json
import pandas as pd
import glob
from tqdm import tqdm
from typing import Any, Dict, List
from pathlib import Path
from pydantic import BaseModel
from configparser import ConfigParser
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_experimental.graph_transformers.llm import UnstructuredRelation
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from groq import Groq
import ast
import warnings
warnings.filterwarnings("ignore")


def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # Return the value as-is if it's not a literal structure
        return val

def process_dataframe_column(df, column):
    return df[column].apply(safe_eval)


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
graph = Neo4jGraph(database=config["Neo4j_resume"]["database"])


client = Groq(api_key=config["Groq"]["api_key"])

class Element(BaseModel):
    type: str
    text: Any



def exterat_elements_from_pdf(file_path: str, metadata: dict, images: bool = False, max_char: int = 1000, new_after_n_chars: int = 800, combine: int = 200,) -> List[Document]:
    from unstructured.partition.pdf import partition_pdf

    strategy = "hi_res"
    model_name = "yolox"
    raw_pdf_elements = partition_pdf(
        filename=file_path,
        extract_images_in_pdf=images,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=max_char,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine,
        image_output_dir_path="./",
        strategy=strategy,
        model_name=model_name
    )

    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element.metadata.text_as_html)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    documents = [Document(page_content=e.text, metadata=metadata) for e in categorized_elements if e.text != ""]
    metaDoc = Document(page_content=json.dumps(metadata), metadata=metadata)
    documents.append(metaDoc)
    return documents

def handle_list_or_string(data):
    if isinstance(data, list):
        return ', '.join(data)
    return data

# Function to format education field, handling both single and multiple entries
def format_education(education):
    if isinstance(education, list):
        return ', '.join([f"{edu.get('Degree')} from {edu.get('University')} in {edu.get('Year')}" for edu in education])
    elif isinstance(education, dict):
        return f"{education.get('Degree')} from {education.get('University')} in {education.get('Year')}"
    return education

def get_structured_documents(data:dict, metadata: dict) -> List[Document]:
    sentences = []

    if data.get('Name'):
        sentences.append(f"The name of the candidate is {data.get('Name')}.")

    if data.get('Email'):
        sentences.append(f"The email of {data.get('Name')} is {data.get('Email')}.")

    if data.get('Phone'):
        sentences.append(f"The phone numbers of {data.get('Name')} are {(data.get('Phone'))}.")

    if data.get('Location'):
        sentences.append(f"{data.get('Name')} is located at {(data.get('Location'))}.")

    if data.get('Skills'):
        sentences.append(f"{data.get('Name')} is skilled in {', '.join(data.get('Skills'))}.")

    if data.get('Experience'):
        sentences.append(f"{data.get('Name')} has {data.get('Experience')} years of experience.")

    if data.get('Job_Title'):
        sentences.append(f"{data.get('Name')}'s job titles include {(data.get('Job_Title'))}.")

    if data.get('Education'):
        sentences.append(f"{data.get('Name')}'s education includes {(data.get('Education'))}.")

    if data.get('Responsibilities'):
        sentences.append(f"{data.get('Name')}'s responsibilities have included {(data.get('Responsibilities'))}.")


    documents = [Document(page_content=doc, metadata=metadata) for doc in sentences]
    return documents



examples = [
    {
        "text": (
            "Lisa-M. Ray, Project Manager | 10+ yrs mgmt, last with HP Inc. Led global teams."
        ),
        "head": "Lisa-M. Ray",
        "head_type": "Person",
        "relation": "LAST_WORKED_AT",
        "tail": "HP Inc.",
        "tail_type": "Company",
    },
    {
        "text": (
            "Contact: mike123@yahoo.com, Mob: +19876543210. Based in Atlanta, GA."
        ),
        "head": "Mike",
        "head_type": "Person",
        "relation": "CONTACT_INFO",
        "tail": "mike123@yahoo.com, +19876543210",
        "tail_type": "Contact Details",
    },
    {
        "text": (
            "Freelance Graphic Designer, '08-'15, clients incl. Adidas, BCG. Adobe Suite expert."
        ),
        "head": "Freelance Graphic Designer",
        "head_type": "Job Title",
        "relation": "CLIENTS_INCLUDE",
        "tail": "Adidas, BCG",
        "tail_type": "Clients",
    },
    {
        "text": (
            "Rahul K. | MS Comp Sci, Stanford 2019 | Interned at NVIDIA, research in AI."
        ),
        "head": "Rahul K.",
        "head_type": "Person",
        "relation": "EDUCATION",
        "tail": "MS Comp Sci, Stanford 2019",
        "tail_type": "Degree and Institution",
    },
    {
        "text": (
            "Jenna S. tech writer since '11, specialize in user manuals, Medtronic & Pfizer collab."
        ),
        "head": "Jenna S.",
        "head_type": "Person",
        "relation": "SPECIALIZES_IN",
        "tail": "user manuals",
        "tail_type": "Specialization",
    },
]

def create_prompt(document_content: str) -> str:
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a given "
        "text. You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type". The "head" '
        "key must contain the text of the extracted entity with one type.",
        "Attempt to extract as many entities and relations as you can. Maintain "
        "Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity. The knowledge graph should be coherent and easily "
        "understandable, so maintaining consistency in entity references is "
        "crucial.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))
    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)
    human_prompt = PromptTemplate(
        template="""Based on the following example, extract entities and 
relations from the provided text. Attempt to extract as many entities and relations as you can.\n\n

Below are a number of examples of text and their extracted entities and relationships.
{examples}

For the following text or table, extract entities and relations as in the provided example. Table is in HTML format.
{format_instructions}\nText: {document_content}
IMPORTANT NOTES:\n- Each key must have a valid value, 'null' is not allowed. \n- Don't add any explanation and text. \n- Extract information as much as possible""",
        input_variables=["document_content"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message_prompt])
    
    # Format the chat_prompt to generate the final string
    formatted_prompt = chat_prompt.format(document_content=document_content)
    # print(formatted_prompt)
    return formatted_prompt


def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def process_response(document) -> GraphDocument:
    
    nodes_set = set()
    relationships = []

    prompt = create_prompt(document.page_content)


    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=  "deepseek-r1-distill-llama-70b" # can be changed to any other model
    )
    parsed_json = json_repair.loads(chat_completion.choices[0].message.content)

    nodes_set = set()
    relationships = []

    for rel in parsed_json:
        try:
            # Nodes need to be deduplicated using a set
            if rel and 'head' in rel and 'tail' in rel:
                rel["head_type"] = rel["head_type"] if rel["head_type"] else "Unknown"
                rel["tail_type"] = rel["tail_type"] if rel["tail_type"] else "Unknown"
                nodes_set.add((rel["head"], rel["head_type"]))
                nodes_set.add((rel["tail"], rel["tail_type"]))
                source_node = Node(
                    id=rel["head"],
                    type=rel["head_type"]
                )
                target_node = Node(
                    id=rel["tail"],
                    type=rel["tail_type"]
                )
                relationships.append(
                    Relationship(
                        source=source_node,
                        target=target_node,
                        type=rel["relation"]
                    )
                )
        except:
            print(f"Error processing relation: {rel}")

    nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]
    return GraphDocument(nodes=nodes, relationships=relationships, source=document)


def process_document(file_path: str ,metadata_info:dict, meta: dict, images: bool = False, max_char: int = 1000, new_after_n_chars: int = 800, combine: int = 200) -> None:
    try:
        start_time = time.time()
        print(f"Starting processing for {file_path}")
        metadata = flatten_json(metadata_info)
    
   
        print(f"Extracting elements from PDF: {file_path}")
        documents = exterat_elements_from_pdf(file_path, metadata, images, max_char, new_after_n_chars, combine)
        print(f"Processing responses for {len(documents)} documents")
        structured_docs = get_structured_documents(metadata_info, metadata)
        structured_graph_document = [process_response(document) for document in (structured_docs)]
        unstructured_graph_document = [process_response(document) for document in (documents)]


        print(f"Adding {len(unstructured_graph_document)},{len(structured_graph_document)} graph documents to the graph")
        graph_docs = structured_graph_document + unstructured_graph_document
        graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
        print('Document processed')
        print('Time taken to process document:', time.time() - start_time)
    except Exception as e:
        print(f'Document not processed due to: {e}')


# Fixed path for processing

pdf_files = glob.glob('./data/resumes/*.pdf')
metadata = pd.read_excel('metadata/METADATA-FINAL.xlsx')


list_or_dict_columns = ['Phone', 'Location', 'Skills', 'Experience', 'Name','Email','Title', 'Education', 'Responsibilities']

# Apply safe_eval to each of these columns
for column in list_or_dict_columns:
    if column in metadata.columns:
        metadata[column] = process_dataframe_column(metadata, column)

if __name__ == "__main__":
    
    for full_filename in tqdm(pdf_files):
        filename = full_filename.split('/')[-1]
        print(filename)

        metadata_filtered = metadata[metadata['file_name']==filename]    
        metadata_info = {}
        if len(metadata_filtered):
            
            metadata_info['Name'] = metadata_filtered['Name'].iloc[0]
            metadata_info['Email'] = metadata_filtered['Email'].iloc[0]
            metadata_info['Phone'] = metadata_filtered['Phone'].iloc[0]
            metadata_info['Location'] = metadata_filtered['Location'].iloc[0]
            metadata_info['Skills'] = metadata_filtered['Skills'].iloc[0]
            metadata_info['Experience'] = metadata_filtered['Experience'].iloc[0]
            metadata_info['Responsibilities'] = metadata_filtered['Responsibilities'].iloc[0]
            metadata_info['Job_Title'] = metadata_filtered['Title'].iloc[0]
            metadata_info['Education'] = metadata_filtered['Education'].iloc[0]

        # print(metadata_info)
        print(f"Processing document: ",filename)
    
        processed = process_document(
            file_path=full_filename,
            metadata_info=metadata_info,
            meta={"source": full_filename},
            images=bool(config["PDF"]["extract_images"]),
            max_char=int(config["PDF"]["max_char"]),
            new_after_n_chars=int(config["PDF"]["new_after_n_chars"]),
            combine=int(config["PDF"]["combine_text_under_n_chars"])
            )
        