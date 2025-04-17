#!/usr/bin/env python3

# Dependencies and imports
import os
from dotenv import load_dotenv
import nest_asyncio
from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from uuid import uuid4
import pandas as pd
from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not LANGCHAIN_API_KEY:
    raise ValueError("LANGCHAIN_API_KEY not found in environment variables")

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Initialize asyncio for Jupyter/Colab compatibility
nest_asyncio.apply()

# Data Collection
documents = SitemapLoader(web_path="https://blog.langchain.dev/sitemap-posts.xml").load()

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=len,
)

split_chunks = text_splitter.split_documents(documents)

# Set up embedding model and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant_vectorstore = Qdrant.from_documents(
    documents=split_chunks,
    embedding=embedding_model,
    location=":memory:"
)

qdrant_retriever = qdrant_vectorstore.as_retriever()

# Set up the LLM
base_llm = ChatOpenAI(model="gpt-4o-mini", tags=["base_llm"])

# Create the RAG prompt template
base_rag_prompt_template = """\
Use the provide context to answer the provided user query. Only use the provided context to answer the query.
If the context is not related to the query respond as follows:
* a pleasant comment about their query
* indicate it is unrelated to the current topic
* ask if they have a query related to what you understand the topic to be
Finally, if you do not know the answer respond with "I don't know".

Context:
{context}

Question:
{question}
"""

base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)

# Set up the RAG chain
retrieval_augmented_qa_chain = (
    {
        "context": itemgetter("question") | qdrant_retriever,
        "question": itemgetter("question")
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {
        "response": base_rag_prompt | base_llm,
        "context": itemgetter("context")
    }
)

# Set up LangSmith
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"LangSmith - {unique_id}"

# Load test data
test_df = pd.read_csv("DataRepository/langchain_blog_test_data.csv")

# Set up LangSmith client and dataset
client = Client()
dataset_name = "langsmith-demo-dataset-aie6-triples-v3"  # Fixed dataset name

try:
    # Try to get existing dataset first
    datasets = client.list_datasets()
    existing_dataset = next(
        (ds for ds in datasets if ds.name == dataset_name),
        None
    )
    
    if existing_dataset:
        dataset = existing_dataset
        print(f"Using existing dataset: {dataset_name}")
    else:
        print(f"Creating new dataset: {dataset_name}")
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="LangChain Blog Test Questions"
        )
        
        # Only populate the dataset if we just created it
        for triplet in test_df.iterrows():
            triplet = triplet[1]
            client.create_example(
                inputs={"question": triplet["question"], "context": triplet["context"]},
                outputs={"answer": triplet["answer"]},
                dataset_id=dataset.id
            )

except Exception as e:
    print(f"Error setting up dataset: {str(e)}")
    raise

# Define evaluation preparation functions
def prepare_data_ref(run, example):
    return {
        "prediction": run.outputs["response"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"]
    }

def prepare_data_noref(run, example):
    return {
        "prediction": run.outputs["response"],
        "input": example.inputs["question"]
    }

def prepare_context_ref(run, example):
    return {
        "prediction": run.outputs["response"],
        "reference": example.inputs["context"],
        "input": example.inputs["question"]
    }

# Set up evaluators
eval_llm = ChatOpenAI(model="gpt-4o-mini", tags=["eval_llm"])

cot_qa_evaluator = LangChainStringEvaluator(
    "cot_qa",
    config={"llm": eval_llm},
    prepare_data=prepare_context_ref
)

unlabeled_dopeness_evaluator = LangChainStringEvaluator(
    "criteria",
    config={
        "criteria": {
            "dopeness": "Is the answer to the question dope, meaning cool - awesome - and legit?"
        },
        "llm": eval_llm,
    },
    prepare_data=prepare_data_noref
)

labeled_score_evaluator = LangChainStringEvaluator(
    "labeled_score_string",
    config={
        "criteria": {
            "accuracy": "Is the generated answer the same as the reference answer?"
        },
    },
    prepare_data=prepare_data_ref
)

# Run evaluation
base_rag_results = evaluate(
    retrieval_augmented_qa_chain.invoke,
    data=dataset_name,
    evaluators=[
        cot_qa_evaluator,
        unlabeled_dopeness_evaluator,
        labeled_score_evaluator,
    ],
    experiment_prefix="Base RAG Evaluation"
)

# Example usage
if __name__ == "__main__":
    # Test the chain
    response = retrieval_augmented_qa_chain.invoke(
        {"question": "What's new in LangChain v0.2?"}
    )
    print("Response:", response["response"].content)
    
    # Test with unrelated question
    response = retrieval_augmented_qa_chain.invoke(
        {"question": "What is the airspeed velocity of an unladen swallow?"}
    )
    print("\nUnrelated question response:", response["response"].content)
