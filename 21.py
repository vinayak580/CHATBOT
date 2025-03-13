import streamlit as st
import requests
import faiss
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import json
import string
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as FAISS_VectorStore
from langchain.schema import Document
from dataclasses import dataclass, field
from typing import List, Optional
from langgraph.graph import StateGraph



# ─────────────────────────────────────────────
# LLM & Embeddings Initialization
# ─────────────────────────────────────────────

# Set up your Groq API key (replace with your actual key)
GROQ_API_KEY = "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"
# Instantiate LLM (using ChatGroq) - note that different agents in your code use similar instantiations.
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

# Load SentenceTransformer and LangChain embeddings for property agent
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# Function Definitions
# ─────────────────────────────────────────────

def scrape_website(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
    return paragraphs

def Mrproptek(user_input):
    url = "https://www.linkedin.com/company/mrproptek/"  # Updated with actual website
    texts = scrape_website(url)
    prompt = f"""Answer the Question {user_input} and use this context {texts}"""
    response = llm.invoke(prompt).content
    return response

def get_embeddings(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)

def create_vector_store(df):
    texts = df.apply(lambda row: (', '.join(f"{col} {row[col]}" for col in df.columns)), axis=1).tolist()
    documents = [Document(page_content=text) for text in texts]
    vector_store = FAISS_VectorStore.from_documents(documents, embeddings)
    return vector_store, texts

def retrieve_relevant_docs(query, vector_store, threshold=0.7, k=6):
    retrieved_docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    relevant_docs = []
    for doc, score in retrieved_docs_with_scores:
        if score >= threshold:
            relevant_docs.append(doc)
    return relevant_docs

def generate_llm_response(user_input, retrieved_data, chat_history):
    context = "\n".join([doc.page_content for doc in retrieved_data])
    conversation_context = "\n".join([f"User: {q}\nAI: {r}" for q, r in chat_history])
    prompt = f"""
    You are an AI assistant. Answer the user's question using the relevant data below.
    Maintain conversational context using previous interactions.

    Previous conversation:
    {conversation_context}

    Context:
    {context}

    User's question: {user_input}
    """
    response = llm.invoke(prompt, max_tokens=400, temperature=0.7).content.strip()
    return response

# Google Maps related functions
#API_KEY = ""  # Replace with your actual API key

def get_distance_time(origin, destination):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": origin,
        "destinations": destination,
        "mode": "driving",
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "rows" in data and data["rows"]:
        elements = data["rows"][0]["elements"]
        time_duration = elements[0]['duration']['text']
        if elements and "distance" in elements[0]:
            return [elements[0]["distance"]["text"], time_duration]
        else:
            return ["Distance not found", "Na"]
    else:
        return ["Error fetching data.", "Error"]

def llm_integration(user_query):
    prompt = f"""
    Identify the two locations mentioned in the user's query: "{user_query}".
    Return the output strictly as a list in the format: "origin", "destination".
    Do not include any extra text or explanation.
    """
    response = llm.invoke(prompt).content
    cities = json.loads(response)
    origin = cities[0]
    destination = cities[1]
    values = get_distance_time(origin, destination)
    distance = values[0]
    time_duration = values[1]
    return [distance, time_duration]

def search_places(query, location, radius=5000):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{query} in {location}",
        "radius": radius,
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "results" in data:
        places = [
            {
                "name": place["name"],
                "address": place.get("formatted_address", "No address available")
            }
            for place in data["results"]
        ]
        return places
    else:
        return []

def llm_google_map_search(user_query):
    prompt = f"""
    Analyze the following text and extract exactly two pieces of information:
    1. The location.
    2. place type

    Return the result as a JSON list with the following format:
    ["", ""]

    Do not include any extra text.
    Text: "{user_query}"
    """
    response = llm.invoke(prompt).content.strip()
    try:
        result = json.loads(response)
    except json.JSONDecodeError as e:
        return []
    location = result[0]
    query = result[1]
    places = search_places(query, location)
    return places

# ─────────────────────────────────────────────
# Chat State & LangGraph Agents
# ─────────────────────────────────────────────

@dataclass
class ChatState:
    history: List[str] = field(default_factory=list)
    user_query: str = ""
    intent: Optional[str] = None

def classify_intent(state: ChatState):
    prompt = f"""
    Identify the intent of the user's query: "{state.user_query}".
    Possible intents:
    - details_about_mrproptek: Request for information about MrProptek Company.
    - property: Query related to real estate or properties.
    - distance_time_related_between_two_place: A question about the distance or time between two places.
    - thing_in_particular_location: A query asking for specific things in a particular location.
    Reply with only one of these intents.
    """
    response = llm.invoke(prompt).content.strip().lower()
    valid_intents = {"property", "details_about_mrproptek", "distance_time_related_between_two_place", "thing_in_particular_location"}
    intent = response if response in valid_intents else "property"
    return {"intent": intent}

def property_agent(state: ChatState):
    user_query = state.user_query
    if user_query.lower() == "exit":
        return {"history": state.history + ["Exiting chat."]}
    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        retrieved_data = retrieve_relevant_docs(user_query, st.session_state.vector_store, threshold=0.7)
        if retrieved_data:
            response = generate_llm_response(user_query, retrieved_data, [])
        else:
            response = llm.invoke(user_query, max_tokens=400, temperature=0.7).content
    else:
        response = "No property data loaded. Please upload a CSV file in the sidebar."
    return {"history": state.history + [response]}

def mrproptek_agent(state: ChatState):
    response = Mrproptek(state.user_query)
    return {"history": state.history + [response]}

def distime_agent(state: ChatState):
    values = llm_integration(state.user_query)
    distance = values[0]
    time_duration = values[1]
    return {"history": state.history + [f"Distance: {distance}", f"Time: {time_duration}"]}

def location_agent(state: ChatState):
    query = state.user_query
    prompt = f"""
    When someone wants to see nearby or similar things, extract the exact location and what exact thing the user is searching for from this query:
    \"{query}\". Provide only the exact location and the thing they are looking for, separated by a slash (/).
    """
    response = llm.invoke(prompt).content.strip()
    try:
        location, item_type = response.split("/")
        location = location.strip()
        item_type = item_type.strip()
    except ValueError:
        return {"history": state.history + ["Sorry, couldn't extract both location and shop type from your query."]}
    places = search_places(item_type, location)
    if not places:
        final_response = f"Sorry, couldn't find any {item_type} in {location}."
    else:
        places_info = "\n".join([f"Name: {place['name']}\nAddress: {place['address']}" for place in places])
        prompt2 = f"""
        I have found the following {item_type}s in {location}:

        {places_info}

        Extract relevant details and provide additional context.
        """
        final_response = llm.invoke(prompt2).content.strip()
    return {"history": state.history + [final_response]}

def route_intent(state: ChatState):
    return state.intent

# Build the LangGraph
graph = StateGraph(ChatState)
graph.add_node("classifys", classify_intent)
graph.add_node("properties_agent", property_agent)
graph.add_node("mrpropteks_agent", mrproptek_agent)
graph.add_node("distimes_agent", distime_agent)
graph.add_node("locations_agent", location_agent)
graph.add_conditional_edges("classifys", route_intent, {
    "property": "properties_agent",
    "details_about_mrproptek": "mrpropteks_agent",
    "distance_time_related_between_two_place": "distimes_agent",
    "thing_in_particular_location": "locations_agent"
})
graph.set_entry_point("classifys")
executable = graph.compile()

# ─────────────────────────────────────────────
# Streamlit App UI
# ─────────────────────────────────────────────

st.title("Multi-Functional Real Estate & Location Assistant")

# Sidebar: Upload CSV for property data
st.sidebar.header("Property Data")
uploaded_csv = st.sidebar.file_uploader("Upload CSV for Property Data", type=["csv"])
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.sidebar.success("CSV loaded successfully.")
        vector_store, texts = create_vector_store(df)
        st.session_state.vector_store = vector_store
    except Exception as e:
        st.sidebar.error(f"Error loading CSV: {e}")
else:
    st.session_state.vector_store = None

# Initialize conversation history in session state
if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Enter your query:")
user_query = st.text_input("Your Question:", key="user_input")

if st.button("Submit Query"):
    if user_query:
        # Create initial chat state
        initial_state = ChatState(history=st.session_state.history, user_query=user_query)
        # Invoke the multi-agent graph (which routes the query based on intent)
        output_state = executable.invoke(vars(initial_state))
        # Update conversation history
        st.session_state.history = output_state["history"]
        # Display conversation history
        st.subheader("Conversation History:")
        for message in st.session_state.history:
            st.write(message)
    else:
        st.warning("Please enter a query.")

if st.button("Clear Chat"):
    st.session_state.history = []
    st.success("Chat history cleared.")
