import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import time

# Load API key
load_dotenv("key.env", override=True)

# ==============================
# INITIALIZE MODEL AND VECTOR STORE
# ==============================
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful and concise medical assistant.

CONTEXT:
{context}

QUESTION:
{question}

Using only the information from the CONTEXT, answer the QUESTION clearly.
If the answer cannot be determined from context, say you are unsure.
""")

output_str = StrOutputParser()
chain = prompt | model | output_str

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024
)

vector_store = Chroma(
    collection_name="medical",
    embedding_function=embedding,
    persist_directory="./medical_db",
)

# ==============================
# PAGE FUNCTION
# ==============================
def run_page():
    st.markdown("## 🩺 Medical AI Assistant")
    st.markdown("Ask me anything about your medical issues. I will use the medical document database to answer.")

    # Init chat history
    if "med_messages" not in st.session_state:
        st.session_state.med_messages = [
            {"role": "assistant", "content": "Hello, I'm Dr. AI. How can I help you today?"}
        ]

    # Render chat history
    for msg in st.session_state.med_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Add empty space so chat_input appears at bottom
    st.markdown("<div style='height: 140px'></div>", unsafe_allow_html=True)

    # Stream output generator
    def response_generator(text):
        for word in text.split():
            yield word + " "
            time.sleep(0.03)

    # Chat input (always at bottom)
    user_prompt = st.chat_input("Describe your symptoms…")

    if user_prompt:
        # Save user message
        st.session_state.med_messages.append(
            {"role": "user", "content": user_prompt}
        )

        # Display user message
        with st.chat_message("user"):
            st.write(user_prompt)

        # RAG search
        try:
            results = vector_store.similarity_search_by_vector(
                embedding=embedding.embed_query(user_prompt),
                k=3
            )
            context = "\n\n".join(doc.page_content for doc in results)
        except:
            context = ""

        # LLM response
        response = chain.invoke({"context": context, "question": user_prompt})

        # Display assistant streaming
        with st.chat_message("assistant"):
            st.write_stream(response_generator(response))

        # Save assistant message
        st.session_state.med_messages.append(
            {"role": "assistant", "content": response}
        )
# ==============================