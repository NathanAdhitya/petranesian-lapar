import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import CSVReader
from llama_index.core.llms import ChatMessage, MessageRole

system_prompt = """
You are a multi-lingual expert system who has knowledge, based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Bahasa utama yang Anda gunakan adalah Bahasa Indonesia.
Tugas Anda adalah untuk menjadi pelayan kantin yang ramah.
Kantin yang Anda layani adalah kantin kampus Universitas Kristen Petra Surabaya.
Pada Universitas Kristen Petra terdapat 5 gedung utama yang setiap gedungnya memiliki kantin, 
yaitu Gedung T, Gedung W, Gedung P, dan Gedung Q.

Arahkanlah mahasiswa dan staff yang lapar ke kantin dan ke stall kantin yang tepat
berdasarkan keinginan mereka. Berikanlah setidaknya 3 hingga 5 rekomendasi makanan dan minuman 
yang relevan berdasarkan kebutuhan mereka.

Untuk setiap jawaban, pastikan Anda memberikan detil yang lengkap.

Contoh:
User: Aku lapar, makan mie di mana ya?
Assistant: Kamu lagi di gedung mana?
User: Gedung P
Assistant: Di Gedung P terdapat 2 stall yang menjual mie goreng. Ada Ndokee Express dan Soto Ayam Jago. Harga mie goreng di Ndokee Express sekitar 12.000 rupiah, sementara di soto ayam jago sekitar 16.000 rupiah.
"""

llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", system_prompt=system_prompt)
embeddings = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")

@st.cache_resource()
def load_data(vector_store=None):
    with st.spinner(text="Mempersiapkan data kantin – sabar ya."):
        csv_parser = CSVReader()
        file_extractor = {".csv": csv_parser}

        # Read & load document from folder
        reader = SimpleDirectoryReader(input_dir="./docs", recursive=True, file_extractor=file_extractor)
        documents = reader.load_data()

    if vector_store is None:
        index = VectorStoreIndex.from_documents(documents)
    return index


# Main Program
st.title("Petranesian Lapar 🍕")
index = load_data()

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context", verbose=True, streaming=True,
    )

# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo! Lagi mau makan/minum apaan? 😉"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response_stream.response_gen)

    # Add user message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_stream.response})
