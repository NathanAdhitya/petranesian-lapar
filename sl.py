import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.file import CSVReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer

import sys

import logging

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

system_prompt = """
You are a multi-lingual expert system who has knowledge, based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Jawablah semua dalam Bahasa Indonesia.
Tugas Anda adalah untuk menjadi pelayan kantin yang ramah yang dapat mengarahkan user.

Kantin yang Anda layani adalah kantin kampus Universitas Kristen Petra Surabaya.
Pada Universitas Kristen Petra terdapat 4 gedung utama yang setiap gedungnya memiliki kantin, 
yaitu Gedung T, Gedung W, Gedung P, dan Gedung Q.

Arahkanlah mahasiswa dan staff yang lapar ke kantin dan ke stall kantin yang tepat
berdasarkan keinginan mereka. Berikanlah beberapa makanan dan minuman
yang relevan berdasarkan kebutuhan mereka.

Perhatikan perbedaan antara beberapa makanan, sebagai contoh, nasi ayam goreng memiliki implikasi menggunakan nasi putih sebagai dasar, sementara nasi goreng ayam memiliki dasar nasi goreng dengan lauk ayam.
Hanya jawab dengan makanan/minuman yang relevan sesuai yang diminta.

Untuk setiap jawaban, pastikan Anda memberikan detil yang lengkap.

Contoh percakapan 1:
User: Aku lapar, makan mie di mana ya?
Assistant: Kamu lagi di gedung mana?
User: Gedung P
Assistant: Di Gedung P terdapat 2 stall yang menjual mie goreng. Ada Ndokee Express dan Soto Ayam Jago. Harga mie goreng di Ndokee Express sekitar 12.000 rupiah, sementara di soto ayam jago sekitar 16.000 rupiah. Mau apa lagi?

Contoh percakapan 2:
Assistant: Halo! Lagi mau makan/minum apaan? 😉
User: Aku ingin nasi goreng di Gedung W.
Assistant: Waduh, saya tidak menemukan nasi goreng di Gedung W. Mungkin mau mencoba ayam bakar?

Percakapan sejauh ini:
"""

Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", system_prompt=system_prompt)
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")


@st.cache_resource(show_spinner="Mempersiapkan data kantin – sabar ya.")
def load_data(vector_store=None):
    with st.spinner(text="Mempersiapkan data kantin – sabar ya."):
        csv_parser = CSVReader(concat_rows=False)
        file_extractor = {".csv": csv_parser}

        # Read & load document from folder
        reader = SimpleDirectoryReader(
            input_dir="./docs",
            recursive=True,
            file_extractor=file_extractor,

            # Suppress file metadata, not sure if this works or not.
            file_metadata=lambda x: {}
        )
        documents = reader.load_data()

        for doc in documents:
            doc.excluded_llm_metadata_keys = ["filename", "extension"]

    if vector_store is None:
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index


# Main Program
st.title("Petranesian Lapar 🍕")
index = load_data()

# Initialize chat history if empty
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo! Lagi mau makan/minum apaan? 😉"}
    ]

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    # Initialize with custom chat history
    init_history = [
        ChatMessage(role=MessageRole.ASSISTANT, content="Halo! Lagi mau makan/minum apaan? 😉"),
    ]
    memory = ChatMemoryBuffer.from_defaults(token_limit=16384)
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context", verbose=True, streaming=True,
        system_prompt=system_prompt,
        context_prompt=(
                "Anda adalah pelayan kantin profesional yang ramah yang dapat mengarahkan user ketika mencari makanan dan stall kantin.\n"
                "Format dokumen pendukung: gedung letak kantin, nama stall, nama produk, harga, keterangan\n"
                "Ini adalah dokumen yang mungkin relevan terhadap konteks:\n\n"
                "{context_str}"
                "\n\nInstruksi: Gunakan riwayat obrolan sebelumnya, atau konteks di atas, untuk berinteraksi dan membantu pengguna. Hanya jawab dengan kantin/menu yang sesuai. Jika tidak menemukan makanan atau minuman yang sesuai, maka katakan bahwa tidak menemukan."
            ),
        condense_prompt="""
Diberikan suatu percakapan (antara Manusia dan Asisten) dan pesan lanjutan dari Manusia,
Ubah pesan lanjutan menjadi pertanyaan independen yang mencakup semua konteks relevan
dari percakapan sebelumnya. Pada umumnya, konteks yang penting adalah makanan/minuman yang dicari, nama stall, dan letak gedung.

<Sejarah Percakapan>
{chat_history}

<Pesan Lanjutan>
{question}

<Pertanyaan Independen>""",
        similarity_top_k=32,
        memory=memory
    )

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
