import os
import streamlit as st
import pandas as pd
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor

# =========================================================================
# 🎬 STAGE 1: VISUAL INTERFACE (Day 1 Blueprint)
# =========================================================================
st.set_page_config(page_title="Production RAG Control Room", layout="wide")
st.title("🔍 Project Milestone: Completed RAG Engine Dashboard")
st.write("---")

# Point LlamaIndex away from default external LLMs for local vector processing
Settings.llm = None

# Track local workspace paths
DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# CONFIGURATION SIDEBAR INTERFACE
with st.sidebar:
    st.header("Pipeline Parameters")
    top_k = st.slider("Select Top-K Matches:", min_value=1, max_value=5, value=3)
    
    # This directly maps your 'threshold = 0.20' notebook rule into the UI!
    confidence_threshold = st.slider(
        "Confidence Guardrail Cutoff:", 
        min_value=0.05, 
        max_value=0.90, 
        value=0.20, 
        step=0.05
    )
    st.info("💡 Any similarity vector score below this slider cutoff will be intercepted and dropped.")

# CREATE INTERACTIVE WORKSPACE LAYOUT
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ System Control Trigger")
    user_query = st.text_input("Ask System Question:", value="What is the security policy?")
    run_pipeline = st.button("Execute Vector Search Engine", type="primary")

# =========================================================================
# 🏗️ STAGE 2: PIPELINE ORCHESTRATION ENGINE (Day 2 + Day 3 Integration)
# =========================================================================
if run_pipeline:
    file_list = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    
    if not file_list:
        st.error(f"📁 Directory Empty: Please drop your policy PDFs into the '{DATA_DIR}' folder in your file tree explorer first!")
    else:
        # STEP A: INITIALIZE MODEL CONVERSIONS
        with st.spinner("1. Loading embedding model weights..."):
            Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            
        # STEP B: EXTRACT TEXT
        with st.spinner("2. Ingesting active PDFs into system memory..."):
            reader = SimpleDirectoryReader(input_dir=DATA_DIR, required_exts=[".pdf"])
            documents = reader.load_data()
            
        # STEP C: STRUCTURE MATRIX SPACE
        with st.spinner("3. Transforming strings into geometric coordinate arrays..."):
            index = VectorStoreIndex.from_documents(documents)
            
        # STEP D: RETRIEVAL WITH DYNAMIC CONFIDENCE GUARDRAILS
        with st.spinner("4. Executing vector angle checks and applying filters..."):
            # SimilarityPostprocessor handles your 'if score < threshold: reject' automatically!
            query_engine = index.as_query_engine(
                similarity_top_k=top_k,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=confidence_threshold)]
            )
            
            # Execute search query over matrix space
            response = query_engine.query(user_query)
            
        # =========================================================================
        # 📊 STAGE 3: DATA PAYLOAD ANALYSIS METRICS
        # =========================================================================
        with col2:
            st.subheader("📊 Retrieved High-Confidence Data Clusters")
            
            # If the best score is lower than our cutoff, trigger our fallback message
            if not response.source_nodes:
                st.error("❌ Guardrail Triggered: Sorry, I could not find an answer in the documents that cleared our Confidence Threshold.")
            else:
                st.balloons()
                for idx, source in enumerate(response.source_nodes):
                    # Display the exact mathematical similarity pattern score
                    st.metric(label=f"Match [{idx+1}] Vector Alignment Score", value=f"{source.score:.4f}")
                    st.write(source.node.get_content())
                    st.write("---")