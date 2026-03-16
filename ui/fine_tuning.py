"""
Fine-Tuning UI — training data management and model fine-tuning.
Connected to core/model_trainer.py via background thread.
"""
import json
import threading
import streamlit as st
from datetime import datetime


def _run_training(training_data, epochs, batch_size, learning_rate, output_dir):
    try:
        from core.model_trainer import LegalModelTrainer
        trainer = LegalModelTrainer()
        trainer.train_with_validation(
            training_data=training_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        st.session_state["training_status"] = "✅ Training completed successfully!"
    except Exception as e:
        st.session_state["training_status"] = f"❌ Training failed: {e}"


def render():
    st.header("🎓 Model Fine-tuning")

    st.markdown("""
    ### Train Your Custom Legal Model
    Collect Q&A pairs from the **RAG Q&A** tab, then fine-tune here.
    """)

    st.markdown(f"### 📊 Training Dataset: **{len(st.session_state.training_data)}** samples")

    if st.session_state.training_data:
        with st.expander("👀 Preview Training Data"):
            for i, sample in enumerate(st.session_state.training_data[:5], 1):
                st.markdown(f"**Sample {i}:**")
                st.json(sample)

    st.markdown("### ➕ Add Training Samples Manually")
    with st.form("training_form"):
        q = st.text_input("Question")
        c = st.text_area("Context")
        a = st.text_area("Answer")
        if st.form_submit_button("Add Sample"):
            if q and c and a:
                st.session_state.training_data.append({"question": q, "context": c, "answer": a})
                st.success("Sample added!")
                st.rerun()
            else:
                st.warning("Please fill in all fields.")

    st.markdown("### ⚙️ Training Configuration")
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Epochs", 1, 10, 3)
        batch_size = st.selectbox("Batch Size", [2, 4, 8], index=1)
    with col2:
        learning_rate = st.select_slider(
            "Learning Rate", options=[1e-5, 5e-5, 1e-4, 5e-4], value=5e-5
        )
        output_dir = st.text_input("Output Directory", "fine_tuned_legal_model")

    if "training_status" in st.session_state and st.session_state["training_status"]:
        st.info(st.session_state["training_status"])

    if st.button("🚀 Start Fine-tuning", use_container_width=True):
        if len(st.session_state.training_data) < 10:
            st.warning(f"⚠️ Need at least 10 samples. Current: {len(st.session_state.training_data)}")
        else:
            st.session_state["training_status"] = "⏳ Training in progress… (this may take several minutes)"
            thread = threading.Thread(
                target=_run_training,
                args=(st.session_state.training_data, epochs, batch_size, learning_rate, output_dir),
                daemon=True,
            )
            thread.start()
            st.info("🎓 Fine-tuning started in the background.")

    if st.session_state.training_data:
        if st.button("💾 Export Training Data"):
            st.download_button(
                "📥 Download Training Data (JSON)",
                data=json.dumps(st.session_state.training_data, indent=2),
                file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
    else:
        st.info("📝 No training data yet. Answer questions in RAG Q&A mode and add them to training data!")
