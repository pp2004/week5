"""
Chat page for Azure GPT-4o Chat Application
Main chat interface with PDF RAG capabilities
"""

import streamlit as st
from config import setup_sidebar_config, validate_azure_config
from chat_engine import ask_gpt, process_uploaded_pdf, search_pdf_context, get_pdf_stats
from db import log_message
from utils import (
    initialize_session_state,
    get_user_id,
    validate_pdf_file,
    format_file_size,
    show_error_with_details
)

# ─── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat - Azure GPT-4o",
    page_icon="💬",
    layout="wide"
)

def main():
    """Main chat page function"""
    st.title("💬 Chat with Azure GPT-4o")

    # ─── Session & sidebar setup ────────────────────────────────────────────────
    initialize_session_state()
    azure_config = setup_sidebar_config()

    # Sidebar: Temperature, max_tokens, and Clear Chat
    st.sidebar.header("🎛️ Chat Settings")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Controls randomness in responses. Lower values make responses more focused."
    )
    st.session_state.temperature = temperature

    # Max tokens slider
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=st.session_state.max_tokens,
        step=100,
        help="Maximum length of the response."
    )
    st.session_state.max_tokens = max_tokens

    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.stop()   # immediately halt this run so Streamlit rewrites everything fresh

    # ─── PDF Upload & RAG ────────────────────────────────────────────────────────
    st.sidebar.header("📄 PDF Upload & RAG")
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a PDF to enable document-based conversations"
    )
    if uploaded_file is not None:
        is_valid, error_msg = validate_pdf_file(uploaded_file)
        if not is_valid:
            st.sidebar.error(error_msg)
        else:
            # If this is a newly selected PDF:
            if (not st.session_state.pdf_processed or
                st.session_state.pdf_filename != uploaded_file.name):
                st.sidebar.info(f"📄 {uploaded_file.name}")
                st.sidebar.info(f"Size: {format_file_size(uploaded_file.size)}")
                if st.sidebar.button("🔄 Process PDF"):
                    with st.spinner("Ingesting PDF..."):
                        success = process_uploaded_pdf(uploaded_file)
                        if success:
                            st.session_state.pdf_processed = True
                            st.session_state.pdf_filename = uploaded_file.name
                            st.sidebar.success("✅ PDF processed successfully!")
                        else:
                            st.sidebar.error("❌ Failed to process PDF")
            else:
                st.sidebar.success(f"✅ PDF ready: {st.session_state.pdf_filename}")
                stats = get_pdf_stats()
                if stats.get("count", 0) > 0:
                    st.sidebar.info(f"📊 {stats['count']} chunks available")

    # ─── Azure configuration check ───────────────────────────────────────────────
    is_configured = all([
        azure_config.get("endpoint"),
        azure_config.get("api_key"),
        azure_config.get("deployment_name"),
        azure_config.get("api_version")
    ])
    if not is_configured:
        st.warning("⚠️ Please configure Azure OpenAI credentials in the sidebar to start chatting.")
        st.info("Enter your Azure OpenAI endpoint, API key, deployment name, and API version in the sidebar form.")
        return

    valid_cfg, err_msg = validate_azure_config(
        azure_config["endpoint"],
        azure_config["api_key"],
        azure_config["deployment_name"],
        azure_config["api_version"]
    )
    if not valid_cfg:
        st.error(f"❌ Configuration Error: {err_msg}")
        return

    # ─── Display existing chat history ──────────────────────────────────────────
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        try:
            # Preferred: modern Streamlit chat bubbles
            with st.chat_message(role):
                st.write(content)
                if role == "assistant" and message.get("has_context"):
                    st.caption("📄 Response includes PDF context")
        except AttributeError:
            # Fallback for older Streamlit—no chat bubbles
            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Assistant:** {content}")
            if role == "assistant" and message.get("has_context"):
                st.caption("📄 Response includes PDF context")

    # ─── Chat input (form‐based fallback) ────────────────────────────────────────
    with st.form("chat_input_form", clear_on_submit=True):
        prompt = st.text_input("Type your message here...", key="chat_user_input")
        submitted = st.form_submit_button("Send")

    if submitted and prompt:
        # 1) Record user message in session history
        user_msg = {"role": "user", "content": prompt}
        st.session_state.chat_history.append(user_msg)

        # Display the user’s message in the UI
        try:
            with st.chat_message("user"):
                st.write(prompt)
        except AttributeError:
            st.markdown(f"**You:** {prompt}")

        # --- LOG THE USER MESSAGE ---
        log_message(
            user_id=get_user_id(),
            message_type="user",
            content=prompt,
            tokens_used=0,                      # user message itself does not consume tokens
            model_name="n/a",                   # you can set to "n/a" or leave default
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
            has_pdf_context=False
        )

        # 2) Gather PDF context if available
        pdf_context = []
        if st.session_state.pdf_processed:
            pdf_context = search_pdf_context(prompt)
        has_pdf_context = len(pdf_context) > 0

        # 3) Call ask_gpt with the full session history + azure_config
        try:
            response, tokens_used, success = ask_gpt(
                st.session_state.chat_history,   # entire message history
                azure_config,                    # azure_config dict
                pdf_context=pdf_context,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens
            )

            # If ask_gpt returns success == False, show an error
            if not success or response is None:
                st.error("❌ Failed to get a valid response from Azure OpenAI.")
                return

            # 4) Render the assistant’s response
            try:
                with st.chat_message("assistant"):
                    st.write(response)
                    if has_pdf_context:
                        st.caption("📄 Response includes PDF context")
            except AttributeError:
                st.markdown(f"**Assistant:** {response}")
                if has_pdf_context:
                    st.caption("📄 Response includes PDF context")

            # 5) Append that assistant message to history & log into DB
            assistant_msg = {
                "role": "assistant",
                "content": response,
                "has_context": has_pdf_context
            }
            st.session_state.chat_history.append(assistant_msg)

            log_message(
                user_id=get_user_id(),
                message_type="assistant",
                content=response,
                tokens_used=tokens_used,
                model_name=azure_config["deployment_name"],
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                has_pdf_context=has_pdf_context
            )

            # 6) Show token usage below
            st.caption(f"🔢 Tokens used: {tokens_used}")

        except Exception as e:
            show_error_with_details(
                "Chat Error",
                "An error occurred while processing your message.",
                str(e)
            )

    # ─── Session info expander ──────────────────────────────────────────────────
    if st.session_state.chat_history:
        with st.expander("📊 Session Info"):
            st.write(f"**Messages in conversation:** {len(st.session_state.chat_history)}")
            pdf_flag = "✅ Available" if st.session_state.pdf_processed else "❌ No PDF"
            st.write(f"**PDF context:** {pdf_flag}")
            st.write(f"**Temperature:** {st.session_state.temperature}")
            st.write(f"**Max Tokens:** {st.session_state.max_tokens}")

if __name__ == "__main__":
    main()
