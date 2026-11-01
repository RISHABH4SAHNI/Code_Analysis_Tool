import streamlit as st
import os
import logging
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from core.workflow_engine import WorkflowEngine
from ui.components import display_analysis_result, display_repository_stats
from config.settings import SUPPORTED_APPROACHES, DEFAULT_ANALYSIS_APPROACH, MAX_QUERY_LENGTH
from utils.helpers import validate_repository_path, clean_error_message

st.set_page_config(
    page_title="Code Analysis Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PATH_LENGTH = 500

def initialize_session_state():
    defaults = {
        'workflow_engine': None,
        'processing_complete': False,
        'current_approach': DEFAULT_ANALYSIS_APPROACH,
        'repo_stats': None,
        'last_analysis': None,
        'service_validation': None,
        'error_log': []
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def log_error(error_msg: str, exception: Exception = None):
    timestamp = datetime.now().isoformat()
    error_entry = {
        'timestamp': timestamp,
        'message': error_msg,
        'exception': str(exception) if exception else None
    }
    st.session_state.error_log.append(error_entry)
    logger.error(f"{error_msg}: {exception}")

def validate_input(text: str, max_length: int, field_name: str) -> bool:
    if not text:
        return True
    if len(text) > max_length:
        st.error(f"{field_name} is too long (max {max_length} characters)")
        return False
    return True

def safe_execute(func, error_msg: str, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_error(error_msg, e)
        st.error(f"{error_msg}: {clean_error_message(str(e))}")
        return None

def main():
    initialize_session_state()

    st.title("Code Analysis Bot")
    st.markdown("Analyze code issues using AI-powered function search and analysis")

    if st.session_state.error_log:
        with st.expander("Error Log", expanded=False):
            for error in st.session_state.error_log[-5:]:
                st.error(f"{error['timestamp']}: {error['message']}")
            if st.button("Clear Error Log"):
                st.session_state.error_log = []
                st.rerun()

    render_sidebar()
    render_main_content()

def render_sidebar():
    with st.sidebar:
        st.header("Configuration")

        st.session_state.current_approach = st.selectbox(
            "Analysis Approach",
            SUPPORTED_APPROACHES,
            index=SUPPORTED_APPROACHES.index(st.session_state.current_approach),
            help="Choose between Function Lookup Table and Call Graph approaches"
        )

        col1, col2 = st.columns(2)
        with col1:
            embedding_service = st.selectbox(
                "Embedding Service",
                ["openai", "ollama", "perplexity"],
                index=0
            )
        with col2:
            llm_service = st.selectbox(
                "LLM Service", 
                ["openai", "perplexity", "ollama"],
                index=0
            )

        engine = st.session_state.workflow_engine
        if engine and engine.analysis_approach != st.session_state.current_approach:
            st.warning("Approach changed. Click Initialize Services, then process the repository again.")

        if st.button("Initialize Services", width="stretch"):
            initialize_services(embedding_service, llm_service)

        display_service_status()
        render_history_section()

def initialize_services(embedding_service: str, llm_service: str):
    with st.spinner("Initializing services..."):
        try:
            st.session_state.workflow_engine = WorkflowEngine(
                embedding_service_type=embedding_service,
                llm_service_type=llm_service,
                analysis_approach=st.session_state.current_approach
            )

            validation = safe_execute(
                st.session_state.workflow_engine.validate_services,
                "Service validation failed"
            )

            if validation:
                st.session_state.service_validation = validation
                failed_services = [k for k, v in validation.items() if not v]

                if failed_services:
                    st.warning(f"Some services failed validation: {failed_services}")
                    st.info("Check your API keys and service configurations")
                else:
                    st.success("All services validated successfully!")

                st.success("Services initialized successfully!")
            else:
                st.error("Service validation failed")

            # A new engine has nothing indexed for this session yet
            st.session_state.processing_complete = False
            st.session_state.repo_stats = None

        except Exception as e:
            log_error("Failed to initialize services", e)
            st.error(f"Failed to initialize services: {clean_error_message(str(e))}")

def display_service_status():
    if st.session_state.service_validation:
        st.subheader("Service Status")
        for service, status in st.session_state.service_validation.items():
            icon = "✓" if status else "✗"
            st.write(f"{icon} {service.replace('_', ' ').title()}")

def render_history_section():
    if st.session_state.workflow_engine:
        st.header("Analysis History")

        history_stats = safe_execute(
            st.session_state.workflow_engine.get_analysis_history_stats,
            "Failed to get history stats"
        )

        if history_stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Analyses", history_stats.get("total_analyses", 0))
            with col2:
                if history_stats.get("total_analyses", 0) > 0:
                    st.metric("Avg Confidence", f"{history_stats.get('average_confidence', 0):.2f}")

            if history_stats.get("total_analyses", 0) > 0:
                if st.button("View Recent Analyses"):
                    show_recent_analyses()

                if st.button("Export History"):
                    export_analysis_history()

def show_recent_analyses():
    recent_analyses = safe_execute(
        st.session_state.workflow_engine.get_recent_analyses,
        "Failed to get recent analyses",
        5
    )

    if recent_analyses:
        st.subheader("Recent Analyses")
        for analysis in recent_analyses:
            timestamp = analysis['timestamp'][:19]
            query_preview = analysis['query'][:30] + "..." if len(analysis['query']) > 30 else analysis['query']

            with st.expander(f"{timestamp}: {query_preview}"):
                st.write(f"**Query:** {analysis['query']}")
                st.write(f"**Enhanced:** {analysis.get('enhanced_query', 'N/A')}")
                st.write(f"**Confidence:** {analysis.get('confidence_score', 0):.2f}")

                services = analysis.get('services_used', {})
                embedding_svc = services.get('embedding', 'unknown')
                llm_svc = services.get('llm', 'unknown')
                st.write(f"**Services:** {embedding_svc} + {llm_svc}")

                result = analysis.get('analysis_result') or 'No result available'
                display_text = result[:500] + "..." if len(result) > 500 else result
                st.markdown("**Analysis Result:**")
                st.write(display_text)

def export_analysis_history():
    output_file = f"analysis_history_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    result = safe_execute(
        st.session_state.workflow_engine.history_manager.export_history,
        "Failed to export history",
        output_file
    )

    if result:
        st.success(f"History exported to {result}")

def render_main_content():
    tab1, tab2, tab3, tab4 = st.tabs([
        "Repository Processing", 
        "Issue Analysis", 
        "Function Lookup",
        "Settings"
    ])

    with tab1:
        render_repository_processing_tab()

    with tab2:
        render_issue_analysis_tab()

    with tab3:
        render_function_lookup_tab()

    with tab4:
        render_settings_tab()

def render_repository_processing_tab():
    st.header("Process Repository")
    st.markdown("Extract functions and create embeddings from a repository")

    repo_path = st.text_input(
        "Repository Path",
        placeholder="/path/to/your/repository",
        help="Enter the full path to your repository",
        max_chars=MAX_PATH_LENGTH
    )

    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            include_tests = st.checkbox("Include test files", value=False)
        with col2:
            max_functions = st.number_input("Max functions to process", min_value=10, max_value=10000, value=1000)

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button(
            "Process Repository", 
            disabled=not st.session_state.workflow_engine,
            width="stretch"
        )
    with col2:
        if st.session_state.processing_complete:
            st.success("Repository processed!")

    if process_btn:
        process_repository(repo_path, include_tests, int(max_functions))

def process_repository(repo_path: str, include_tests: bool, max_functions: int):
    if not repo_path or not repo_path.strip():
        st.error("Please enter a repository path")
        return

    repo_path = os.path.abspath(os.path.expanduser(repo_path.strip()))

    if not validate_repository_path(repo_path):
        st.error("Invalid repository path or no Python files found")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("Processing repository... This may take a few minutes."):
        try:
            status_text.text("Extracting functions and storing embeddings...")
            progress_bar.progress(10)

            result = safe_execute(
                st.session_state.workflow_engine.process_repository,
                "Repository processing failed",
                repo_path,
                include_tests,
                max_functions
            )

            if result:
                progress_bar.progress(100)
                status_text.text("Processing complete!")

                st.session_state.processing_complete = True
                st.session_state.repo_stats = result

                display_processing_results(result)
            else:
                progress_bar.empty()
                status_text.empty()
                display_troubleshooting_info()

        except Exception as e:
            log_error("Repository processing error", e)
            display_troubleshooting_info()

def display_processing_results(result: Dict[str, Any]):
    st.success("Repository processed successfully!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Functions", result.get("total_functions", 0))
    with col2:
        st.metric("Embedded Functions", result.get("embedded_functions", 0))
    with col3:
        total = result.get("total_functions", 0)
        embedded = result.get("embedded_functions", 0)
        success_rate = f"{(embedded/total*100):.1f}%" if total > 0 else "0%"
        st.metric("Success Rate", success_rate)
    with col4:
        st.metric("Approach", result.get("approach", "unknown"))

    if result.get("skipped_files"):
        st.warning(f"{result['skipped_files']} file(s) could not be parsed and were skipped (see logs)")

    if "summary" in result:
        display_repository_stats(result["summary"])

def display_troubleshooting_info():
    st.info("**Troubleshooting:**")
    st.info("- Repository path is correct and contains Python files")
    st.info("- OpenSearch is running (check localhost:9200)")
    st.info("- API keys are properly configured in .env file")
    st.info("- Network connectivity is available for external services")

def render_issue_analysis_tab():
    st.header("Analyze Issue")
    st.markdown("Describe your code issue and get AI-powered analysis")

    if not st.session_state.processing_complete:
        st.warning("Please process a repository first in the 'Repository Processing' tab")
        return

    user_query = st.text_area(
        "Describe your issue",
        placeholder="e.g., Authentication function is throwing errors when validating JWT tokens",
        height=100,
        max_chars=MAX_QUERY_LENGTH,
        help=f"Maximum {MAX_QUERY_LENGTH} characters"
    )

    if not validate_input(user_query, MAX_QUERY_LENGTH, "Query"):
        return

    with st.expander("Additional Context (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            repo_name = st.text_input("Repository Name", max_chars=100)
            tech_stack = st.text_input("Tech Stack (comma-separated)", max_chars=200)
        with col2:
            priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])
            max_depth = None
            if st.session_state.workflow_engine.analysis_approach == "call_graph":
                max_depth = st.slider("Call Graph Depth", 1, 5, 3)

    context = {}
    if repo_name and validate_input(repo_name, 100, "Repository name"):
        context["repo_name"] = repo_name
    if tech_stack and validate_input(tech_stack, 200, "Tech stack"):
        context["tech_stack"] = [t.strip() for t in tech_stack.split(",")]
    if priority:
        context["priority"] = priority
    if max_depth is not None:
        context["call_graph_depth"] = max_depth

    if st.button("Analyze Issue", disabled=not st.session_state.workflow_engine, width="stretch"):
        analyze_issue(user_query, context)

def analyze_issue(user_query: str, context: Dict[str, Any]):
    if not user_query or not user_query.strip():
        st.error("Please describe your issue")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("Analyzing issue... This may take a minute."):
        try:
            status_text.text("Enhancing query, searching functions and analyzing with AI...")
            progress_bar.progress(10)

            result = safe_execute(
                st.session_state.workflow_engine.analyze_user_issue,
                "Issue analysis failed",
                user_query.strip(),
                context
            )

            progress_bar.progress(100)
            status_text.text("Analysis complete!")

            if result:
                st.session_state.last_analysis = result
                display_analysis_result(result)
            else:
                st.error("Analysis failed")

        except Exception as e:
            log_error("Issue analysis error", e)

def render_function_lookup_tab():
    st.header("Function Lookup Export")
    st.markdown("Export function lookup table and manage function data")

    if not st.session_state.processing_complete:
        st.warning("Please process a repository first")
        return

    if st.session_state.workflow_engine.analysis_approach == "call_graph":
        st.info("Lookup table export is only available with the function_lookup_table approach.")
        return

    col1, col2 = st.columns(2)

    with col1:
        export_format = st.selectbox("Export Format", ["json", "csv"])

    with col2:
        if st.button("Export Function Lookup Table", width="stretch"):
            export_function_lookup_table(export_format)

    if st.session_state.repo_stats and "summary" in st.session_state.repo_stats:
        st.subheader("Current Repository Statistics")
        display_repository_stats(st.session_state.repo_stats["summary"])

def export_function_lookup_table(export_format: str):
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"function_lookup_{timestamp}.{export_format}"

        result = safe_execute(
            st.session_state.workflow_engine.export_function_lookup_table,
            "Export failed",
            filename,
            export_format
        )

        if result:
            st.success(f"Function lookup table exported to {result}")

    except Exception as e:
        log_error("Export function lookup table failed", e)

def render_settings_tab():
    st.header("Settings & Configuration")

    st.subheader("System Information")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Current Approach:** {st.session_state.current_approach}")
        st.write(f"**Processing Complete:** {'Yes' if st.session_state.processing_complete else 'No'}")

    with col2:
        if st.session_state.workflow_engine:
            st.write("**Workflow Engine:** Initialized")
        else:
            st.write("**Workflow Engine:** Not initialized")

    st.subheader("Configuration")

    if st.button("Reset Application State"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Application state reset")
        st.rerun()

    if st.checkbox("Show Debug Information"):
        st.subheader("Debug Information")
        st.json({
            "session_state_keys": list(st.session_state.keys()),
            "error_count": len(st.session_state.error_log),
            "supported_approaches": SUPPORTED_APPROACHES
        })

if __name__ == "__main__":
    main()
