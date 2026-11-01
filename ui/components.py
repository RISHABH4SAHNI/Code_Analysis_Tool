import streamlit as st
from typing import Dict, Any
import plotly.express as px
import pandas as pd

def display_analysis_result(result: Dict[str, Any]):
    if "error" in result:
        st.error(result["error"])
        return
    
    st.success("Analysis Complete!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Functions Found", result.get("functions_found", 0))
    with col2:
        st.metric("Functions Analyzed", result.get("functions_analyzed", 0))
    with col3:
        confidence = result.get("confidence_score", 0)
        st.metric("Confidence Score", f"{confidence:.2f}")
        
        if confidence >= 0.8:
            st.success("High Confidence")
        elif confidence >= 0.6:
            st.warning("Medium Confidence")
        else:
            st.error("Low Confidence")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_area("Original Query", value=result.get("query", ""), height=80, disabled=True,
                     key="result_original_query")
    with col2:
        st.text_area("Enhanced Query", value=result.get("enhanced_query", result.get("query", "")), height=80,
                     disabled=True, key="result_enhanced_query")
    
    st.subheader("AI Analysis")
    st.markdown(result.get("analysis") or "No analysis available")
    
    if result.get("suggested_fixes"):
        st.subheader("Suggested Fixes")
        for fix in result["suggested_fixes"]:
            st.markdown(f"- {fix['description']}")
    
    if "search_results" in result and result["search_results"]:
        st.subheader("Relevant Functions")
        
        if len(result["search_results"]) > 1:
            relevance_data = []
            for i, search_result in enumerate(result["search_results"], 1):
                relevance_data.append({
                    "Function": search_result.function_metadata.name,
                    "Relevance Score": search_result.relevance_score,
                    "File": search_result.function_metadata.file_path.split('/')[-1]
                })
            
            df = pd.DataFrame(relevance_data)
            fig = px.bar(df, x="Function", y="Relevance Score", 
                        title="Function Relevance Scores",
                        hover_data=["File"])
            st.plotly_chart(fig, width="stretch")
        
        for i, search_result in enumerate(result["search_results"], 1):
            label = "LLM-requested context" if search_result.match_type == "additional_context" else f"Score: {search_result.relevance_score:.3f}"
            with st.expander(f"Function {i}: {search_result.function_metadata.name} ({label})"):
                display_function_info(search_result)

def display_function_info(search_result):
    if not hasattr(search_result, 'function_metadata'):
        st.error("Invalid search result format")
        return
        
    func = search_result.function_metadata
    
    function_type = "(async)" if func.is_async else "(sync)"
    st.markdown(f"**{func.name}** {function_type}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Location**")
        st.write(f"**File:** {func.file_path}")
        st.write(f"**Lines:** {func.start_line}-{func.end_line}")
        if func.class_context:
            st.write(f"**Class:** {func.class_context}")
        st.write(f"**Module:** {func.module_name}")
    
    with col2:
        st.markdown("**Properties**")
        st.write(f"**Type:** {'Async' if func.is_async else 'Sync'}")
        st.write(f"**Relevance:** {search_result.relevance_score:.3f}")
        if func.decorators:
            st.write(f"**Decorators:** {', '.join(func.decorators)}")
        if func.calls:
            st.write(f"**Calls:** {len(func.calls)} functions")
        
        error_handling = func.error_handling.get('has_try_catch', False)
        st.write(f"**Error Handling:** {'Yes' if error_handling else 'No'}")
    
    if func.calls:
        more = f" ... and {len(func.calls) - 10} more" if len(func.calls) > 10 else ""
        st.write(f"**Function Calls ({len(func.calls)}):** {', '.join(func.calls[:10])}{more}")
    
    st.markdown("**Source Code**")
    st.code(func.code or func.code_with_line_numbers, language="python")
    
    nested = getattr(search_result, 'nested_functions', None) or {}
    graph_context = nested.get('call_graph_context')
    if graph_context:
        # Call graph approach: summary of the dependency walk
        st.markdown("**Call Graph Context**")
        st.write(f"**Dependencies ({graph_context.get('dependency_count', 0)}):** "
                 f"{', '.join(graph_context.get('dependency_names', [])) or 'None'}")
        st.write(f"**Graph depth:** {graph_context.get('graph_depth', 0)}")
        st.write(f"**Files involved:** {', '.join(graph_context.get('files_involved', []))}")
        st.write(f"**Call paths to this function:** {graph_context.get('call_paths_count', 0)}")
        return
    
    # Lookup table approach: {lookup_id: {function_name, file_path, code, ...}}
    nested_functions = {k: v for k, v in nested.items() if isinstance(v, dict) and 'function_name' in v}
    if nested_functions:
        st.markdown(f"**Nested Functions ({len(nested_functions)})**")
        for nested_info in nested_functions.values():
            # Streamlit does not allow an expander inside an expander
            st.markdown(f"*{nested_info['function_name']}* - {nested_info['file_path']} "
                        f"(lines {nested_info['start_line']}-{nested_info['end_line']})")
            st.code(nested_info['code'], language="python")

def display_repository_stats(stats: Dict[str, Any]):
    if not stats:
        st.info("No statistics available")
        return
        
    st.subheader("Repository Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Functions", stats.get("total_functions", 0))
    with col2:
        st.metric("Async Functions", stats.get("async_functions", 0))
    with col3:
        st.metric("With Error Handling", stats.get("functions_with_error_handling", 0))
    with col4:
        st.metric("Unique Modules", stats.get("unique_modules", 0))
    
    if stats.get("total_functions", 0) > 0:
        chart_data = pd.DataFrame({
            "Type": ["Async Functions", "Sync Functions", "With Error Handling", "Without Error Handling"],
            "Count": [
                stats.get("async_functions", 0),
                stats.get("total_functions", 0) - stats.get("async_functions", 0),
                stats.get("functions_with_error_handling", 0),
                stats.get("total_functions", 0) - stats.get("functions_with_error_handling", 0)
            ]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(chart_data[:2], values="Count", names="Type", title="Function Types")
            st.plotly_chart(fig1, width="stretch")
        with col2:
            fig2 = px.pie(chart_data[2:], values="Count", names="Type", title="Error Handling")
            st.plotly_chart(fig2, width="stretch")
