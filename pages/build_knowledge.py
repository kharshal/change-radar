# ============================================================================
# FILE: pages/build_knowledge.py (UPDATED)
# ============================================================================
"""
Build knowledge layer page for data upload and configuration.
"""
import streamlit as st
import time
from datetime import datetime, timedelta
import os
from utils.session_state import SessionState
from services.kpi_service import KPIService
from services.file_handling import FileHandling
from services.knowledge_layer import run_generation
from config.config import Config


class BuildKnowledge:
    """Build knowledge layer for data upload and KPI configuration."""
    
    @staticmethod
    def render():
        """Render the build knowledge page."""
        st.markdown(
            """
            <div class="page-header">
                <h1>Build Knowledge Layer</h1>
                <h3>Upload data and process documents to create insight dashboard</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        BuildKnowledge._render_data_layer()
        BuildKnowledge._render_business_layer()
        BuildKnowledge._render_relationship_layer()
        BuildKnowledge._render_kpi_selection()
    
    @staticmethod
    def _render_data_layer():
        """Render Data Knowledge Layer section."""
        with st.expander("1️⃣ Data Knowledge Layer", expanded=True):
            if not SessionState.get('dkl_generated'):
                st.markdown("""
                <div class="upload-box">
                    <h3 style="color: #2563eb; padding: 20px 0 0 0;">📊 Upload your Data</h3>
                </div>
                """, unsafe_allow_html=True)
                
                dkl_files = st.file_uploader(
                    "",
                    accept_multiple_files=True,
                    type=Config.ALLOWED_DATA_TYPES,
                    key="dkl_uploader"
                )
                
                # Show file info
                if dkl_files:
                    num_files, size_mb = FileHandling.get_file_info(dkl_files)
                    st.info(f"📁 {num_files} file(s) selected ({size_mb:.2f} MB)")
                
                st.markdown("**Provide Data Description**")
                business_context = st.text_area(
                    "Description",
                    placeholder="Describe your data to enhance generation accuracy...",
                    height=80,
                    label_visibility="collapsed",
                    key="business_context"
                )
                
                if st.button("Submit", type="primary", key="dkl_submit", use_container_width=True):
                    if dkl_files:
                        with st.spinner("🧠 AI is analyzing your data... This may take a few minutes."):
                            # Create session folder
                            session_folder = FileHandling.create_session_folder()
                            SessionState.set('session_folder', session_folder)
                            
                            # Save uploaded files
                            saved_files = FileHandling.save_uploaded_files(
                                dkl_files, 
                                session_folder, 
                                subfolder="data"
                            )
                            SessionState.set('data_files', dkl_files)
                            
                            for file in st.session_state.data_files:
                                table_name = file.name.replace('.csv', '').replace('.xlsx', '')
                                if table_name not in st.session_state.table_descriptions:
                                    st.session_state.table_descriptions[table_name] = {}
                            
                            # Run generation
                            data_folder = f"{session_folder}/data"
                            
                            st.session_state.warehouse_info = {'description': business_context} # REMOVE
                            
                            try:
                                
                                API_URL = st.secrets["api_url"]
                                API_KEY = st.secrets["api_key"]
                                # Call the backend function from knowledgelayer.py
                                output_file_path, processing_time= run_generation(
                                    uploaded_files=st.session_state.data_files,
                                    warehouse_info=st.session_state.warehouse_info,
                                    table_descriptions=st.session_state.table_descriptions,
                                    api_key=API_KEY, api_url=API_URL,
                                    folder_path=st.session_state.session_folder
                                )
                                SessionState.set('dkl_output', output_file_path)
                                SessionState.set('dkl_generated', True)
                                SessionState.set('processing_time', processing_time)
                                st.rerun()
                            except Exception as e:
                                st.error(f"An error occurred during generation: {e}")
                                st.exception(e)
                    else:
                        st.error("Please upload data files first!")
                        
            else:
                st.markdown("""
                <div class="success-box">
                    <p class="success-text">✅ Data Knowledge Layer Successfully Generated</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show generation details
                results = SessionState.get('dkl_results', {})
                if results:
                    st.write(f"**Session Folder:** `{SessionState.get('session_folder')}`")
                
                dkl_output = st.session_state.get('dkl_output')
                
                print("DKL File:", dkl_output)
                
                col1, col2 = st.columns(2)
                with col1:
                    if dkl_output and os.path.exists(dkl_output):                    
                        with open(dkl_output, "rb") as file:
                            st.download_button(
                                label="⬇️ Download", 
                                data=file, 
                                file_name=f"Data_Knowledge_Layer_{datetime.now().strftime('%Y%m%d')}.xlsx", 
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                                use_container_width=True
                            )
                                            
                with col2:
                    if st.button("⬆️ Upload", use_container_width=True, key="reupload_dkl"):
                        SessionState.set('dkl_generated', False)
                        SessionState.set('dkl_results', {})
                        st.rerun()
    
    @staticmethod
    def _render_business_layer():
        """Render Business Knowledge Layer section."""
        with st.expander("2️⃣ Business Knowledge Layer", expanded=True):
            if not SessionState.get('fkl_uploaded'):
                st.markdown("""
                <div class="upload-box">
                    <h4 style="color: #7c3aed; padding: 20px 14px 0 14px;">📄 Upload business process and functional documents</h4>
                </div>
                """, unsafe_allow_html=True)
                
                fkl_files = st.file_uploader(
                    "",
                    accept_multiple_files=True,
                    type=Config.ALLOWED_DOC_TYPES,
                    key="fkl_uploader",
                    disabled=not SessionState.get('dkl_generated')
                )
                
                # Show file info
                if fkl_files:
                    num_files, size_mb = FileHandling.get_file_info(fkl_files)
                    st.info(f"📁 {num_files} file(s) selected ({size_mb:.2f} MB)")
                
                if st.button("Submit", type="primary", key="fkl_submit", 
                           use_container_width=True, 
                           disabled=not SessionState.get('dkl_generated')):
                    if fkl_files:
                        with st.spinner("Processing business process files and extracting KPIs..."):
                            # Save files to docs folder
                            session_folder = SessionState.get('session_folder')
                            saved_files = FileHandling.save_uploaded_files(
                                fkl_files, 
                                session_folder, 
                                subfolder="business_knowledge"
                            )
                            SessionState.set('doc_files', saved_files)
                            
                            # Extract KPIs
                            time.sleep(1.5)
                            kpi_names = KPIService.extract_from_excel(fkl_files)
                            
                            if kpi_names:
                                SessionState.set('kpi_list', kpi_names)
                                SessionState.set('fkl_uploaded', True)
                                st.success(f"✅ Found {len(kpi_names)} KPIs from {len(saved_files)} files!")
                                st.rerun()
                            else:
                                st.error("No 'KPI Name' column found in files.")
                    else:
                        st.error("Please upload files first!")
            else:
                st.markdown("""
                <div class="success-box">
                    <p class="success-text">✅ Uploaded business documents are successfully parsed and accepted!</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"📊 Extracted {len(SessionState.get('kpi_list', []))} KPIs from documents")
                
                # Show saved files
                doc_files = SessionState.get('doc_files', [])
                
                col1, col2 = st.columns(2)
#                 with col1:
#                     st.download_button(
#                         "⬇️ Download", 
#                         data="Mock FKL", 
#                         file_name="fkl_output.json", 
#                         use_container_width=True
#                     )
                with col2:
                    if st.button("⬆️ Upload More", key="reupload_fkl", use_container_width=True):
                        SessionState.set('fkl_uploaded', False)
                        st.rerun()
    
    @staticmethod
    def _render_relationship_layer():
        """Render data Relationship Layer section."""
        with st.expander("3️⃣ Dependencies and Relationship Layer", expanded=True):
            if not SessionState.get('causal_graph_generated'):
                st.info("Extract data dependencies and relationships from the data")
                if st.button("Submit", type="primary", use_container_width=True,
                           disabled=not SessionState.get('fkl_uploaded')):
                    with st.spinner("System is extracting dependencies for you... This may take a few moments"):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.05)
                            progress_bar.progress(i + 1)
                    SessionState.set('causal_graph_generated', True)
                    st.rerun()
            else:
                st.markdown("""
                <div class="success-box">
                    <p class="success-text">✅ Successfully extracted dependencies from data</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "⬇️ Download", 
                        data="Mock Graph", 
                        file_name="causal_graph.json", 
                        use_container_width=True
                    )
                with col2:
                    st.button("⬆️ Upload", key="reupload_graph", use_container_width=True)
    
    @staticmethod
    def _render_kpi_selection():
        """Render KPI selection section."""
        with st.expander("4️⃣ KPI Selection", expanded=True):
            if SessionState.get('causal_graph_generated') and SessionState.get('kpi_list'):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**Select KPIs you want to monitor**")
                
                for kpi in SessionState.get('kpi_list', []):
                    is_selected = kpi in SessionState.get('selected_kpis', [])
                    
                    if st.checkbox(kpi, value=is_selected, key=f"kpi_{kpi}"):
                        if kpi not in SessionState.get('selected_kpis', []):
                            SessionState.get('selected_kpis').append(kpi)
                    else:
                        if kpi in SessionState.get('selected_kpis', []):
                            SessionState.get('selected_kpis').remove(kpi)
                
                if SessionState.get('selected_kpis'):
                    st.info(f"✓ Selected {len(SessionState.get('selected_kpis'))} KPI(s)")
                    if st.button("🚀 Launch Insight Dashboard", type="primary", use_container_width=True):
                        
                        with st.spinner("🔬 Running anomaly detection pipeline..."):
                            progress_bar = st.progress(0)
                            for i in range(100):
                                time.sleep(0.05)
                                progress_bar.progress(i + 1)

                        SessionState.set('show_dashboard', True)
                        st.rerun()
                            
#                             from services.pipeline_orchestrator import PipelineOrchestrator

#                             # Get data folder
#                             session_folder = SessionState.get('session_folder')
#                             data_folder = f"{session_folder}/data"

#                             # Get DKL config (you should save this when generating DKL)
#                             dkl_config = SessionState.get('dkl_config', {})

#                             # Run pipeline
#                             pipeline_results = PipelineOrchestrator.run_complete_pipeline(
#                                 data_folder=data_folder,
#                                 dkl_config=dkl_config,
#                                 selected_kpis=SessionState.get('selected_kpis'),
#                                 verbose=True
#                             )

#                             if pipeline_results['success']:
#                                 # Save results to session state
#                                 SessionState.set('anomaly_results', pipeline_results)
#                                 SessionState.set('show_dashboard', True)
#                                 st.success("✅ Anomaly detection complete!")
#                                 st.rerun()
#                             else:
#                                 st.error(f"❌ Pipeline failed: {pipeline_results.get('error')}")                        
                        
                else:
                    st.warning("⚠️ Please select at least one KPI")
            else:
                st.info("Complete previous steps to enable KPI selection")

