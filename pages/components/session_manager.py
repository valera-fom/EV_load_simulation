#!/usr/bin/env python3
"""
Session State Manager for EV Simulation
Allows users to save and load their entire session state
"""

import json
import os
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any
import glob
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy arrays and other non-serializable objects."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return super().default(obj)

class SessionManager:
    """Manages saving and loading of session states."""
    
    def __init__(self, sessions_dir: str = "saved_sessions"):
        self.sessions_dir = sessions_dir
        self._ensure_sessions_dir()
    
    def _ensure_sessions_dir(self):
        """Create sessions directory if it doesn't exist."""
        if not os.path.exists(self.sessions_dir):
            os.makedirs(self.sessions_dir)
    
    def save_session(self, session_name: str = None) -> bool:
        """
        Save current session state to a JSON file.
        
        Args:
            session_name: Optional custom name, otherwise auto-generated
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate filename
            if session_name:
                # Clean the name for filename
                safe_name = "".join(c for c in session_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_name = safe_name.replace(' ', '_')
                timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
                filename = f"{safe_name}_{timestamp}.json"
            else:
                timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
                filename = f"session_{timestamp}.json"
            
            filepath = os.path.join(self.sessions_dir, filename)
            
            # Collect all session state data
            session_data = self._collect_session_data()
            
            # Add metadata
            session_data['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'session_name': session_name or f"Auto-saved session {timestamp}",
                'version': '1.0'
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            return True
            
        except Exception as e:
            st.error(f"Error saving session: {e}")
            return False
    
    def load_session(self, filepath: str) -> bool:
        """
        Load session state from a JSON file.
        
        Args:
            filepath: Path to the session file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Remove metadata before loading
            metadata = session_data.pop('_metadata', {})
            
            # Load session data
            self._restore_session_data(session_data)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading session: {e}")
            return False
    
    def delete_session(self, filepath: str) -> bool:
        """
        Delete a saved session file.
        
        Args:
            filepath: Path to the session file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            st.error(f"Error deleting session: {e}")
            return False
    
    def list_saved_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of all saved sessions.
        
        Returns:
            List of session info dictionaries
        """
        sessions = []
        pattern = os.path.join(self.sessions_dir, "*.json")
        
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metadata = data.get('_metadata', {})
                sessions.append({
                    'filepath': filepath,
                    'filename': os.path.basename(filepath),
                    'name': metadata.get('session_name', 'Unknown'),
                    'saved_at': metadata.get('saved_at', 'Unknown'),
                    'size': os.path.getsize(filepath)
                })
            except Exception:
                # Skip corrupted files
                continue
        
        # Sort by saved_at (newest first)
        sessions.sort(key=lambda x: x['saved_at'], reverse=True)
        return sessions
    
    def _collect_session_data(self) -> Dict[str, Any]:
        """Collect all relevant session state data."""
        session_data = {}
        
        # Define the keys we want to save
        keys_to_save = [
            # EV Configuration
            'dynamic_ev', 'ev_soc',
            
            # Charger Configuration
            'charger_config',
            
            # Time Control
            'sim_duration', 'time_peaks',
            
            # Portugal Scenarios
            'portugal_scenario',
            
            # Optimization Strategies
            'optimization_strategy', 'active_strategies',
            
            # Time of Use
            'time_of_use_timeline', 'tou_period_count',
            
            # Data Source
            'data_source', 'power_values', 'synthetic_load_curve',
            'synthetic_timestamps', 'synthetic_metadata', 'synthetic_params',
            
            # Grid Settings
            'available_load_fraction', 'grid_mode',
            
            # Random Seed
            'random_seed', 'seed_mode',
            
            # Simulation Results (if available)
            'simulation_results',
            
            # Custom Scenarios
            'custom_scenarios',
            
            # Capacity Analysis
            'capacity_analysis_results',
            
            # Graph Controls
            'show_battery_effects', 'show_average_line', 'show_legend',
            'smooth_graph', 'show_safety_margin'
        ]
        
        for key in keys_to_save:
            if key in st.session_state:
                try:
                    # Skip matplotlib figures and other non-serializable objects
                    value = st.session_state[key]
                    if hasattr(value, 'savefig'):  # Matplotlib figure
                        continue
                    elif hasattr(value, 'figure'):  # Plotly figure
                        continue
                    elif callable(value):  # Functions
                        continue
                    else:
                        session_data[key] = value
                except Exception as e:
                    st.warning(f"Could not save {key}: {e}")
                    continue
        
        return session_data
    
    def _restore_session_data(self, session_data: Dict[str, Any]):
        """Restore session state from collected data."""
        for key, value in session_data.items():
            # Convert lists back to numpy arrays if they contain numeric data
            if isinstance(value, list):
                try:
                    # Try to convert to numpy array - if it fails, keep as list
                    st.session_state[key] = np.array(value)
                except:
                    st.session_state[key] = value
            else:
                st.session_state[key] = value

def render_session_manager_ui():
    """Render the session manager UI."""
    st.header("🔄 Session Manager")
    st.write("Save and load your complete simulation configuration.")
    
    # Initialize session manager
    session_manager = SessionManager()
    
    # Create tabs for different operations
    tab1, tab2 = st.tabs(["💾 Save Session", "📂 Load Session"])
    
    with tab1:
        st.subheader("Save Current Session")
        
        # Session name input
        session_name = st.text_input(
            "Session Name (optional)",
            placeholder="e.g., My EV Simulation Config",
            help="Give your session a descriptive name"
        )
        
        # Save button
        if st.button("💾 Save Session", type="primary"):
            if session_manager.save_session(session_name):
                st.success("✅ Session saved successfully!")
                st.rerun()
    
    with tab2:
        st.subheader("Load Saved Session")
        
        # Get list of saved sessions
        sessions = session_manager.list_saved_sessions()
        
        if not sessions:
            st.info("No saved sessions found. Save a session first!")
        else:
            # Create a selectbox for session selection
            session_options = [f"{s['name']} ({s['saved_at'][:19]})" for s in sessions]
            selected_index = st.selectbox(
                "Select a session to load:",
                range(len(session_options)),
                format_func=lambda i: session_options[i] if i < len(session_options) else "Unknown"
            )
            
            if selected_index < len(sessions):
                selected_session = sessions[selected_index]
                
                # Show session details
                st.write("**Session Details:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {selected_session['name']}")
                    st.write(f"**Saved:** {selected_session['saved_at'][:19]}")
                with col2:
                    st.write(f"**File:** {selected_session['filename']}")
                    st.write(f"**Size:** {selected_session['size']} bytes")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📂 Load Session", type="primary"):
                        if session_manager.load_session(selected_session['filepath']):
                            st.success("✅ Session loaded successfully!")
                            st.info("🔄 Refreshing page to apply changes...")
                            st.rerun()
                
                with col2:
                    if st.button("🗑️ Delete Session", type="secondary"):
                        # Show confirmation modal
                        st.warning("⚠️ Are you sure you want to delete this session?")
                        st.write(f"**Session:** {selected_session['name']}")
                        st.write(f"**File:** {selected_session['filename']}")
                        
                        # Use buttons without columns for sidebar compatibility
                        if st.button("✅ Yes, Delete", key="confirm_delete", type="primary"):
                            if session_manager.delete_session(selected_session['filepath']):
                                st.success("✅ Session deleted successfully!")
                                st.rerun()
                        
                        if st.button("❌ Cancel", key="cancel_delete"):
                            st.rerun()
    
 