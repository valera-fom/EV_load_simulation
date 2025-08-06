#!/usr/bin/env python3
"""
PDF Generator for EV Simulation Reports
Supports both light and dark themes with beautiful styling
"""

import hashlib
import _hashlib

# Store original function
original_md5 = hashlib.md5

# Create patched version that ignores usedforsecurity
def patched_md5(data=None, *, usedforsecurity=None, **kwargs):
    # Remove usedforsecurity from kwargs if present
    kwargs.pop('usedforsecurity', None)
    if data is not None:
        return original_md5(data, **kwargs)
    else:
        return original_md5(**kwargs)

# Apply patch
hashlib.md5 = patched_md5

import streamlit as st
import os
import tempfile
import datetime
import io
import matplotlib.pyplot as plt
import numpy as np
import base64
import json
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

# Set matplotlib to use PDF backend
matplotlib.use('Agg')

# Import ReportLab with only built-in fonts
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.colors import HexColor, black, white, blue
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
    st.success("✅ ReportLab loaded successfully!")
except ImportError as e:
    REPORTLAB_AVAILABLE = False
    st.warning(f"ReportLab not available: {e}")

def create_pdf_report(simulation_results, simulation_description, include_options, theme='light'):
    """
    Generate a beautiful PDF report with simulation results and graphs using ReportLab.
    Uses only built-in fonts and no web dependencies to avoid SSL issues.
    
    Args:
        simulation_results (dict): Simulation results from session state
        simulation_description (str): User-provided description
        include_options (dict): Dictionary of what to include in the report
        theme (str): 'light' or 'dark' theme for the PDF
    
    Returns:
        tuple: (success: bool, error_message: str, pdf_data: bytes)
    """
    if REPORTLAB_AVAILABLE:
        try:
            return create_reportlab_pdf(simulation_results, simulation_description, include_options, theme)
        except Exception as e:
            st.warning(f"ReportLab failed ({str(e)}), using matplotlib fallback...")
            return create_matplotlib_pdf(simulation_results, simulation_description, include_options)
    
    # Fall back to matplotlib if ReportLab is not available
    return create_matplotlib_pdf(simulation_results, simulation_description, include_options)


def create_reportlab_pdf(simulation_results, simulation_description, include_options, theme='light-blue'):
    """Create beautiful PDF using ReportLab with comprehensive data and professional styling."""
    try:
        # Define theme colors
        if theme == 'dark-blue':
            # Dark blue theme colors - modern dark theme like the image
            bg_color = HexColor('#0f0f0f')  # Very dark background
            text_color = HexColor('#ffffff')  # Pure white text
            title_color = HexColor('#3b82f6')  # Modern blue
            subtitle_color = HexColor('#60a5fa')  # Lighter blue
            table_header_bg = HexColor('#1f2937')  # Dark gray header
            table_row_bg1 = HexColor('#111827')  # Very dark row
            table_row_bg2 = HexColor('#1f2937')  # Slightly lighter row
            table_grid = HexColor('#374151')  # Medium gray grid
            highlight_color = HexColor('#1e40af')  # Medium blue for highlighting
        elif theme == 'light-green':
            # Light green theme colors
            bg_color = HexColor('#ffffff')
            text_color = HexColor('#000000')
            title_color = HexColor('#059669')  # Green
            subtitle_color = HexColor('#047857')  # Darker green
            table_header_bg = HexColor('#059669')
            table_row_bg1 = HexColor('#f0fdf4')  # Very light green
            table_row_bg2 = HexColor('#ffffff')
            table_grid = HexColor('#bbf7d0')  # Light green grid
            highlight_color = HexColor('#dcfce7')  # Light green for highlighting
        elif theme == 'light-brown':
            # Light brown theme colors
            bg_color = HexColor('#ffffff')
            text_color = HexColor('#000000')
            title_color = HexColor('#92400e')  # Brown
            subtitle_color = HexColor('#78350f')  # Darker brown
            table_header_bg = HexColor('#92400e')
            table_row_bg1 = HexColor('#fefce8')  # Very light brown
            table_row_bg2 = HexColor('#ffffff')
            table_grid = HexColor('#fde68a')  # Light brown grid
            highlight_color = HexColor('#fef3c7')  # Light brown for highlighting
        elif theme == 'dark-orange':
            # Dark orange theme colors
            bg_color = HexColor('#1c1917')  # Very dark background
            text_color = HexColor('#ffffff')  # Pure white text
            title_color = HexColor('#ea580c')  # Orange
            subtitle_color = HexColor('#f97316')  # Lighter orange
            table_header_bg = HexColor('#292524')  # Dark gray header
            table_row_bg1 = HexColor('#1c1917')  # Very dark row
            table_row_bg2 = HexColor('#292524')  # Slightly lighter row
            table_grid = HexColor('#44403c')  # Medium gray grid
            highlight_color = HexColor('#c2410c')  # Dark orange for highlighting
        elif theme == 'dark-green':
            # Dark green theme colors
            bg_color = HexColor('#0f172a')  # Very dark background
            text_color = HexColor('#ffffff')  # Pure white text
            title_color = HexColor('#10b981')  # Green
            subtitle_color = HexColor('#34d399')  # Lighter green
            table_header_bg = HexColor('#1e293b')  # Dark gray header
            table_row_bg1 = HexColor('#0f172a')  # Very dark row
            table_row_bg2 = HexColor('#1e293b')  # Slightly lighter row
            table_grid = HexColor('#334155')  # Medium gray grid
            highlight_color = HexColor('#059669')  # Dark green for highlighting
        else:
            # Light blue theme colors (default)
            bg_color = HexColor('#ffffff')
            text_color = HexColor('#000000')
            title_color = HexColor('#1f77b4')
            subtitle_color = HexColor('#2c3e50')
            table_header_bg = HexColor('#1f77b4')
            table_row_bg1 = HexColor('#f8f9fa')
            table_row_bg2 = HexColor('#ffffff')
            table_grid = HexColor('#dee2e6')
            highlight_color = HexColor('#e3f2fd')  # Light blue for highlighting
        
        def get_table_style(highlight_rows=None, center_align=False):
            """Helper function to create theme-aware table styles."""
            style = [
                ('BACKGROUND', (0, 0), (-1, 0), table_header_bg),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER' if center_align else 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), table_row_bg1),
                ('GRID', (0, 0), (-1, -1), 1, table_grid),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [table_row_bg1, table_row_bg2]),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                ('TEXTCOLOR', (0, 1), (-1, -1), text_color),
            ]
            
            # Add highlighting for specific rows
            if highlight_rows:
                for row_idx in highlight_rows:
                    style.extend([
                        ('BACKGROUND', (0, row_idx), (-1, row_idx), highlight_color),
                        ('FONTNAME', (0, row_idx), (-1, row_idx), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, row_idx), (-1, row_idx), 11),
                    ])
            
            return style
        
        # Create PDF in memory
        pdf_buffer = io.BytesIO()
        
        # Create the PDF document with proper margins and metadata
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=1.5*cm,
            bottomMargin=1.5*cm
        )
        
        # Set page background color for dark themes
        if theme in ['dark-blue', 'dark-orange', 'dark-green']:
            def add_page_background(canvas, doc):
                canvas.setFillColor(bg_color)
                canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1)
        else:
            add_page_background = None

        
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom styles with beautiful formatting
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=title_color,
            fontName='Helvetica-Bold',
            spaceBefore=10
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=15,
            spaceBefore=25,
            textColor=subtitle_color,
            fontName='Helvetica-Bold',
            leftIndent=0
        )
        
        section_style = ParagraphStyle(
            'CustomSection',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=20,
            textColor=title_color,
            fontName='Helvetica-Bold',
            leftIndent=0
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            fontName='Helvetica',
            leftIndent=0,
            textColor=text_color
        )
        
        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontSize=12,
            fontName='Helvetica-Bold',
            textColor=white,
            alignment=TA_CENTER
        )
        
        table_data_style = ParagraphStyle(
            'TableData',
            parent=styles['Normal'],
            fontSize=10,
            fontName='Helvetica',
            alignment=TA_LEFT
        )
        
        # Build the story (content)
        story = []
        
        # Title page with beautiful header
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph("EV Load Simulation Report", title_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Generated on: {current_date}", normal_style))
        story.append(Spacer(1, 20))
        
        # Description section (always include if not empty)
        if simulation_description and simulation_description.strip():
            story.append(Paragraph("📝 Simulation Description", subtitle_style))
            story.append(Paragraph(simulation_description, normal_style))
            story.append(Spacer(1, 15))
        
        # Enhanced Parameters section
        if include_options.get('parameters', False):
            story.append(Paragraph("Simulation Parameters", subtitle_style))
            
            # Extract comprehensive parameters from simulation results
            params = {}
            if 'simulation_parameters' in simulation_results:
                params = simulation_results['simulation_parameters']
            
            # Add additional comprehensive parameters from session state
            additional_params = {}
            
            # Basic simulation info
            additional_params.update({
                'Total EVs': simulation_results.get('total_evs', st.session_state.get('total_evs', 'N/A')),
                'Simulation Duration': f"{simulation_results.get('sim_duration', st.session_state.get('sim_duration', 'N/A'))} hours",
                'Grid Mode': simulation_results.get('grid_mode', 'N/A'),
                'Available Load Fraction': f"{st.session_state.get('available_load_fraction', 0.8):.0%}",
            })
            
            # EV Configuration details
            ev_config = {}
            if 'ev_calculator_results' in st.session_state:
                ev_results = st.session_state.ev_calculator_results
                if 'battery_capacity' in ev_results:
                    ev_config['Battery Capacity'] = f"{ev_results['battery_capacity']} kWh"
                if 'charging_power' in ev_results:
                    ev_config['Charging Power'] = f"{ev_results['charging_power']} kW"
                if 'initial_soc' in ev_results:
                    ev_config['Initial SOC'] = f"{ev_results['initial_soc']}%"
            
            # Charger Configuration details
            charger_config = {}
            if 'charger_calculator_results' in st.session_state:
                charger_results = st.session_state.charger_calculator_results
                if 'total_capacity' in charger_results:
                    charger_config['Charging Power'] = f"{charger_results['total_capacity']} kW"
            
        
        
            
            # Merge parameters, prioritizing simulation_results
            all_params = {**additional_params, **params}
            
            # Add EV and Charger configurations as separate sections
            if ev_config:
                all_params['🚗 EV Configuration'] = ""
                for key, value in ev_config.items():
                    all_params[f"  • {key}"] = value
            
            if charger_config:
                all_params['🔌 Charger Configuration'] = ""
                for key, value in charger_config.items():
                    all_params[f"  • {key}"] = value
            
            # Create table data
            table_data = [['Parameter', 'Value']]
            for key, value in all_params.items():
                if isinstance(value, bool):
                    formatted_value = "True" if value else "False"
                elif isinstance(value, (int, float)):
                    formatted_value = f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
                else:
                    formatted_value = str(value)
                table_data.append([key, formatted_value])
            
            # Create table with beautiful styling
            table = Table(table_data, colWidths=[4*inch, 3*inch])
            table.setStyle(TableStyle(get_table_style()))
            
            story.append(table)
            story.append(Spacer(1, 15))
        

        
        # Optimization Summary section
        if include_options.get('optimization', False):
            story.append(Paragraph("Optimization Summary", subtitle_style))
            
            # Extract optimization details from session state
            optimization_summary = {}
            
            # Time of Use details
            time_of_use_applied = 'smart_charging' in st.session_state.get('active_strategies', [])
            optimization_summary['Time of Use Applied'] = time_of_use_applied
            
            # PV + Battery details - only show if applied
            pv_battery_applied = 'pv_battery' in st.session_state.get('active_strategies', [])
            optimization_summary['PV + Battery Applied'] = pv_battery_applied
            if pv_battery_applied:
                optimization_summary['PV Adoption %'] = f"{st.session_state.optimization_strategy.get('pv_adoption_percent', 0)}%"
                optimization_summary['Battery Capacity'] = f"{st.session_state.optimization_strategy.get('battery_capacity', 0)} kWh"
                optimization_summary['Max Charge Rate'] = f"{st.session_state.optimization_strategy.get('max_charge_rate', 0)} kW"
                optimization_summary['Max Discharge Rate'] = f"{st.session_state.optimization_strategy.get('max_discharge_rate', 0)} kW"
                optimization_summary['PV Start Hour'] = f"{st.session_state.optimization_strategy.get('pv_start_hour', 8)}:00"
                optimization_summary['PV Duration'] = f"{st.session_state.optimization_strategy.get('pv_duration', 8)} hours"
                optimization_summary['Solar Energy %'] = f"{st.session_state.optimization_strategy.get('solar_energy_percent', 70)}%"
            
            # Grid Battery details - only show if applied
            grid_battery_applied = 'grid_battery' in st.session_state.get('active_strategies', [])
            optimization_summary['Grid Battery Applied'] = grid_battery_applied
            if grid_battery_applied:
                optimization_summary['Grid Battery Adoption %'] = f"{st.session_state.optimization_strategy.get('grid_battery_adoption_percent', 0)}%"
                optimization_summary['Grid Battery Capacity'] = f"{st.session_state.optimization_strategy.get('grid_battery_capacity', 0)} kWh"
                optimization_summary['Grid Battery Max Rate'] = f"{st.session_state.optimization_strategy.get('grid_battery_max_rate', 0)} kW"
                optimization_summary['Grid Battery Charge Start'] = f"{st.session_state.optimization_strategy.get('grid_battery_charge_start_hour', 7)}:00"
                optimization_summary['Grid Battery Charge Duration'] = f"{st.session_state.optimization_strategy.get('grid_battery_charge_duration', 8)} hours"
                optimization_summary['Grid Battery Discharge Start'] = f"{st.session_state.optimization_strategy.get('grid_battery_discharge_start_hour', 18)}:00"
                optimization_summary['Grid Battery Discharge Duration'] = f"{st.session_state.optimization_strategy.get('grid_battery_discharge_duration', 4)} hours"
            
            # V2G details - only show if applied
            v2g_applied = 'v2g' in st.session_state.get('active_strategies', [])
            optimization_summary['V2G Applied'] = v2g_applied
            if v2g_applied:
                optimization_summary['V2G Adoption %'] = f"{st.session_state.optimization_strategy.get('v2g_adoption_percent', 0)}%"
                optimization_summary['V2G Max Discharge Rate'] = f"{st.session_state.optimization_strategy.get('v2g_max_discharge_rate', 0)} kW"
                optimization_summary['V2G Discharge Start'] = f"{st.session_state.optimization_strategy.get('v2g_discharge_start_hour', 18)}:00"
                optimization_summary['V2G Discharge Duration'] = f"{st.session_state.optimization_strategy.get('v2g_discharge_duration', 3)} hours"
                optimization_summary['V2G Recharge Arrival'] = f"{st.session_state.optimization_strategy.get('v2g_recharge_arrival_hour', 26)}:00"
            
            # Create table data
            table_data = [['Parameter', 'Value']]
            strategy_rows = []  # Track rows that need highlighting
            
            for key, value in optimization_summary.items():
                if isinstance(value, bool):
                    formatted_value = "True" if value else "False"
                elif isinstance(value, (int, float)):
                    formatted_value = f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                # Check if this is a strategy row (ends with "Applied")
                if key.endswith("Applied"):
                    strategy_rows.append(len(table_data))
                
                table_data.append([key, formatted_value])
            
            # Create table with beautiful styling
            table = Table(table_data, colWidths=[4*inch, 3*inch])
            table.setStyle(TableStyle(get_table_style(strategy_rows)))
            
            story.append(table)
            story.append(Spacer(1, 15))
        

        
        # Graphs section with proper scaling
        if include_options.get('graphs', False):
            story.append(Paragraph("Load Curves", subtitle_style))
            
            # Try to get the main simulation figure
            main_fig = None
            if 'main_simulation_figure' in st.session_state:
                main_fig = st.session_state.main_simulation_figure
            
            if main_fig:
                # Save figure as image with enhanced vibrancy and contrast
                img_buffer = io.BytesIO()
                
                # Set figure properties for better visibility in PDF with theme-appropriate background
                if theme == 'dark-blue':
                    # Dark blue theme for graphs - much lighter background
                    main_fig.patch.set_facecolor('#9ca3af')  # Much lighter gray background
                    main_fig.patch.set_alpha(1.0)
                    
                    # Enhance all axes for better visibility
                    for ax in main_fig.axes:
                        # Set background to much lighter for better contrast
                        ax.set_facecolor('#d1d5db')  # Very light gray background
                        ax.patch.set_alpha(1.0)
                        
                        # Make grid lines more visible with dark background
                        ax.grid(True, alpha=0.4, linewidth=0.8, color='#6b7280')
                        
                        # Enhance line colors and thickness for better visibility
                        for line in ax.lines:
                            line.set_linewidth(2.5)  # Even thicker lines
                            line.set_alpha(1.0)      # Full opacity
                        
                        # Enhance text elements for better contrast
                        ax.tick_params(axis='both', which='major', labelsize=11, colors='#000000')
                        ax.xaxis.label.set_size(13)
                        ax.yaxis.label.set_size(13)
                        ax.title.set_size(15)
                        
                        # Ensure title and axis labels are dark for better contrast
                        ax.xaxis.label.set_color('#000000')
                        ax.yaxis.label.set_color('#000000')
                        ax.title.set_color('#000000')
                        
                        # Set axis colors for better visibility
                        ax.spines['bottom'].set_color('#000000')
                        ax.spines['top'].set_color('#000000')
                        ax.spines['left'].set_color('#000000')
                        ax.spines['right'].set_color('#000000')
                    
                    # Save with higher DPI and better compression
                    main_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                                   facecolor='#9ca3af', edgecolor='none', 
                                   pad_inches=0.1, transparent=False)
                elif theme == 'light-green':
                    # Light green theme for graphs
                    main_fig.patch.set_facecolor('#f0fdf4')  # Very light green background
                    main_fig.patch.set_alpha(1.0)
                    
                    # Enhance all axes for better visibility
                    for ax in main_fig.axes:
                        # Set background to light green for better contrast
                        ax.set_facecolor('#f7fee7')  # Slightly darker than figure background
                        ax.patch.set_alpha(1.0)
                        
                        # Make grid lines more visible with darker background
                        ax.grid(True, alpha=0.4, linewidth=0.8, color='#84cc16')
                        
                        # Enhance line colors and thickness for better visibility
                        for line in ax.lines:
                            line.set_linewidth(2.5)  # Even thicker lines
                            line.set_alpha(1.0)      # Full opacity
                        
                        # Enhance text elements for better contrast
                        ax.tick_params(axis='both', which='major', labelsize=11, colors='#333333')
                        ax.xaxis.label.set_size(13)
                        ax.yaxis.label.set_size(13)
                        ax.title.set_size(15)
                        
                        # Set axis colors for better visibility
                        ax.spines['bottom'].set_color('#333333')
                        ax.spines['top'].set_color('#333333')
                        ax.spines['left'].set_color('#333333')
                        ax.spines['right'].set_color('#333333')
                    
                    # Save with higher DPI and better compression
                    main_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                                   facecolor='#f0fdf4', edgecolor='none', 
                                   pad_inches=0.1, transparent=False)
                elif theme == 'light-brown':
                    # Light brown theme for graphs
                    main_fig.patch.set_facecolor('#fefce8')  # Very light brown background
                    main_fig.patch.set_alpha(1.0)
                    
                    # Enhance all axes for better visibility
                    for ax in main_fig.axes:
                        # Set background to light brown for better contrast
                        ax.set_facecolor('#fef9c3')  # Slightly darker than figure background
                        ax.patch.set_alpha(1.0)
                        
                        # Make grid lines more visible with darker background
                        ax.grid(True, alpha=0.4, linewidth=0.8, color='#eab308')
                        
                        # Enhance line colors and thickness for better visibility
                        for line in ax.lines:
                            line.set_linewidth(2.5)  # Even thicker lines
                            line.set_alpha(1.0)      # Full opacity
                        
                        # Enhance text elements for better contrast
                        ax.tick_params(axis='both', which='major', labelsize=11, colors='#333333')
                        ax.xaxis.label.set_size(13)
                        ax.yaxis.label.set_size(13)
                        ax.title.set_size(15)
                        
                        # Set axis colors for better visibility
                        ax.spines['bottom'].set_color('#333333')
                        ax.spines['top'].set_color('#333333')
                        ax.spines['left'].set_color('#333333')
                        ax.spines['right'].set_color('#333333')
                    
                    # Save with higher DPI and better compression
                    main_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                                   facecolor='#fefce8', edgecolor='none', 
                                   pad_inches=0.1, transparent=False)
                elif theme == 'dark-orange':
                    # Dark orange theme for graphs - lighter background
                    main_fig.patch.set_facecolor('#78716c')  # Medium gray background
                    main_fig.patch.set_alpha(1.0)
                    
                    # Enhance all axes for better visibility
                    for ax in main_fig.axes:
                        # Set background to lighter for better contrast
                        ax.set_facecolor('#a8a29e')  # Lighter gray background
                        ax.patch.set_alpha(1.0)
                        
                        # Make grid lines more visible with dark background
                        ax.grid(True, alpha=0.4, linewidth=0.8, color='#57534e')
                        
                        # Enhance line colors and thickness for better visibility
                        for line in ax.lines:
                            line.set_linewidth(2.5)  # Even thicker lines
                            line.set_alpha(1.0)      # Full opacity
                        
                        # Enhance text elements for better contrast
                        ax.tick_params(axis='both', which='major', labelsize=11, colors='#000000')
                        ax.xaxis.label.set_size(13)
                        ax.yaxis.label.set_size(13)
                        ax.title.set_size(15)
                        
                        # Ensure title and axis labels are dark for better contrast
                        ax.xaxis.label.set_color('#000000')
                        ax.yaxis.label.set_color('#000000')
                        ax.title.set_color('#000000')
                        
                        # Set axis colors for better visibility
                        ax.spines['bottom'].set_color('#000000')
                        ax.spines['top'].set_color('#000000')
                        ax.spines['left'].set_color('#000000')
                        ax.spines['right'].set_color('#000000')
                    
                        # Save with higher DPI and better compression
                        main_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                                    facecolor='#78716c', edgecolor='none', 
                                    pad_inches=0.1, transparent=False)
                elif theme == 'dark-green':
                    # Dark green theme for graphs - lighter background
                    main_fig.patch.set_facecolor('#475569')  # Medium gray background
                    main_fig.patch.set_alpha(1.0)
                    
                    # Enhance all axes for better visibility
                    for ax in main_fig.axes:
                        # Set background to lighter for better contrast
                        ax.set_facecolor('#64748b')  # Lighter gray background
                        ax.patch.set_alpha(1.0)
                        
                        # Make grid lines more visible with dark background
                        ax.grid(True, alpha=0.4, linewidth=0.8, color='#475569')
                        
                        # Enhance line colors and thickness for better visibility
                        for line in ax.lines:
                            line.set_linewidth(2.5)  # Even thicker lines
                            line.set_alpha(1.0)      # Full opacity
                        
                        # Enhance text elements for better contrast
                        ax.tick_params(axis='both', which='major', labelsize=11, colors='#000000')
                        ax.xaxis.label.set_size(13)
                        ax.yaxis.label.set_size(13)
                        ax.title.set_size(15)
                        
                        # Ensure title and axis labels are dark for better contrast
                        ax.xaxis.label.set_color('#000000')
                        ax.yaxis.label.set_color('#000000')
                        ax.title.set_color('#000000')
                        
                        # Set axis colors for better visibility
                        ax.spines['bottom'].set_color('#000000')
                        ax.spines['top'].set_color('#000000')
                        ax.spines['left'].set_color('#000000')
                        ax.spines['right'].set_color('#000000')
                    
                    # Save with higher DPI and better compression
                    main_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                                   facecolor='#475569', edgecolor='none', 
                                   pad_inches=0.1, transparent=False)
                else:
                    # Light blue theme for graphs (default)
                    main_fig.patch.set_facecolor('#f0f0f0')  # Light gray background
                    main_fig.patch.set_alpha(1.0)
                    
                    # Enhance all axes for better visibility
                    for ax in main_fig.axes:
                        # Set background to light gray for better contrast
                        ax.set_facecolor('#f5f5f5')  # Slightly darker than figure background
                        ax.patch.set_alpha(1.0)
                        
                        # Make grid lines more visible with darker background
                        ax.grid(True, alpha=0.4, linewidth=0.8, color='#666666')
                        
                        # Enhance line colors and thickness for better visibility
                        for line in ax.lines:
                            line.set_linewidth(2.5)  # Even thicker lines
                            line.set_alpha(1.0)      # Full opacity
                        
                        # Enhance text elements for better contrast
                        ax.tick_params(axis='both', which='major', labelsize=11, colors='#333333')
                        ax.xaxis.label.set_size(13)
                        ax.yaxis.label.set_size(13)
                        ax.title.set_size(15)
                        
                        # Set axis colors for better visibility
                        ax.spines['bottom'].set_color('#333333')
                        ax.spines['top'].set_color('#333333')
                        ax.spines['left'].set_color('#333333')
                        ax.spines['right'].set_color('#333333')
                    
                    # Save with higher DPI and better compression
                    main_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                                   facecolor='#f0f0f0', edgecolor='none', 
                                   pad_inches=0.1, transparent=False)
                img_buffer.seek(0)
                
                # Add image to PDF with full width (maintain aspect ratio)
                img = Image(img_buffer, width=7.5*inch, height=5.5*inch, kind='proportional')
                story.append(img)
            else:
                story.append(Paragraph("No simulation graphs available", normal_style))
            
            story.append(Spacer(1, 15))
        
        # Enhanced Performance Metrics section
        if include_options.get('summary', False):
            story.append(Paragraph("Performance Metrics", subtitle_style))
            
            # Extract comprehensive statistics
            stats = {}
            if 'summary_stats' in simulation_results:
                stats = simulation_results['summary_stats']
            
            # Calculate additional comprehensive statistics
            load_curve = simulation_results.get('load_curve', [])
            if load_curve is not None and len(load_curve) > 0:
                # Ensure load_curve is a numpy array
                if not isinstance(load_curve, np.ndarray):
                    load_curve = np.array(load_curve)
                
                # Calculate peak value once to avoid multiple calculations
                peak_load = np.max(load_curve)
                mean_load = np.mean(load_curve)
                
                # Convert step indices to hours (assuming 1-minute resolution)
                peak_step = np.argmax(load_curve)
                min_step = np.argmin(load_curve)
                peak_hour = peak_step // 60  # Convert minutes to hours
                min_hour = min_step // 60    # Convert minutes to hours
                
                # Calculate hours above peak thresholds
                hours_above_90 = np.sum((load_curve > (0.9 * peak_load)).astype(int)) // 60  # Convert minutes to hours
                hours_above_80 = np.sum((load_curve > (0.8 * peak_load)).astype(int)) // 60  # Convert minutes to hours
                hours_above_70 = np.sum((load_curve > (0.7 * peak_load)).astype(int)) // 60  # Convert minutes to hours
                
                additional_stats = {
                    'Maximum Load': peak_load,
                    'Average Load': mean_load,
                    'Minimum Load': np.min(load_curve),
                    'Load Range': peak_load - np.min(load_curve),
                    'Peak-to-Average Ratio': peak_load / mean_load if float(mean_load) > 0 else 0,
                    'Peak Load Time': f"Hour {peak_hour}",
                    'Minimum Load Time': f"Hour {min_hour}",
                }
                
                # Add performance metrics from simulation results
                performance_metrics = {}
                # Total EVs and simulation duration
                if 'total_evs' in simulation_results:
                    performance_metrics['Total EVs Simulated'] = simulation_results['total_evs']
                
                if 'sim_duration' in simulation_results:
                    performance_metrics['Simulation Duration'] = f"{simulation_results['sim_duration']} hours"
                
                # Grid compliance metrics (if available)
                if 'grid_limit' in simulation_results and simulation_results['grid_limit'] is not None:
                    grid_limit = np.array(simulation_results['grid_limit'])
                    # Fix numpy array boolean evaluation and shape mismatch
                    load_curve_array = np.array(load_curve)
                    
                    # Ensure arrays have the same shape
                    if load_curve_array.shape != grid_limit.shape:
                        # Resize the shorter array to match the longer one
                        min_length = min(len(load_curve_array), len(grid_limit))
                        load_curve_array = load_curve_array[:min_length]
                        grid_limit = grid_limit[:min_length]
                    
                    grid_violations = np.sum((load_curve_array > grid_limit).astype(int))
                    max_violation = np.max(load_curve_array - grid_limit) if np.any(load_curve_array > grid_limit) else 0
                    performance_metrics.update({
                        'Grid Compliance %': f"{((len(load_curve_array) - grid_violations) / len(load_curve_array) * 100):.1f}%",
                        'Maximum Grid Violation': f"{max_violation:.2f} kW",
                    })
                
                # Merge all statistics
                additional_stats.update(performance_metrics)
                
                # Merge statistics, prioritizing simulation_results
                all_stats = {**additional_stats, **stats}
            else:
                all_stats = stats
            
            # Create table data
            table_data = [['Metric', 'Value']]
            for key, value in all_stats.items():
                if isinstance(value, (int, float)):
                    if 'Ratio' in key or 'Compliance' in key or 'Standard Deviation' in key or 'Variance' in key:
                        formatted_value = f"{value:.2f}"
                    elif 'Maximum' in key or 'Average' in key or 'Minimum' in key or 'Range' in key or 'Violation' in key:
                        formatted_value = f"{value:,.2f} kW"
                    else:
                        formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                else:
                    formatted_value = str(value)
                table_data.append([key, formatted_value])
            
            # Create table with beautiful styling
            table = Table(table_data, colWidths=[4*inch, 3*inch])
            table.setStyle(TableStyle(get_table_style()))
            
            story.append(table)
            story.append(Spacer(1, 15))
        
        # Enhanced TOU Configuration section
        if include_options.get('tou_config', False):
            story.append(Paragraph("Time of Use Configuration", subtitle_style))
            
            # Try multiple sources for TOU data with better extraction
            tou_data = None
            tou_source = "None"
            
            # First try session state
            if 'time_of_use_timeline' in st.session_state:
                tou_data = st.session_state.time_of_use_timeline
                tou_source = "session_state"
            
            # If not found, try simulation results
            if not tou_data and 'tou_timeline' in simulation_results:
                tou_data = simulation_results['tou_timeline']
                tou_source = "simulation_results"
            
            # Also try direct access to session state keys
            if not tou_data and hasattr(st.session_state, 'time_of_use_timeline'):
                tou_data = st.session_state.time_of_use_timeline
                tou_source = "direct_session_state"
            
            # Debug: Check if TOU is enabled
            tou_enabled = False
            if 'smart_charging_enabled' in st.session_state:
                tou_enabled = st.session_state.smart_charging_enabled
            elif 'active_strategies' in st.session_state:
                tou_enabled = 'smart_charging' in st.session_state.active_strategies
            
            if tou_data and 'periods' in tou_data and len(tou_data['periods']) > 0:
                # Create table data with compact hour format
                table_data = [['Period', 'Hours', 'Adoption %']]
                for i, period in enumerate(tou_data['periods']):
                    # Handle the actual data structure: 'hours' list and 'adoption' percentage
                    hours_list = period.get('hours', [])
                    adoption = period.get('adoption', 0)
                    
                    # Convert hours list to display format
                    if hours_list:
                        # Sort hours and find continuous ranges
                        sorted_hours = sorted(hours_list)
                        ranges = []
                        start = sorted_hours[0]
                        end = start
                        
                        for hour in sorted_hours[1:]:
                            if hour == end + 1:
                                end = hour
                            else:
                                # End of current range
                                if start == end:
                                    ranges.append(f"{start}")
                                else:
                                    ranges.append(f"{start}-{end}")
                                start = hour
                                end = hour
                        
                        # Add the last range
                        if start == end:
                            ranges.append(f"{start}")
                        else:
                            ranges.append(f"{start}-{end}")
                        
                        hours_display = ", ".join(ranges)
                    else:
                        hours_display = "N/A"
                    
                    table_data.append([
                        period.get('name', f"Period {i+1}"),
                        hours_display,
                        f"{adoption:.1f}%"
                    ])
                
                # Create table with beautiful styling (adjusted column widths)
                table = Table(table_data, colWidths=[2*inch, 2*inch, 2*inch])
                table.setStyle(TableStyle(get_table_style(center_align=True)))
                
                story.append(table)
                
                #
            else:
                # Provide more detailed information about why TOU data is not available
                if not tou_enabled:
                    story.append(Paragraph("Time of Use is not enabled in this simulation", normal_style))
                elif not tou_data:
                    story.append(Paragraph("No TOU configuration data found in session state or simulation results", normal_style))
                else:
                    story.append(Paragraph("TOU configuration is empty or invalid", normal_style))
                
                
            
            story.append(Spacer(1, 15))
        
        # EV Calculator Results section
        if include_options.get('ev_calculator', False) and 'ev_calculator_results' in st.session_state:
            story.append(Paragraph("🚗 EV Calculator Results", subtitle_style))
            
            ev_results = st.session_state.ev_calculator_results
            if ev_results:
                # Create table data
                table_data = [['Metric', 'Value']]
                for key, value in ev_results.items():
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:,.0f}" if isinstance(value, int) else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    table_data.append([key.replace('_', ' ').title(), formatted_value])
                
                # Create table with beautiful styling
                table = Table(table_data, colWidths=[4*inch, 3*inch])
                table.setStyle(TableStyle(get_table_style()))
                
                story.append(table)
            else:
                story.append(Paragraph("No EV calculator results available", normal_style))
        
        # Set PDF metadata using canvas
        def add_metadata(canvas, doc):
            canvas.setTitle("EV Load Simulation Report")
            canvas.setAuthor("EV Simulation Tool")
            canvas.setSubject("Electric Vehicle Load Simulation Analysis")
            canvas.setCreator("EV Simulation Tool - PDF Generator")
        
        # Build the PDF with metadata and page background
        if add_page_background:
            doc.build(story, onFirstPage=lambda canvas, doc: [add_page_background(canvas, doc), add_metadata(canvas, doc)], 
                     onLaterPages=lambda canvas, doc: [add_page_background(canvas, doc), add_metadata(canvas, doc)])
        else:
            doc.build(story, onFirstPage=add_metadata, onLaterPages=add_metadata)
        
        # Get the PDF data
        pdf_buffer.seek(0)
        pdf_data = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        return True, "", pdf_data
        
    except Exception as e:
        return False, f"Error generating PDF with ReportLab: {str(e)}", None


def render_pdf_save_ui():
    """
    Render the PDF save UI with content selection options.
    """
    st.markdown("### 📄 Save Results")
    
    
    # Single radio button with all 6 themes in horizontal layout
    theme = st.radio(
        "Select theme:",
        ["Light Blue", "Dark Blue", "Light Green", "Light Brown", "Dark Orange", "Dark Green"],
        index=0,
        horizontal=True,
        help="Choose from light and dark themes"
    )
    
    # Convert theme name to function parameter
    theme_map = {
        "Light Blue": "light-blue",
        "Dark Blue": "dark-blue", 
        "Light Green": "light-green",
        "Light Brown": "light-brown",
        "Dark Orange": "dark-orange",
        "Dark Green": "dark-green"
    }
    theme = theme_map[theme]
    
    # Content selection
    st.markdown("**Select content to include in the report:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_parameters = st.checkbox("⚙️ Simulation Parameters", value=True)
        include_graphs = st.checkbox("📊 Load Curves", value=True)
        include_summary = st.checkbox("📈 Performance Metrics", value=True)
    
    with col2:
        include_optimization = st.checkbox("🔧 Optimization Summary", value=True)
        include_tou_config = st.checkbox("⏰ Time of Use Configuration", value=True)
    
    # Filename field
    default_filename = f"simulation_report_{datetime.datetime.now().strftime('%d_%m_%y')}.pdf"
    filename = st.text_input(
        "📄 Filename (optional):",
        value=default_filename,
        placeholder="Enter filename for the PDF report..."
    )
    
    # Use default filename if user didn't type anything
    if not filename or filename.strip() == "":
        filename = default_filename
    
    # Description field (always available, no checkbox needed)
    simulation_description = st.text_area(
        "📝 Simulation Description (optional):",
        placeholder="Enter a description of this simulation...",
        height=100
    )
    
    # Create include options dictionary
    include_options = {
        'parameters': include_parameters,
        'graphs': include_graphs,
        'summary': include_summary,
        'optimization': include_optimization,
        'tou_config': include_tou_config,
    }
    
    # Check if at least one option is selected
    if not any(include_options.values()):
        st.warning("Please select at least one content option to include in the report.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("Generate PDF Report", type="primary"):
            if 'simulation_results' in st.session_state and st.session_state.simulation_results is not None:
                # Generate PDF
                success, error_message, pdf_data = create_pdf_report(
                    st.session_state.simulation_results,
                    simulation_description,
                    include_options,
                    theme
                )
                
                if success:
                    # Create download link with custom filename
                    pdf_b64 = base64.b64encode(pdf_data).decode()
                    # Ensure filename has .pdf extension
                    if not filename.lower().endswith('.pdf'):
                        filename = filename + '.pdf'
                    html_download = f"""
                    <a href="data:application/pdf;base64,{pdf_b64}" 
                       download="{filename}" 
                       style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; margin: 10px 0;">
                        📥 Download PDF Report
                    </a>
                    """
                    
                    st.markdown(html_download, unsafe_allow_html=True)
                    st.success("✅ PDF report generated successfully!")
                else:
                    st.error(f"❌ {error_message}")
            else:
                st.warning("⚠️ No simulation results available. Please run a simulation first.")
    
    with col2:
        st.markdown("""
        **💡 Tips:**
        - Select the content you want to include in your report
        - Add a description to provide context for your simulation
        - The report will be generated as a beautiful A4 PDF document
        - All graphs and data will be preserved in high quality
        - Uses ReportLab with built-in fonts (no web dependencies)
        """) 