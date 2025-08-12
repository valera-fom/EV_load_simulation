# Saved Sessions Directory

This directory contains saved session states for the EV Simulation application.

> **Note**: The `.gitkeep` file ensures this directory is tracked in git even when empty.

## What are Session Files?

Session files (`.json`) contain the complete configuration state of your simulation, including:

- EV Configuration (battery capacity, charging power, etc.)
- Charger Configuration (rates, counts)
- Time Control settings (duration, peaks)
- Optimization Strategies (TOU, PV+Battery, V2G, etc.)
- Data Source settings
- Grid Settings
- Random Seed configuration
- Simulation Results (if available)

## File Naming Convention

Files are automatically named with timestamps:
- `session_YYYY_MM_DD_HHMMSS.json` (auto-generated)
- `Custom_Name_YYYY_MM_DD_HHMMSS.json` (user-named)

## Usage

1. **Save Session**: Use the "🔄 Session Manager" in the sidebar
2. **Load Session**: Select a saved session to restore your configuration
3. **Manage Sessions**: Delete, export, or backup your sessions

## Backup

You can export all sessions as a ZIP file for backup purposes.

## Security Note

Session files contain your simulation configuration. Keep them secure if they contain sensitive data. 