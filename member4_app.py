# member4_app.py
"""
Launches the unified JobShield AI Webpage (`index.html`) and attaches the SQLite
User History Database (`jobshield_history.db`).
Running either `python app_server.py` or `python member4_app.py` opens the exact
same webpage as clicking "Go Live" in VS Code.
"""

from app_server import run_server

if __name__ == "__main__":
    run_server(port=8000, open_browser=True)
