"""HuggingFace Spaces entrypoint — delegates to gradio_app."""
import os
os.makedirs("./output", exist_ok=True)

from gradio_app import app

app.launch(server_name="0.0.0.0", server_port=7860)
