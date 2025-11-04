import asyncio
import threading
import os
from flask import Flask
from app.scheduler import run


app = Flask(__name__)

@app.route("/")
def index():
    return "Service is running", 200


def start_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("[STOP]")
