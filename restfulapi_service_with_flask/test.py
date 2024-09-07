from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import time

app = Flask(__name__)

# Initialize ThreadPoolExecutor with a specific number of worker threads
executor = ThreadPoolExecutor(max_workers=4)

def long_running_task(duration):
    """A dummy long-running task"""
    time.sleep(duration)
    return f"Task completed in {duration} seconds"

@app.route('/start-task', methods=['POST'])
def start_task():
    duration = request.json.get('duration', 5)  # Get duration from POST data, default to 5 seconds
    # Submit the long-running task to the thread pool
    future = executor.submit(long_running_task, duration)
    return jsonify({"message": "Task started", "duration": duration}), 202

@app.route('/check-status', methods=['GET'])
def check_status():
    # Dummy endpoint to show the app is still responsive
    return jsonify({"status": "App is running"}), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True)