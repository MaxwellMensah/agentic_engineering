import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen

PORT = 8080
A2A_ENDPOINT = f"http://localhost:{PORT}/a2a/message"


# AGENT B: Receiver Endpoint (HTTP Server)
class AgentBHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # Suppress default server noise

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        incoming_message = json.loads(post_data.decode("utf-8"))

        print("\n[Agent B Received HTTP Request]")
        print(f" Sender: {incoming_message.get('sender_id')}")
        print(f" Intent: {incoming_message.get('intent')}")
        print(f" Payload: {incoming_message.get('body')}")

        # Construct A2A Response Payload
        a2a_response = {
            "status": "success",
            "receiver_id": "agent-b-v1",
            "response": f"Processed intent '{incoming_message.get('intent')}' successfully.",
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(a2a_response).encode("utf-8"))


def run_agent_b_server():
    server = HTTPServer(("localhost", PORT), AgentBHTTPHandler)
    server.serve_forever()


# AGENT A: Sender Client (HTTP Request)
def agent_a_send_message():
    time.sleep(1)  # Ensure server is running

    a2a_payload = {
        "protocol_version": "1.0",
        "sender_id": "agent-a-v1",
        "intent": "TASK_REQUEST",
        "body": {
            "task": "Verify cryptographic hash",
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
    }

    print(f"[Agent A Sending HTTP Request] -> {A2A_ENDPOINT}")
    req = Request(
        A2A_ENDPOINT,
        data=json.dumps(a2a_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urlopen(req) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        print("\n[Agent A Received Response]")
        print(f" Data: {res_data}")


if __name__ == "__main__":
    # Start Agent B in background thread
    server_thread = threading.Thread(target=run_agent_b_server, daemon=True)
    server_thread.start()

    # Agent A executes call
    agent_a_send_message()
