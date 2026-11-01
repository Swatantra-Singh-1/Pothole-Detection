
import socket
from datetime import datetime

HOST = "0.0.0.0"
PORT = 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)

    print(f"GPS receiver listening on {HOST}:{PORT}")

    while True:
        conn, addr = server.accept()
        print(f"\n[{datetime.now()}] Connected from {addr}")

        with conn:
            conn.settimeout(30)

            try:
                while True:
                    data = conn.recv(4096)

                    if not data:
                        print("Client disconnected without more data.")
                        break

                    print("Received:", repr(data))

            except socket.timeout:
                print("No data received for 30 seconds.")

            except OSError as e:
                print("Socket error:", e)

        print("Waiting for the next connection...")
