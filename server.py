import socket
import json

HOST = '127.0.0.1'
PORT = 5001
LOG_FILE = "detection_log.json"

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f"[OK] Python TCP Server listening on {HOST}:{PORT}...")

    try:
        while True:
            print("C++クライアントからの接続を待機中...")
            conn, addr = server_socket.accept()
            print(f"[OK] Connected by {addr}")

            buffer = ""
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        print("C++クライアントが切断されました。")
                        break

                    buffer += data.decode('utf-8')

                    # 改行区切りでJSONをパースして保存
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            try:
                                payload = json.loads(line)
                                count = payload.get("count", 0)
                                inference_ms = payload.get("inference_ms", 0.0)
                                classes = payload.get("classes", [])

                                print(f"[Recv] 検出数: {count} | 推論時間: {inference_ms:.1f} ms | クラス: {classes}")

                                # ログファイルへ追記保存
                                with open(LOG_FILE, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

                            except json.JSONDecodeError:
                                print(f"[Error] パース失敗: {line}")

    except KeyboardInterrupt:
        print("\nサーバーを終了します。")
    finally:
        server_socket.close()

if __name__ == '__main__':
    start_server()
#python server.py