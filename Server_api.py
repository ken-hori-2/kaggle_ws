from flask import Flask, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # セッションの秘密鍵を設定

# APIキーをハードコーディング
API_KEY = 'dp-testapikeyken' # 'your_api_key'
# API_KEY = "dp-testapikeyken"

# @app.route('/log_action', methods=['POST'])
# def log_action():

# @app.route('/api/items', methods=['GET']) # GETを使う場合
@app.route('/api/items', methods=['POST']) # POSTを使う場合
def get_items():
    # api_key = request.headers.get('Authorization') # ???
    api_key = request.args.get('Authorization') # こっちはできる
    
    print("Server_api api_key=", api_key)
    print("Server_api API_KEY", API_KEY)
    
    if api_key != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

    category = request.args.get('category')
    
    # 認証成功した場合、以降の処理を実行
    # ... (既存のコード)
    # ここで、受け取ったパラメータを使って処理を行う
    # 例: データベースからデータを取得する
    result = f"Received category: {category}, api_key: {api_key}"

    return result


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='127.0.0.1', port=5000)