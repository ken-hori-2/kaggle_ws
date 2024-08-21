from flask import Flask, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # セッションの秘密鍵を設定

# APIキーをハードコーディング
# 許可されたAPIキーのリスト (実際にはデータベースなどから取得する)
ALLOWED_API_KEYS = ['dp-testapikeyken', 'dp-testapikeyhori']



# @app.route('/log_action', methods=['POST'])
# def log_action():

@app.route('/api/items', methods=['GET']) # GETを使う場合 # @app.route('/api/items', methods=['POST']) # POSTを使う場合
def get_items():
    api_key = request.headers.get('Authorization') # head version.(headの方が機密性が高いので、API_KEYはこっちにする)
    # api_key = request.args.get('Authorization') # arg version.(actionなど検索ワードはparamsにする)
    
    if api_key in ALLOWED_API_KEYS:
        # APIキーが有効な場合
        category = request.args.get('category') # headerとparams version.(actionなど検索ワードはparamsにする)
        # category = request.headers.get('category')
    
        # 認証成功した場合、以降の処理を実行
        # ... (既存のコード)
        # ここで、受け取ったパラメータを使って処理を行う
        # 例: データベースからデータを取得する
        result = f"[Access granted !!] Received category: {category}, api_key: {api_key}"
        return result
        # return 'Access granted'
    else:
        # APIキーが無効な場合
        return 'Unauthorized', 401


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='127.0.0.1', port=5000)