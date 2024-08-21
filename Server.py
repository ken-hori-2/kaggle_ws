from flask import Flask, request

app = Flask(__name__)

@app.route('/api/items', methods=['GET'])
def get_items():
    category = request.args.get('category')
    order = request.args.get('order')
    api_key = request.args.get('api_key_dp')

    # ここで、受け取ったパラメータを使って処理を行う
    # 例: データベースからデータを取得する
    result = f"Received category: {category}, order: {order}, api_key: {api_key}"

    return result

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='127.0.0.1', port=5000)

