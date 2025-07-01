from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=['GET'])
def none():
    print("S")
    return jsonify({'error': 'Missing file or form data (layout, frames)'}), 200

if __name__ == '__main__':
    # 按要求将端口设置为 8000
    app.run(host='0.0.0.0', port=8000, debug=True)