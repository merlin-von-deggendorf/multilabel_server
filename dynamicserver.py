import os
from flask import Flask, render_template, jsonify, request
import io
from PIL import Image
import argparse
import json

app = Flask(__name__)




# replace events with carts
carts = []

@app.route('/api/events', methods=['POST'])
def receive_event():
    try:
        payload = request.data.decode('utf-8')
        cart = json.loads(payload)
        print(cart)
        # optionally validate cart is a list or dict here
        carts.append(cart)     # store the whole shopping cart
        return '', 204
    except Exception as e:
        return jsonify({'message': f'Error parsing event data: {e}'}), 400

@app.route('/')
def show_carts():
    return render_template('index.html', carts=carts)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Flask server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Hostname to listen on')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    app.run(debug=args.debug, host=args.host, port=args.port)