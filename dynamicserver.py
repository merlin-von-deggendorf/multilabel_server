import os
from flask import Flask, render_template, jsonify, request
import resnet18
import io
from PIL import Image
import argparse

app = Flask(__name__)

model_name = 'deeplearning'
classes,trans=resnet18.ClassificationModel.load_classes(model_name)
grapes = resnet18.ClassificationModel(num_classes=len(classes))
grapes.load_model(model_name)



@app.route('/')
def wahl():
    # Serve the classifier.html template
    return render_template('glaumer.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the POST request contains the file part.
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in request.'})
    
    file = request.files['file']
    
    # Check if a file is selected.
    if file.filename == '':
        return jsonify({'message': 'No file selected.'})
    
    # Save the file in the uploads directory.
    try:
        # Read the file directly into memory and open as an image
        image_stream = io.BytesIO(file.read())
        image = Image.open(image_stream)
        
        # Optionally, if the model expects a different format, perform any necessary conversion here
        
        # Pass the image directly to the classifier.
        # You'll need to update classify_image in resnet18.ClassificationModel to work with a PIL Image.
        result_index,result_name,translation = grapes.classify_ram_image(image)

        
        return jsonify({'message': f'{translation}'})
    except Exception as e:
        return jsonify({'message': f'Error during classification: {str(e)}'})



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Flask server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Hostname to listen on')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    app.run(debug=args.debug, host=args.host, port=args.port)