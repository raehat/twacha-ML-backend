from flask import Flask, request, jsonify
from fastai.vision import *
import base64

app = Flask(__name__)

NAME_OF_FILE = 'model' 
PATH_TO_MODELS_DIR = Path('')
classes = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
           'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

def setup_model_pth(path_to_pth_file, learner_name_to_load, classes):
    data = ImageDataBunch.single_from_classes(
        path_to_pth_file, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data, models.densenet169, model_dir='models')
    learn.load(learner_name_to_load, device=torch.device('cpu'))
    return learn

learn = setup_model_pth(PATH_TO_MODELS_DIR, NAME_OF_FILE, classes)

def predict_image(image_data):
    # Decode base64-encoded image data
    img_data = base64.b64decode(image_data)
    
    # Assuming image_data is a bytearray
    img = open_image(BytesIO(img_data))
    pred_class, _, outputs = learn.predict(img)
    formatted_outputs = ["{:.1f}%".format(value * 100) for value in torch.nn.functional.softmax(outputs, dim=0)]
    pred_probs = sorted(zip(classes, map(str, formatted_outputs)), key=lambda p: p[1], reverse=True)

    return {"class": str(pred_class), "probs": pred_probs}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        image_data_base64 = request_data.get('image_data', '')
        
        # Ensure the received data is not empty
        if not image_data_base64:
            return jsonify({'error': 'No image data provided'})

        result = predict_image(image_data_base64)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/')
def hello():
    return 'Hello'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
