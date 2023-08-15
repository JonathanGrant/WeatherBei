from flask import Flask, request, send_file
import io
from WeatherGPT import resize_img, get_7color, floyd_steinberg_dithering, colors, error_weights, WeatherDraw
import PIL.Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 160 * 1024 * 1024 # 160MB

@app.route('/convert_image', methods=['POST'])
def convert_image():
    image_bytes = request.files['image'].read()
    image = PIL.Image.open(io.BytesIO(image_bytes))
    
    dithered = floyd_steinberg_dithering(image, colors, error_weights) 
    seven_color = get_7color(dithered)
    resized = resize_img(seven_color)
    
    img_io = io.BytesIO()
    resized.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

@app.route('/get_weather_image')  
def get_weather_image():
    zip_code = request.args.get('zip_code')
    image = WeatherDraw().step(zip_code)[0]
    
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run()
