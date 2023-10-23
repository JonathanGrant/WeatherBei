from flask import Flask, request, send_file
import io
import tempfile
from WeatherGPT import resize_img, WeatherDraw
import PIL.Image
import cachetools

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 160 * 1024 * 1024 # 160MB
# TODO: Set up caches for weather images
weather_img_cache = cachetools.TTLCache(42069, 60*60*4)  # 4 hours TTL in seconds


@app.route('/get_weather_image')  
def get_weather_image():
    # Get latitude and longitude from request
    lati = float(f"{float(request.args.get('lati')):.2f}")
    long = float(f"{float(request.args.get('long')):.2f}")
    try:
        image = weather_img_cache[(lati, long)]
    except:
        # Get weather forecast
        # Generate (1) text + image
        # Add text to image in portrait mode
        # Add to cache
        image = WeatherDraw().portrait_step(lati, long)[0]
        weather_img_cache[(lati, long)] = image
    # Return image
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run()
