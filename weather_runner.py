import io
from WeatherGPT import resize_img, get_7color, floyd_steinberg_dithering, colors, error_weights, WeatherDraw
import PIL.Image


def convert_image(image):
    dithered = floyd_steinberg_dithering(image, colors, error_weights)
    dithered.save('latest_dithered.jpg')
    return get_7color(dithered)

def get_weather_image(zip_code='10001'):
    image = WeatherDraw().step(zip_code)[0]
    return image

def run(zip_code='10001'):
    img = get_weather_image(zip_code=zip_code)
    img.save('latest_raw.jpg')
    img = convert_image(img)
    img.save('latest.jpg')

run()

