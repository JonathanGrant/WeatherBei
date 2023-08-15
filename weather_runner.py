import io
from WeatherGPT import resize_img, get_7color, floyd_steinberg_dithering, colors, error_weights, WeatherDraw
import PIL.Image


def convert_image(image):
    dithered = floyd_steinberg_dithering(image, colors, error_weights) 
    seven_color = get_7color(dithered)
    resized = resize_img(seven_color)
    return resized

def get_weather_image(zip_code='10001'):
    image = WeatherDraw().step(zip_code)[0]
    return image

def run(zip_code='94024'):
    img = get_weather_image(zip_code=zip_code)
    img = convert_image(img)
    img.save('latest.jpg')

run()

