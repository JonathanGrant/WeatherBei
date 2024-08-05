import io
from WeatherGPT import WeatherDraw
import PIL.Image


def get_weather_image(zip_code='60654'):
    image = WeatherDraw().step(zip_code)[0]
    return image

def run(zip_code='60654'):
    img = get_weather_image(zip_code=zip_code)
    img.save('latest_raw.jpg')

run()

