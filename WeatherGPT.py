# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import requests
import structlog
import openai
import random
import zoneinfo
import enum
import time
import retrying
import IPython.display as display
from base64 import b64decode
import base64
from io import BytesIO
import os
import textwrap
import PIL
import datetime
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from ChatPodcastGPT import Chat
import concurrent.futures
from huggingface_hub import hf_hub_download
import geopy
import numpy as np

logger = structlog.getLogger()
animals = [x.strip() for x in open('animals.txt').readlines()]
art_styles = [x.strip() for x in open('art_styles.txt').readlines()]
font_path = hf_hub_download("jonathang/fonts-ttf", "Vogue.ttf")
other_font_path = hf_hub_download("ybelkada/fonts", "Arial.TTF")
mono_font_path = hf_hub_download("jonathang/fonts-ttf", "DejaVuSansMono.ttf")

# +
import cachetools

@cachetools.cached(cache={})
def get_lat_long(zip):
    try:
        loc = geopy.Nominatim(user_agent='weatherboy-gpt').geocode(str(zip))
        return loc.latitude, loc.longitude
    except:
        return get_lat_long_gmaps(zip)

@cachetools.cached(cache={})
def get_lat_long_gmaps(zip):
    api_key = os.environ["GMAPS_API"] or open('/Users/jong/.gmaps_key').read()
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zip}&key={api_key}"
    resp = requests.get(url).json()
    latlng = resp['results'][0]['geometry']['location']
    return latlng['lat'], latlng['lng']


# -

class Weather:
    def __init__(self, zip_code='10001'):
        self.zip_code = zip_code

    def get_weather(self):
        lat, long = get_lat_long(self.zip_code)
        today = datetime.datetime.now().astimezone(zoneinfo.ZoneInfo("US/Eastern"))
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat:.3f}&longitude={long:.3f}&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation_probability,precipitation,rain,showers,snowfall,snow_depth,weathercode,pressure_msl,surface_pressure,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,visibility,evapotranspiration,windspeed_10m,winddirection_10m,windgusts_10m,uv_index,freezinglevel_height,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,terrestrial_radiation&daily=weathercode,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,uv_index_max,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours,precipitation_probability_max,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant,shortwave_radiation_sum&temperature_unit=fahrenheit&windspeed_unit=mph&precipitation_unit=inch&timezone=auto&forecast_days=1&forecast_days=1&start_date={today:%Y-%m-%d}&end_date={today:%Y-%m-%d}&models=best_match"
        # url = f"https://forecast.weather.gov/MapClick.php?lat={lat:.2f}&lon={long:.2f}&unit=0&lg=english&FcstType=json"
        headers = {'accept': 'application/json'}
        return requests.get(url, headers=headers).json()

    def get_info(self):
        data = self.get_weather()
        new_data = {}
        now = datetime.datetime.now().astimezone(zoneinfo.ZoneInfo("US/Eastern"))
        i = data['hourly']['time'].index(now.strftime("%Y-%m-%dT%H:00"))
        new_data['now'] = {k: v[i] for k, v in data['hourly'].items()}
        new_data['day'] = data['daily']
        new_data['morning'] = {k: v[7:13] for k, v in data['hourly'].items()}
        new_data['afternoon'] = {k: v[13:19] for k, v in data['hourly'].items()}
        return new_data


class Image:
    class Size(enum.Enum):
        SMALL = "256x256"
        MEDIUM = "512x512"
        LARGE = "1024x1024"

    @classmethod
    @retrying.retry(stop_max_attempt_number=5, wait_fixed=2000)
    def create(cls, prompt, n=1, size=Size.SMALL):
        logger.info('requesting openai.Image...')
        resp = openai.Image.create(prompt=prompt, n=n, size=size.value, response_format='b64_json')
        logger.info('received openai.Image...')
        if n == 1: return resp["data"][0]
        return resp["data"]


def overlay_text_on_image(img, text, position, text_color=(255, 255, 255), box_color=(0, 0, 0), decode=False):
    # Convert the base64 string back to an image
    if decode:
        img_bytes = base64.b64decode(img)
        img = PIL.Image.open(BytesIO(img_bytes))

    # Get image dimensions
    img_width, img_height = img.size

    # Create a ImageDraw object
    draw = PIL.ImageDraw.Draw(img)
    
    # Reduce the font size until it fits the image width or height
    l, r = 1, 50
    while l < r:
        font_size = (l + r) // 2
        font = PIL.ImageFont.truetype(font_path, font_size)
        left, upper, right, lower = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = lower - upper
        if text_width <= img_width and text_height <= img_height:
            l = font_size + 1
        else:
            r = font_size - 1
    font_size = max(l-1, 1)

    left, upper, right, lower = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = lower - upper

    if position == 'top-left':
        x, y = 0, 0
    elif position == 'top-right':
        x, y = img_width - text_width, 0
    elif position == 'bottom-left':
        x, y = 0, img_height - text_height
    elif position == 'bottom-right':
        x, y = img_width - text_width, img_height - text_height
    else:
        raise ValueError("Position should be 'top-left', 'top-right', 'bottom-left' or 'bottom-right'.")

    # Draw a semi-transparent box around the text
    draw.rectangle([x, y, x + text_width, y + text_height], fill=box_color)

    # Draw the text on the image
    draw.text((x, y), text, font=font, fill=text_color)

    return img


def create_collage(image1, image2, image3, image4):
    # assuming images are the same size
    width, height = image1.size

    new_img = PIL.Image.new('RGB', (2 * width, 2 * height))

    # place images in collage image
    new_img.paste(image1, (0,0))
    new_img.paste(image2, (width, 0))
    new_img.paste(image3, (0, height))
    new_img.paste(image4, (width, height))

    return new_img


def write_text_on_image(text, image_size, font_size=24):
    image = PIL.Image.new('RGB', image_size, color='white')
    draw = PIL.ImageDraw.Draw(image)
    font = PIL.ImageFont.truetype(mono_font_path, font_size)

    margin = offset = 0
    for lines in '\n'.join(text.split('. ')).split('\n'):
        for line in textwrap.wrap(lines, width=image_size[0]//(font_size*7//12)): # You can adjust the width parameter
            draw.text((margin, offset), line, font=font, fill="black")
            offset += font_size

    return image


def resize_img(img):
    # Define target size
    target_width, target_height = 800, 480
    # Calculate the aspect ratio
    aspect_ratio = img.width / img.height
    # Determine new size while keeping aspect ratio
    if aspect_ratio > target_width / target_height:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_image = img.resize((new_width, new_height), PIL.Image.LANCZOS)
    # Create a black background
    background = PIL.Image.new("RGB", (target_width, target_height), "white")
    # Calculate the position to paste the resized image onto the background
    position = ((target_width - new_width) // 2, (target_height - new_height) // 2)
    # Paste the resized image onto the background
    background.paste(resized_image, position)
    return background


class WeatherDraw:
    def clean_text(self, weather_info):
        chat = Chat("Given the following weather conditions, write a very small, concise plaintext summary. Just include the weather, no dates.")
        text = chat.message(str(weather_info)[:4000])
        return text

    def generate_image(self, weather_info, **kwargs):
        animal = random.choice(animals)
        logger.info(f"Got animal {animal}")
        chat = Chat(f'''Given
the following weather conditions, write a plaintext, short, and vivid description of an
image of an adorable anthropomorphised {animal} doing an activity in the weather.
The image should make obvious what the weather is.
The animal should be extremely anthropomorphised.
Only write the short description and nothing else.
Do not include specific numbers.'''.replace('\n', ' '))
        description = chat.message(str(weather_info)[:4000])
        hd_modifiers = """3840x2160
8k 3D / 16k 3D
8k resolution / 16k resolution
Detailed
Ultra HD
Ultrafine detail
""".split('\n')
        prompt = f'{random.choice(art_styles)} of {description} {random.choice(hd_modifiers)}'
        logger.info(prompt)
        img = Image.create(prompt, **kwargs)
        return img["b64_json"], prompt

    def step_one_forecast(self, weather_info, **kwargs):
        img, txt = self.generate_image(weather_info, **kwargs)
        return img, txt

    def weather_img(self, weather_data):
        return write_text_on_image(
            self.clean_text(weather_data['day']),
            ((800-480)//2, 480),
        )

    def left_text_img(self, weather_data):
        now = datetime.datetime.now().astimezone(zoneinfo.ZoneInfo("US/Eastern"))
        animal = random.choice(animals)
        chat = Chat(f'Give me a concise rare fun fact about the cute animal, {animal}.')
        return write_text_on_image(
            f'{now:%Y-%m-%d}\n{now:%H:%M}\n\n{chat.message()}',
            ((800-480)//2, 480),
        )

    def step(self, zip_code='10001', **kwargs):
        forecast = Weather(zip_code).get_info()
        images, texts = [None] * 4, [None] * 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
            runs = {}
            for time, data in forecast.items():
                if time == 'etc': continue
                runs[e.submit(self.step_one_forecast, data, **kwargs)] = time, data
            for r in concurrent.futures.as_completed(runs.keys()):
                img, txt = r.result()
                time, data = runs[r]
                ridx = list(forecast.keys()).index(time)
                images[ridx] = overlay_text_on_image(img, time, 'top-right', decode=True)
                texts[ridx] = txt
        # return create_collage(*images, self.weather_img(forecast)), *texts)
        img = resize_img(create_collage(*images))
        img.paste(self.weather_img(forecast), (480+160, 0))
        img.paste(self.left_text_img(forecast), (0, 0))

        return img, *texts


# +
# out = WeatherDraw().step()
# out[0]

# +
from numba import jit
@jit
def njit_find_nearest_palette_color(old_pixel, palette):
    distances = np.sum((palette - old_pixel) ** 2, axis=1)
    nearest_color_idx = np.argmin(distances)
    return palette[nearest_color_idx]

def apply_error_diffusion(image, palette, error_weights):
    pixels = np.array(image).astype('float32')
    width, height, _ = pixels.shape

    for c in range(3):
        for y in range(height):
            for x in range(width):
                old_pixel = pixels[x, y, c]
                new_pixel = njit_find_nearest_palette_color(old_pixel, palette)
                pixels[x, y, c] = new_pixel[c]  # Convert RGB color to integer for each channel

                quant_error = old_pixel - new_pixel[c]  # Use the new_pixel value of the current channel

                # Error propagation for each channel
                for dy, dx, weight in error_weights:
                    if 0 <= x + dx < width and 0 <= y + dy < height:
                        pixels[x + dx, y + dy, c] += quant_error * weight

    return PIL.Image.fromarray(pixels.astype(np.uint8))


def floyd_steinberg_dithering(image, palette, error_weights):
    dithered_image = apply_error_diffusion(image, palette, error_weights)
    return dithered_image

error_weights = [
    (0, 1, 7/16),
    (1, -1, 1/16),
    (1, 0, 5/16),
    (1, 1, 3/16)
]
colors = np.array([
    (0, 0, 0),  # Black
    (0, 1, 0),  # Green
    (0, 1, 1),  # Blue
    (1, 0, 0),  # Red
    (1, 0, 1),  # Yellow
    (1, 1, 0),  # Orange
    (1, 1, 1),  # White
]) * 255

def disp_concat(*images):
    concatenated_image = PIL.Image.new('RGB', (images[0].width * len(images), images[0].height))
    for i, img in enumerate(images):
        concatenated_image.paste(img, (images[0].width*i, 0))
    
    # Display the concatenated image
    display.display(concatenated_image)

get_7color = lambda img: PIL.Image.fromarray(np.array(colors[np.argmin(((np.array(img, dtype=float).reshape(-1, 3)[:, np.newaxis, :] - colors)**2).sum(axis=2), axis=1)].reshape(img.height, img.width, 3), dtype=np.uint8))

# +
# # %%time
# fsd = floyd_steinberg_dithering(out[0], colors, error_weights)
# disp_concat(fsd, get_7color(out[0]))

# +
# get_7color(fsd)

# +
# resize_img(a)
# get_7color(fsd)

# +
# # %%time
# resize_img(get_7color(floyd_steinberg_dithering(WeatherDraw().step('94024')[0], colors, error_weights)))
# -


