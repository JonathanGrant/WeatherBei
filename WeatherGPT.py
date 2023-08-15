# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: '3.11'
#     language: python
#     name: '3.11'
# ---

# +
import requests
import structlog
import openai
import random
import enum
import time
import retrying
import IPython.display as display
from base64 import b64decode
import base64
from io import BytesIO
import PIL
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
        url = f"https://forecast.weather.gov/MapClick.php?lat={lat:.2f}&lon={long:.2f}&unit=0&lg=english&FcstType=json"
        headers = {'accept': 'application/json'}
        return requests.get(url, headers=headers).json()

    def get_info(self):
        data = self.get_weather()
        new_data = {}
        new_data['now'] = data['currentobservation']
    
        # The 'time' and 'data' keys seem to have hourly/daily data
        # Assuming the first entry in these lists is for the current hour
        new_data['hour'] = {
            'time': data['time']['startValidTime'][0],
            'tempLabel': data['time']['tempLabel'][0],
            'temperature': data['data']['temperature'][0],
            'pop': data['data']['pop'][0],
            'weather': data['data']['weather'][0],
            'iconLink': data['data']['iconLink'][0],
            'text': data['data']['text'][0],
        }
    
        # And the rest of the 'time' and 'data' lists are for the rest of the day
        new_data['day'] = {
            'time': data['time']['startValidTime'][1:],
            'tempLabel': data['time']['tempLabel'][1:],
            'temperature': data['data']['temperature'][1:],
            'pop': data['data']['pop'][1:],
            'weather': data['data']['weather'][1:],
            'iconLink': data['data']['iconLink'][1:],
            'text': data['data']['text'][1:],
        }
    
        # Everything not included above
        new_data['etc'] = str(data)
    
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


class WeatherDraw:
    def clean_text(self, weather_info):
        chat = Chat("Given the following weather conditions, write a very small, concise plaintext summary that will overlay on top of an image.")
        text = chat.message(str(weather_info))
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
        description = chat.message(str(weather_info))
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
        # text = self.clean_text(weather_info)
        # return overlay_text_on_image(img, text, 'bottom-left')
        return img, txt

    def weather_img(self, weather_data):
        import io
        # Create a new image with white background
        image = PIL.Image.new('RGB', (256, 256), (255, 255, 255))
        draw = PIL.ImageDraw.Draw(image)
    
        # Load a font
        font = PIL.ImageFont.truetype(other_font_path, 12)

        # Draw text on the image
        y_text = 5
        items_to_display = {
            'now': {'Temperature': weather_data['now']['Temp'], 
                    'Condition': weather_data['now']['Weather'],}, 
            'hour': {'Temperature': weather_data['hour']['temperature'], 
                    'Condition': weather_data['hour']['weather']},
            'day': {'High': int(max(float(t) for t in weather_data['day']['temperature'])), 
                    'Low': int(min(float(t) for t in weather_data['day']['temperature'])), 
                    'Condition': weather_data['day']['weather'][0]}, 
        }
    
        for category, values in items_to_display.items():
            draw.text((5, y_text), category, font=font, fill=(0, 0, 0))
            y_text += 15
            for key, value in values.items():
                text = f"{key}: {value}"
                draw.text((10, y_text), text, font=font, fill=(0, 0, 0))
                y_text += 15
    
        # Download the weather condition icon for now, day and next hour
        for index, time in enumerate(items_to_display.keys()):
            if time == 'day':
                icon_url = weather_data['day']['iconLink'][0]
            elif time == 'now':
                icon_url = 'https://forecast.weather.gov/newimages/medium/'+weather_data['now']['Weatherimage']
            else:
                icon_url = weather_data[time]['iconLink']
            print(time, icon_url)
            try:
                response = requests.get(icon_url)
                icon = PIL.Image.open(io.BytesIO(response.content))
            except:
                continue
            # Resize the icon
            icon = icon.resize((60, 60))
            # Paste the icon on the image
            image.paste(icon, (index*70 + 10, 190))

        return image

    def step(self, zip_code='10001', **kwargs):
        forecast = Weather(zip_code).get_info()
        images, texts = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
            runs = {}
            for time, data in forecast.items():
                if time == 'etc': continue
                runs[e.submit(self.step_one_forecast, data, **kwargs)] = time, data
            for r in concurrent.futures.as_completed(runs.keys()):
                img, txt = r.result()
                time, data = runs[r]
                images.append(overlay_text_on_image(img, time, 'top-right', decode=True))
                # images.append(overlay_text_on_image(img, '', 'top-right', decode=True))
                texts.append(txt)
        return create_collage(*images, self.weather_img(forecast)), *texts


# +
# out = WeatherDraw().step('94024')
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

get_7color = lambda img: PIL.Image.fromarray(np.array(colors[np.argmin(((np.array(img, dtype=float).reshape(-1, 3)[:, np.newaxis, :] - colors)**2).sum(axis=2), axis=1)].reshape(512, 512, 3), dtype=np.uint8))


# +
# # %%time
# fsd = floyd_steinberg_dithering(out[0], colors, error_weights)
# disp_concat(fsd, get_7color(fsd))

# +
# a = get_7color(fsd)
# -

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
    resized_image = img.resize((new_width, new_height), PIL.Image.ANTIALIAS)
    # Create a black background
    background = PIL.Image.new("RGB", (target_width, target_height), "white")
    # Calculate the position to paste the resized image onto the background
    position = ((target_width - new_width) // 2, (target_height - new_height) // 2)
    # Paste the resized image onto the background
    background.paste(resized_image, position)
    return background

# +
# resize_img(a)

# +
# # %%time
# resize_img(get_7color(floyd_steinberg_dithering(WeatherDraw().step('94024')[0], colors, error_weights)))
# -


