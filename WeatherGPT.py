# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
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
SD_APIKEY = os.environ.get("SD_APIKEY") or open(os.path.expanduser("~/.stability_apikey")).read().strip()
ARTISTS = open("artists.txt").read().splitlines()

# +
import cachetools

@cachetools.cached(cache={})
def get_lat_long(zip, country='USA'):
    try:
        loc = geopy.Nominatim(user_agent='weatherboy-gpt').geocode(str(zip)+ f', {country}')
        return loc.latitude, loc.longitude
    except:
        return get_lat_long_gmaps(zip)

@cachetools.cached(cache={})
def get_lat_long_gmaps(zip, country='USA'):
    api_key = os.environ.get("GMAPS_API", None) or open('/Users/jong/.gmaps_key').read()
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zip}, {country}&key={api_key}"
    resp = requests.get(url).json()
    latlng = resp['results'][0]['geometry']['location']
    return latlng['lat'], latlng['lng']


# -

class Weather:
    def __init__(self, zip_code='10001'):
        self.zip_code = zip_code

    def get_weather(self, lat_long=None):
        if lat_long is None:
            lat, long = get_lat_long(self.zip_code)
        else:
            lat, long = lat_long
        url = f"https://forecast.weather.gov/MapClick.php?&lat={lat:.2f}&lon={long:.0f}&FcstType=json"
        print(url)
        return requests.get(url).json()

    def get_info(self, lat_long=None, just_now=False):
        weather_json = self.get_weather(lat_long=lat_long)
        new_data = {}
        start_period_names = weather_json['time']['startPeriodName'][:4]
        start_times = weather_json['time']['startValidTime'][:4]
        temp_labels = weather_json['time']['tempLabel'][:4]
        temperatures = weather_json['data']['temperature'][:4]
        pops = weather_json['data']['pop'][:4]
        weathers = weather_json['data']['weather'][:4]
        icon_links = weather_json['data']['iconLink'][:4]
        texts = weather_json['data']['text'][:4]
        location = weather_json.get('location', {}).get('areaDescription')
        
        for start_period, start_time, temp_label, temp, pop, weather, icon_link, text in zip(
            start_period_names, start_times, temp_labels, temperatures, pops, weathers, icon_links, texts
        ):
            new_data[start_period] = {
                'start_time': start_time,
                'temp_label': temp_label,
                'temperature': temp,
                'pop': pop,
                'weather': weather,
                'icon_link': icon_link,
                'text': text,
                'location': location,
            }
            if just_now:
                return new_data[start_period]
        return new_data


# +
# class Image:
#     class Size(enum.Enum):
#         SMALL = "256x256"
#         MEDIUM = "512x512"
#         LARGE = "1024x1024"

#     @classmethod
#     @retrying.retry(stop_max_attempt_number=5, wait_fixed=2000)
#     def create(cls, prompt, n=1, size=Size.LARGE):
#         logger.info('requesting openai.Image...')
#         resp = openai.OpenAI(api_key=openai.api_key).images.generate(prompt=prompt, n=n, size=size.value, model="dall-e-3", response_format='b64_json', timeout=45)
#         logger.info('received openai.Image...')
#         return resp.data[0].b64_json

class Image:
    SIZE = {
        "DALLE2_SMALL": "256x256",
        "DALLE2_MEDIUM": "512x512",
        "DALLE2_LARGE": "1024x1024",
        "DALLE3_SQUARE": "1024x1024",
        "DALLE3_HORIZONTAL": "1792x1024",
        "FOFR_MEDIUM": "512x512",
        "FOFR_HORIZONTAL": "960x480",
        "SDXL": "1024x1024",
        "FLUXSCHNELL_SQUARE": "1024x1024",
    }
    MODEL = {
        "2": "dall-e-2",
        "3": "dall-e-3",
        "fofr": "fofr",
        "SDXL": "SDXL",
        "flux-schnell": "flux-schnell",
    }

    @classmethod
    @retrying.retry(stop_max_attempt_number=5, wait_fixed=2000)
    def create(cls, prompt, n=1, model=MODEL["flux-schnell"], size=SIZE["FLUXSCHNELL_SQUARE"]):
        logger.info(f'requesting Image with prompt={prompt}, n={n}, model={model}, size={size}...')
        
        if model.startswith("dall-e"):
            resp = openai.OpenAI(api_key=openai.api_key).images.generate(prompt=prompt, n=n, size=size, model=model, response_format="b64_json", timeout=45)
            resp = resp.data[0].b64_json
        
        elif model.startswith("SD"):
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            w, h = size.split('x')
            body = {
              "steps": 40,
              "width": int(w),
              "height": int(h),
              "cfg_scale": 5,
              "samples": n,
              "text_prompts": [
                {"text": prompt, "weight": 1},
                {"text": "blurry, bad, ugly", "weight": -1},
              ],
            }
            headers = {
              "Accept": "application/json",
              "Content-Type": "application/json",
              "Authorization": f"Bearer {SD_APIKEY}",
            }
            response = requests.post(url, headers=headers, json=body)
            if response.status_code != 200:
                raise Exception("Non-200 response: " + str(response.text))
            data = response.json()
            res = [img["base64"] for img in data["artifacts"]]
            if n == 1:
                return res[0]
        
        else:  # Handle all Replicate models in a unified way
            width, height = size.split('x')
            width, height = int(width), int(height)
            replicate_api_key = os.environ.get("REPLICATE_APIKEY") or open("/Users/jong/.replicate_apikey").read().strip()
            if model == "flux-schnell":
                url = "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions"
                json_payload = {
                    "input": {
                        "prompt": prompt,
                        "num_outputs": n,
                        "aspect_ratio": "1:1",  # Assuming square output for simplicity
                        "output_format": "webp",
                        "output_quality": 90
                    }
                }
            else:
                url = "https://api.replicate.com/v1/predictions"
                json_payload = {
                    "version": "a83d4056c205f4f62ae2d19f73b04881db59ce8b81154d314dd34ab7babaa0f1",
                    "input": {
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "num_images": n,
                    }
                }

            resp = requests.post(url, headers={"Content-Type": "application/json", "Authorization": f"Token {replicate_api_key}"}, json=json_payload)
            resp = resp.json()
            while resp.get("status", "fail").lower() not in {"fail", "succeeded"}:
                logger.info(f"Sleeping 1...")
                time.sleep(1)
                resp = requests.get(f"https://api.replicate.com/v1/predictions/{resp['id']}", headers={"Content-Type": "application/json", "Authorization": f"Token {replicate_api_key}"})
                resp = resp.json()
            image_data = requests.get(resp['output'][0]).content
            resp = base64.b64encode(image_data).decode()
        
        logger.info('received Image...')
        return resp



# -

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


# +
import textwrap
import warnings


def write_text_on_image_2(text, image_size):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = PIL.Image.new('RGB', image_size, 'white')
        d = PIL.ImageDraw.Draw(img)
        
        initial_font_size = 100
        max_font_size = initial_font_size
        font = PIL.ImageFont.truetype(mono_font_path, max_font_size)
    
        while True:
            # Get the width of a single character ('a' for demonstration)
            char_w, _ = d.textsize('a', font=font)
            
            # Calculate approximate characters per line
            chars_per_line = image_size[0] // char_w
            wrapper = textwrap.TextWrapper(
                width=chars_per_line, expand_tabs=False,
                replace_whitespace=False, break_long_words=False,
                break_on_hyphens=False,
            )
    
            word_list = wrapper.wrap(text=text)
            wrapped_text = '\n'.join(word_list)
            
            text_w, text_h = d.multiline_textsize(wrapped_text, font=font)
    
            if text_w <= image_size[0] and text_h <= image_size[1]:
                break
            
            max_font_size -= 2
            font = PIL.ImageFont.truetype(mono_font_path, max_font_size)
            
        x = (image_size[0] - text_w) / 2
        y = (image_size[1] - text_h) / 2
        
        d.multiline_text((x, y), wrapped_text, fill=(0, 0, 0), font=font)
        
        return img


# -

def resize_img(img):
    # Define the target dimensions for the 2048x2048 image
    target_size = 2048

    # The aspect ratio of the desired container (800x480)
    container_width, container_height = 800, 480
    container_aspect_ratio = container_width / container_height

    # Scale up the container dimensions to fit the 2048x2048 image
    if container_aspect_ratio > 1:
        container_new_width = target_size
        container_new_height = int(container_new_width / container_aspect_ratio)
    else:
        container_new_height = target_size
        container_new_width = int(container_new_height * container_aspect_ratio)

    # Resize the original image to 2048x2048
    resized_image = img.resize((target_size, target_size), PIL.Image.LANCZOS)

    # Create a white background with the scaled container dimensions
    background = PIL.Image.new("RGB", (container_new_width, container_new_height), "white")

    # Calculate the position to paste the resized image onto the background
    position = ((container_new_width - target_size) // 2, (container_new_height - target_size) // 2)

    # Paste the resized image onto the background
    background.paste(resized_image, position)

    return background



# +
from timezonefinder import TimezoneFinder

def find_timezone(latitude, longitude):
    obj = TimezoneFinder()
    
    # The `timezone_at` method performs the lookup
    result = obj.timezone_at(lat=latitude, lng=longitude)
    
    # In case the coordinates do not correspond to any time zone
    if result is None:
        return "UTC"
        
    return result
import pytz

def current_time_in_timezone(timezone_str):
    # Get the current UTC time
    utc_now = datetime.datetime.now(pytz.utc)
    
    # Convert it to the desired timezone
    target_tz = pytz.timezone(timezone_str)
    localized_dt = utc_now.astimezone(target_tz)
    
    # Format the localized datetime
    formatted_time = localized_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return formatted_time


# +
DEFAULT_MODEL = 'gpt-4o-mini'  # gpt-4-1106-preview

class WeatherDraw:
    def clean_text(self, weather_info):
        chat = Chat("Given the following weather conditions, write a very small, concise plaintext summary. Just include the weather, no dates.")
        text = chat.message(str(weather_info)[:4000], model=DEFAULT_MODEL)
        return text

    def generate_image(self, weather_info, resize=True, **kwargs):
        animal = random.choice(animals)
        num_animals = random.randint(1, 3)
        logger.info(f"Got animal {animal}")
        chat = Chat(f'''Given
the following weather conditions, write a plaintext and concise description of an
image of {num_animals} adorable anthropomorphised {animal}{"s" if num_animals > 1 else ""} doing an activity in the weather.
The image should make obvious what the weather is.
The animal should be extremely anthropomorphised.
Only write the short description and nothing else.
Do not include specific numbers.'''.replace('\n', ' '))
        description = chat.message(str(weather_info)[:4000], model=DEFAULT_MODEL)
        prompt = description
        if weather_info.get('location') is not None:
            prompt += f' {weather_info["location"]} background.'
        prompt += f' 8k quality. In the style of {random.choice(ARTISTS)}'
        logger.info(prompt)
        img = Image.create(prompt, **kwargs)
        img_bytes = base64.b64decode(img)
        img = PIL.Image.open(BytesIO(img_bytes))
        img = img.resize((256, 256))
        return img, prompt

    def step_one_forecast(self, weather_info, **kwargs):
        img, txt = self.generate_image(weather_info, **kwargs)
        return img, txt

    def weather_img(self, weather_data):
        return write_text_on_image(
            list(weather_data.values())[0]['text'],
            ((800-480)//2, 480),
        )

    def left_text_img(self, weather_data):
        now = datetime.datetime.now().astimezone(zoneinfo.ZoneInfo("US/Eastern"))
        animal = random.choice(animals)
        chat = Chat(f'Give me a small, concise, rare, cute, fun fact about an animal.')
        return write_text_on_image(
            f'{now:%Y-%m-%d}\n{now:%H:%M}\n\n{chat.message(animal, model=DEFAULT_MODEL)}',
            ((800-480)//2, 480),
        )

    def step(self, zip_code='10001', forecast=None, images=None, **kwargs):
        if forecast is None:
            forecast = Weather(zip_code).get_info()
        self.forecast = forecast
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
                images[ridx] = overlay_text_on_image(img, time, 'top-right', decode=False)
                texts[ridx] = txt

        img = resize_img(create_collage(*images))
        # Assuming container dimensions were scaled to fit the 2048x2048 image
        # Calculate new positions based on the scaled container
        container_width, container_height = 2048, 2048  # Assuming the full 2048x2048 size is being used
        
        # Calculate new x, y positions
        weather_img_position = (container_width - 160, 0)  # Update this according to your needs
        left_text_img_position = (0, 0)  # Assuming this remains unchanged
        
        # Paste images in the new locations
        img.paste(self.weather_img(forecast), weather_img_position)
        img.paste(self.left_text_img(forecast), left_text_img_position)

        return img, *texts

# +
# # %%time
# x = WeatherDraw()
# out = x.step(zip_code='10001')
# out[0]

# +
# x.forecast
# -




