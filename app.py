from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import fal_client
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time
import textwrap

if not os.path.exists("static"):
    os.makedirs("static")

# Initialize the FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Allow CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify the allowed origins here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    prompt: str

def init_api_keys():
    # Load environment variables from .env file
    load_dotenv()
    
    try:
        # Google Generative AI API key
        api_key = os.getenv("Google_API")
        if api_key:
            genai.configure(api_key=api_key)
        else:
            raise ValueError("Google API key not found in environment variables.")
    except Exception as e:
        raise RuntimeError(f"Error initializing API keys: {e}")

def generate_image(prompt):
    load_dotenv()
    # FAL API key
    # fal_api_key = os.getenv("Fal_API")
    # if not fal_api_key:
    #     raise ValueError("FAL API key not found in environment variables.")
    
    # Use fal-client to generate the image
    handler = fal_client.submit(
        "fal-ai/flux/schnell",
        arguments={"prompt": prompt},
    )
    
    result = handler.get()
    image_url = result['images'][0]['url']
    
    # Download the image
    response = requests.get(image_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img_path = f"static/final_image{time.time()}.jpg"
        img.save(img_path)
        return img_path, prompt
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")

def generate_response(prompt: str):
    # Generate a response from Google Generative AI model
    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest", generation_config=generation_config)
    prompt_parts = [f"without any added explanations, just write an Arabic proverb about image contains: {prompt}"]

    try:
        response = model.generate_content(prompt_parts)
        if hasattr(response, 'text') and response.text.strip():
            return response.text
        else:
            return "لا توجد استجابة متاحة"
    except Exception as exception:
        return f"Error: {exception}"

def add_stylish_text(image_path, text, font_path, output_image_path, margin=30, max_width=40, stroke_width=2):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Load the font
    font = ImageFont.truetype(font_path, size=40)
    
    # Wrap the text
    wrapped_text = textwrap.fill(text, width=max_width)
    
    # Calculate text size and position
    lines = wrapped_text.split('\n')
    line_height = font.getbbox('A')[3] - font.getbbox('A')[1]
    total_text_height = (line_height + margin + 10) * len(lines)
    
    image_width, image_height = image.size
    y = image_height - total_text_height - margin
    
    # Draw each line of the wrapped text with stroke
    for line in lines:
        line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
        x = (image_width - line_width) // 2  # Center the text horizontally
        draw.text((x, y), line, font=font, fill="white", stroke_width=stroke_width, stroke_fill="black")
        y += line_height + margin
    
    # Save the image
    image.save(output_image_path)
    
    return output_image_path


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.head("/")
async def head_root():
    return

@app.post("/generate_image/")
async def create_image(item: Item):
    # Initialize API keys
    init_api_keys()
    
    # Generate image and get the prompt
    image_path, prompt_used = generate_image(item.prompt)
    
    # Generate Arabic proverb
    arabic_proverb = generate_response(prompt_used)
    
    # Add styled Arabic proverb to the image
    output_image_path = add_stylish_text(image_path, arabic_proverb, "static/Cairo-Black.ttf", image_path)
    
    # Return the path of the generated image and the proverb
    return {"image_path": output_image_path, "proverb": arabic_proverb}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
