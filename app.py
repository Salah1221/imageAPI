from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import fal_client
import google.generativeai as genai
from dotenv import load_dotenv

# Initialize the FastAPI app
app = FastAPI()

class Item(BaseModel):
    prompt: str

def init_api_keys():
    # Load environment variables from .env file
    load_dotenv()
    
    try:
        # Google Generative AI API key
        google_api_key = os.getenv("AIzaSyCR7Rg4Y1mxqVH-nYMIGtiEv7a-zt-hB-M")
        if google_api_key:
            genai.configure(api_key='AIzaSyCR7Rg4Y1mxqVH-nYMIGtiEv7a-zt-hB-M')
        else:
            raise ValueError("Google API key not found in environment variables.")
    except Exception as e:
        raise RuntimeError(f"Error initializing Google API key: {e}")
    
    # Check for FAL API key
    fal_api_key = os.getenv("d84d9857-17de-456d-a3c9-bb4d7b9974da:3482b708facf19e192e92d02878196db")
    if not fal_api_key:
        raise ValueError("FAL API key not found in environment variables.")
    
    return fal_api_key

def generate_image(prompt, fal_api_key):
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
        img_path = "generated_image.jpg"
        img.save(img_path)
        return img_path, prompt
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to download image.")

def generate_response(prompt: str):
    # Generate a response from Google Generative AI model
    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest", generation_config=generation_config)
    prompt_parts = [f"Without any added explanations, just write an Arabic proverb about an image that contains: {prompt}"]

    try:
        response = model.generate_content(prompt_parts)
        if hasattr(response, 'text') and response.text.strip():
            return response.text
        else:
            return "لا توجد استجابة متاحة"
    except Exception as exception:
        return f"Error: {exception}"

def add_stylish_text(image_path, text, font_path, output_path):
    # Add styled text to the generated image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Get image size
    img_width, img_height = img.size
    font_size = int(img_height / 10)

    try:
        # Use the font provided
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        font = ImageFont.load_default()

    text_position = (50, img_height - 100)
    text_color = "white"

    # Draw text with outline for visibility
    outline_range = 2
    for adj in [(0, 0), (outline_range, 0), (-outline_range, 0), (0, outline_range), (0, -outline_range)]:
        draw.text((text_position[0] + adj[0], text_position[1] + adj[1]), text, fill="black", font=font)

    draw.text(text_position, text, fill=text_color, font=font)
    img.save(output_path)
    return output_path

@app.get("/")
async def root():
    return {"message": "Welcome to the image generation API"}

@app.post("/generate_image/")
async def create_image(item: Item):
    # Initialize API keys
    try:
        fal_api_key = init_api_keys()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Generate image and get the prompt
    try:
        image_path, prompt_used = generate_image(item.prompt, fal_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")
    
    # Generate Arabic proverb
    try:
        arabic_proverb = generate_response(prompt_used)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proverb generation failed: {e}")
    
    # Add styled Arabic proverb to the image
    try:
        output_image_path = add_stylish_text(image_path, arabic_proverb, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "final_image_with_text.jpg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add text to image: {e}")
    
    # Return the path of the generated image and the proverb
    return {"image_path": output_image_path, "proverb": arabic_proverb}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
