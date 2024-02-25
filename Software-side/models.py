#Open AI -> Cookbook for vide understanding and response generation! 
#Using Open AI / GPT 4 to send a video by dividing the video into multiple frames, sending multiple frames to Open AI with a prompt to generate output, and then text to speech conversion again using the open - ai key 
# Main Next Steps to Think about ? :
# 0. Figure out where we need to add out open AI API key, and how much that would cost!
# Do a trial with a random video and see how the output looks like! -> And see it works before going on to live stream! 
# Once it works with a random video -> Capture the live stream and then send the frames to OpenAI:
    # How many seconds of video can it process at a time?
    # Would there be a lag in response time? -> How much would the lag be and can I reduce it in anyway ? 
    # How much would it cost to process a say 1 hour video?
#----------------------------------------------------------
# 1. Add access to live stream and then send the frames to OpenAI
# 2. Convert incoming audio to text and send it to the model as a prompt!
# 3. Do I want to use anything open source instaed of OpenAI for certain sections such as the text to speech conversion and speech to text conversion?
from IPython.display import display, Image, Audio
import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

video = cv2.VideoCapture("data/bison.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")


display_handle = display(None, display_id=True)
for img in base64Frames:
    display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
    time.sleep(0.025)

PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::60]),
        ],
    },
]

params = {
    "model": "gpt-4-vision-preview",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 500,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)

response = requests.post(
    "https://api.openai.com/v1/audio/speech",
    headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    },
    json={
        "model": "tts-1-1106",
        "input": result.choices[0].message.content,
        "voice": "onyx",
    },
)

audio = b""
for chunk in response.iter_content(chunk_size=1024 * 1024):
    audio += chunk
Audio(audio)