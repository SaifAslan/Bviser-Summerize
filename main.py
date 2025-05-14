from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI  # Import OpenAI for DeepSeek client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client for DeepSeek
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# Models for request and response bodies
class MeetingSummaryRequest(BaseModel):
    minutes: str
    title: str
    creator_name: str
    date: str

class AgendaSummaryRequest(BaseModel):
    agenda: str
    title: str

class CustomPromptRequest(BaseModel):
    custom_prompt: str

# In-memory store for custom prompts
custom_prompt = "يرجى تقديم ملخص موجز ومهني لمحضر الاجتماع التالي بدون اي marking down, اكتب العنوان في الاول ثم التلخيص ثم اسم الشخص و التاريخ. اذا كان بالعربية لحص بالعربية و اذا كان يالانجليزية لخص بالانحليزية. يجب انغلب الطابع الرسمي علي التلخيص." 

# Endpoint to summarize minutes of the meeting
@app.post("/summarize")
async def summarize_meeting(request: MeetingSummaryRequest):
    try:
        global custom_prompt
        
        # Prepare input for DeepSeek model
        input_text = (
            f"{custom_prompt}\n\n"
            f"Title: {request.title}\n"
            f"Creator: {request.creator_name}\n"
            f"Date: {request.date}\n\n"
            f"Minutes:\n{request.minutes}"
        )
        
        # Call DeepSeek model
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": input_text},
            ],
            stream=False
        )

        # Extract and return the summary
        summary = response.choices[0].message.content
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating summary: {e}")

@app.post("/summarize-agenda")
async def summarize_agenda(request: AgendaSummaryRequest):
    try:
        agenda_prompt = "لخص بند الاعمال هذا باللغة التي ياتي بها٬ اذا كان بالعربية لحص بالعربية و اذا كان يالانجليزية لخص بالانحليزية. يجب انغلب الطابع الرسمي علي التلخيص." 
        
        # Prepare input for DeepSeek model
        input_text = (
            f"{agenda_prompt}\n\n"
            f"Title: {request.title}\n"
            f"Agenda:\n{request.minutes}"
        )
        
        # Call DeepSeek model
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": input_text},
            ],
            stream=False
        )

        # Extract and return the summary
        summary = response.choices[0].message.content
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating summary: {e}")

# Endpoint to retry summarization with different phrasing
@app.post("/retry-summary")
async def retry_summary():
    try:
        global last_prompt

        if not last_prompt:
            raise HTTPException(status_code=400, detail="No previous prompt to retry.")
        
        # Retry using the last input
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": last_prompt},
            ],
            stream=False
        )
        summary = response.choices[0].message.content
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while retrying summary: {e}")

# Endpoint to customize the summarization prompt
@app.post("/customize-prompt")
async def customize_prompt(request: CustomPromptRequest):
    try:
        global custom_prompt
        custom_prompt = request.custom_prompt
        return {"message": "Custom prompt updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while updating custom prompt: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
