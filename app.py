from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import re


# ==========================================
# 1. FastAPI App
# ==========================================

app = FastAPI(
    title="T5 Text Summarizer",
    description="Text Summarization using Fine-Tuned T5",
    version="1.0"
)


# ==========================================
# 2. Model Path
# ==========================================

MODEL_PATH = "./saved_summary_model"


# ==========================================
# 3. Load Tokenizer and Model
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    legacy=True
)

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_PATH
)


# ==========================================
# 4. Select Device
# ==========================================

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


model.to(device)
model.eval()

print("Using device:", device)


# ==========================================
# 5. Templates and Static Files
# ==========================================

templates = Jinja2Templates(directory="templates")

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ==========================================
# 6. Request Schema
# ==========================================

class DialogueInput(BaseModel):
    dialogue: str


# ==========================================
# 7. Text Cleaning
# ==========================================

def clean_data(text: str) -> str:

    text = re.sub(r"\r\n", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = text.strip().lower()

    return text


# ==========================================
# 8. Summarization Function
# ==========================================

def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return summary


# ==========================================
# 9. API Endpoint
# ==========================================

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(
        dialogue_input.dialogue
    )

    return {
        "summary": summary
    }


# ==========================================
# 10. Home Page
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )