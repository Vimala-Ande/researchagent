import os
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI(
    title="AI Research Paper Analysis Agent",
    description="Upload a research paper PDF and get an AI-generated analysis.",
    version="1.0.0",
)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0,
)

def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)

    return "\n".join(parts).strip()


def analyze_research_paper(paper_text: str) -> str:
    if not paper_text.strip():
        raise ValueError(
            "No readable text was found in this PDF. "
            "Please upload a text-based research paper PDF."
        )

    max_chars = 120000

    if len(paper_text) > max_chars:
        paper_text = paper_text[:max_chars] + (
            "\n\n[Note: The PDF was very large, so only the first "
            "part of the extracted text was analyzed.]"
        )

    prompt = f"""
You are an AI Research Paper Analysis Agent.

Carefully analyze the research paper below.

================ RESEARCH PAPER ================

{paper_text}

==================================================

Generate a clear and structured analysis using these sections:

1. Paper Title
2. Authors
3. Abstract / Summary
4. Problem Statement
5. Objectives
6. Proposed Solution
7. Methodology
   Explain the methodology step by step.
8. Technologies / Tools Used
9. Results and Findings
10. Advantages
11. Limitations
12. Future Scope
13. Keywords
14. Conclusion
15. Important Contributions
16. Viva Questions
   Generate 5 important viva questions with short answers.

IMPORTANT RULES:
- Analyze only information available in the paper.
- Do not invent facts.
- If information is not available, write "Not mentioned in the paper."
- Keep the explanation simple and understandable.
- Use clear headings and bullet points.
- Preserve important technical terms from the paper.
"""

    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        content = "".join(parts)

    return str(content).strip()


# Home page redirects users to the required playground URL.
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=/agent/playground/">
        <title>AI Research Paper Analysis Agent</title>
    </head>
    <body>
        <p>Opening Research Paper Analysis Agent...</p>
        <p>
            If you are not redirected,
            <a href="/agent/playground/">open the Agent Playground</a>.
        </p>
    </body>
    </html>
    """


# This is the exact URL the user wants:
# https://YOUR-RENDER-URL/agent/playground/
@app.get("/agent/playground/", response_class=HTMLResponse)
async def agent_playground():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Research Paper Analysis Agent</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            margin: 0;
            padding: 30px;
        }

        .container {
            max-width: 950px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 14px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        }

        h1 {
            margin-top: 0;
            text-align: center;
        }

        .description {
            text-align: center;
            color: #555;
        }

        .upload-box {
            margin-top: 25px;
            padding: 25px;
            border: 2px dashed #aaa;
            border-radius: 12px;
            text-align: center;
        }

        input[type="file"] {
            margin: 15px;
        }

        button {
            background: #222;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
        }

        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        #status {
            margin-top: 20px;
            font-weight: bold;
            text-align: center;
        }

        #result {
            margin-top: 25px;
            padding: 25px;
            background: #f8f8f8;
            border-radius: 10px;
            white-space: pre-wrap;
            line-height: 1.6;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>AI Research Paper Analysis Agent</h1>

    <p class="description">
        Upload any text-based research paper PDF and get a structured AI analysis.
    </p>

    <div class="upload-box">

        <input
            id="pdfFile"
            type="file"
            accept=".pdf,application/pdf"
        >

        <br>

        <button id="analyzeButton">
            Analyze Research Paper
        </button>

    </div>

    <div id="status"></div>

    <div id="result"></div>

</div>

<script>

const button = document.getElementById("analyzeButton");
const fileInput = document.getElementById("pdfFile");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("result");

button.addEventListener("click", async () => {

    const file = fileInput.files[0];

    if (!file) {
        statusBox.textContent = "Please select a PDF file.";
        return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
        statusBox.textContent = "Please upload a PDF file.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    button.disabled = true;
    statusBox.textContent = "Analyzing paper... Please wait.";
    resultBox.textContent = "";

    try {

        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Analysis failed.");
        }

        statusBox.textContent = "Analysis completed successfully.";

        resultBox.textContent = data.analysis;

    } catch (error) {

        statusBox.textContent = "Error";

        resultBox.textContent = error.message;

    } finally {

        button.disabled = false;

    }

});

</script>

</body>
</html>
"""


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select a PDF file."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    try:

        pdf_bytes = await file.read()

        if not pdf_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty."
            )

        paper_text = extract_pdf_text(pdf_bytes)

        analysis = analyze_research_paper(paper_text)

        return {
            "filename": file.filename,
            "analysis": analysis
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
