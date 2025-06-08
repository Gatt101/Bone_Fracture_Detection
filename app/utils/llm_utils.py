import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def clean_llm_response(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def generate_suggestion(severity, confidence, chat_history=None):
    try:
        prompt = f"""You are an orthopedic radiologist.

Generate a complete markdown report based on this template:

# Fracture Analysis Report

## Severity Assessment
- Classification: {severity}
- Confidence Score: {confidence*100:.2f}%
- Key Radiographic Features: [List 3-4 general characteristics]

## Urgency Level
- Priority Level: [Emergency/Urgent/Semi-Urgent]
- Recommended Action Timeline: [Immediate/Within 24h/Within 72h]
- Triage Considerations: [General risk factors]

## Clinical Recommendations
1. [Standard imaging protocol]
2. [Appropriate pain management]
3. [General mobility restrictions]
4. [Specialist referral protocol]
5. [Standard follow-up schedule]

## Treatment Complexity
- Complexity Tier: [High/Medium/Low]
- Potential Interventions:
  - [General surgical approach]
  - [Alternative treatment options]
  - [Standard rehabilitation plan]

Only return the structured markdown content without explanation or extra text.
Do NOT mention specific anatomical locations.
Focus on fracture characteristics rather than location.
"""

        payload = {
            "model": "qwen-qwq-32b",
            "messages": [
                {"role": "system", "content": "Respond using markdown format. Describe fracture characteristics generally without anatomical location."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 600
        }

        response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)

        if response.status_code == 200:
            raw = response.json()['choices'][0]['message']['content']
            markdown_start = raw.find("# Fracture Analysis Report")
            if markdown_start != -1:
                content = raw[markdown_start:].strip()
            else:
                content = raw.strip()

            content = re.sub(r"\b(femoral|tibial|humeral|radius|ulna|vertebral)\b", "", content, flags=re.IGNORECASE)
            return content

        return f"""# Fracture Analysis Report

## Severity Assessment
- Classification: {severity}
- Confidence Score: {confidence*100:.2f}%
- Key Radiographic Features: 
  - Significant fracture pattern detected
  - Clinical correlation recommended

## Urgency Level
- Priority Level: {"Emergency" if severity == "Severe" else "Urgent"}
- Recommended Action Timeline: {"Immediate" if severity == "Severe" else "Within 24h"}
- Triage Considerations: Requires urgent orthopedic evaluation

## Clinical Recommendations
1. Comprehensive imaging workup
2. Appropriate analgesia protocol
3. Immobilization measures
4. Immediate orthopedic consultation
5. Follow-up within 24-48 hours

## Treatment Complexity
- Complexity Tier: {"High" if severity == "Severe" else "Medium"}
- Potential Interventions:
  - Surgical stabilization options
  - Non-operative management alternatives
  - Post-treatment rehabilitation program
"""

    except Exception as e:
        return f"""# Fracture Analysis Report

## Severity Assessment
- Classification: {severity}
- Confidence Score: {confidence*100:.2f}%
- Key Radiographic Features: 
  - Fracture pattern requiring attention
  - Detailed evaluation needed

## Urgency Level
- Priority Level: {"Emergency" if severity == "Severe" else "Urgent"}
- Recommended Action Timeline: {"Immediate" if severity == "Severe" else "Within 24h"}

## Clinical Recommendations
1. Urgent orthopedic consultation
2. Standard fracture management protocol
3. Immobilization measures

## Treatment Complexity
- Complexity Tier: {"High" if severity == "Severe" else "Medium"}"""

def generate_chatbot_response(user_input, chat_history=None):
    system_prompt = (
        "You are an expert orthopedic AI assistant. "
        "Help with fracture assessments, image interpretation, and treatment suggestions. "
        "Consider the conversation history to provide contextually relevant responses. "
        "Format your responses in markdown with proper headers, bullet points, and emphasis where appropriate. "
        "Do NOT include <think> tags, internal thoughts, or reasoning steps. "
        "Only return the final response directly to the user."
    )

    messages_for_llm = [{"role": "system", "content": system_prompt}]

    if chat_history:
        messages_for_llm.extend(chat_history[-10:])

    messages_for_llm.append({"role": "user", "content": user_input})

    payload = {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "messages": messages_for_llm,
        "temperature": 0.7,
        "max_tokens": 300
    }

    response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        raw_text = response.json()['choices'][0]['message']['content'].strip()
        cleaned = clean_llm_response(raw_text)
        return cleaned
    else:
        return f"## Error\nFailed to get response from server (Status: {response.status_code})" 