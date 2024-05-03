## About

Personalised AI sports trainer

## Demo

[![Watch the video](https://img.youtube.com/vi/jzyasbyhzpw/maxresdefault.jpg)](https://youtu.be/jzyasbyhzpw)

## Built With

OpenAI llm, Python, Streamlit

## Features

- General ask and answer
- Record workout entry
- Answer questions from workout logs, training related data
- Custom persona:
  - sports specialty
  - speaking style

## Requirements:

- IDE
- OPEN AI API Key

## User Guide

#### To run locally

1. Download repository in a folder
2. Open VScode from folder
3. In VS Code terminal:
   Create virtual environment: python3 -m venv .venv<br>
   Install required dependencies: pip install -r requirements.txt<br>
   Optional: Create .env file with API Key
4. To run app locally:<br>
   (In Terminal): streamlit run Main.py<br>

## User/Technical Flow Chart

https://www.figma.com/file/nJ8bhalNFvyOFGvDFpSP09/AI-Chatbot-Flow-Chart?type=whiteboard

## Roadmap

### Completed

- V1.0.0 build with langchain agent, and RAG function [27 Apr 2024]:
  - Feature:
    - ask and answer
    - retrieve from context
    - write file to add formatted entry
  - Customisation on FrontEnd:
    - Input API Key
    - Input context file paths
    - Speaking Tone
    - Sports Expertise
- V2.0.0 build with langgraph agent [3 May 2024]:
  - Improved agent response but uses more token

### Upcoming

- Improve agent function:
  - to implement multi agent collaboration
- Add function to update, delete training log entry
- Add other llm models
