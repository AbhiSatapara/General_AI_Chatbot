# General AI Assistant

A simple, general-purpose conversational AI chatbot built with Streamlit and Groq. No institutional branding and no login are required.

## Features
- Ask questions on any topic (coding, learning, writing, analytics, general knowledge, etc.).
- ChatGPT-style conversation interface with suggested prompts.
- New chats, saved conversation list, delete conversations.
- Chat history stored locally in JSON and restored after refresh.
- Recent conversation context sent to the AI for follow-up questions.
- Streaming response display with friendly API/network/rate-limit error messages.
- Responsive layout for desktop and mobile screens.
- Automatic web search for company, college, organization, current, and other specific-information questions.
- Web sources are shown in an expandable section when search results are available.

## Requirements
- Python 3.10+
- A Groq API key and internet connection (web search is optional and uses DDGS without a separate search API key)


## Setup
1. Extract this project ZIP and open its folder in a terminal.
2. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your Groq API key:
   ```env
   GROQ_API_KEY=your_actual_key
   GROQ_MODEL=openai/gpt-oss-120b
   ```
4. Start the application:
   ```bash
   streamlit run app.py
   ```
5. Open the Streamlit URL shown in your terminal.

## Data and privacy
- Conversation history is saved in `data/chat_history.json` on the machine running the app.
- This version has no login or user separation. If deployed for multiple users, add authentication and a user-scoped database before using it with private conversations.
- Keep `.env` private. Never upload API keys to GitHub.

## Configuration
- `GROQ_API_KEY`: required API key.
- `GROQ_MODEL`: optional model setting; defaults to `openai/gpt-oss-120b`.
- `WEB_SEARCH_ENABLED`: optional setting; defaults to `true`. Set to `false` to disable automatic web search. Search is triggered for company/college/organization and current-information questions. The search provider may be rate-limited or unavailable sometimes.

