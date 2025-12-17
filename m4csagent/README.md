# AI Customer Support Agent
An intelligent customer support agent built with LangGraph, Groq LLM, and Streamlit. This AI-powered assistant provides instant help with FAQs, account issues, and personalized support based on user history. The system uses a graph-based workflow to intelligently route queries and includes a Human-in-the-Loop (HITL) mechanism for high-risk queries.

## Features

- **Intent Detection**: Automatically classifies user queries into FAQ, account actions, history, or escalation categories
- **FAQ Handling**: Responds to common questions about password reset, account setup, billing, and features using an intelligent knowledge base
- **User History**: Maintains personalized context across sessions with long-term memory storage
- **Memory Management**:
  - Short-term memory for in-session context (last 5 messages)
  - Long-term memory for cross-session continuity (user history)
  - Automatic message filtering and context window management
- **Human-in-the-Loop (HITL)**: Automatically escalates high-risk queries containing keywords like "error", "complaint", "refund", "lawsuit" to human support
- **Smart Routing**: Uses LangGraph's conditional edges to route queries to appropriate handlers
- **State Management**: Built with TypedDict-based state management for reliable data flow
- **Empathetic Responses**: Provides clear, helpful answers with a professional tone

## Prerequisites

- **Python**: Version 3.8 or higher
- **Virtual Environment**: Recommended to isolate dependencies
- **API Keys**:
  - Groq API key (for LLM-powered intent detection and responses)



## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd m4csagent
```

### 2. Create Virtual Environment
Create a virtual environment named `env4a`:
```bash
python -m venv env4a
```

### 3. Activate Virtual Environment
**On macOS/Linux:**
```bash
source env4a/bin/activate
```

**On Windows:**
```powershell
env4a\Scripts\activate
```

### 4. Install Dependencies
Install required packages from `requirements.txt`:
```bash
pip install -r app/requirements.txt
```

**Required packages:**
```
streamlit
langchain
langchain_groq
langgraph
python-dotenv
```

### 5. Set Up Environment Variables
Create a `.env` file in the project root with your Groq API key:
```env
GROQ_API_KEY=<your-groq-api-key>
```

**Example:**
```env
GROQ_API_KEY=gsk_abc123xyz...
```

### 6. Run the Application
Start the Streamlit app from the project root:
```bash
streamlit run app/app.py
```

Access the app at `http://localhost:8501` in your browser.

### 7. Deactivate Virtual Environment (when done)
```bash
deactivate
```



## Usage

### Chat Interface
Use the conversational UI to interact with the support agent:

**FAQ Questions:**
- "How do I reset my password?"
- "How can I set up my account?"
- "Where can I view my billing history?"
- "How do I upgrade to premium features?"

**History Queries:**
- "What did I ask you before?"
- "Can you remind me what we discussed last time?"

**High-Risk Queries (Auto-escalated to Human Support):**
- "This feature is not working properly"
- "I want to file a complaint"
- "I need a refund for my subscription"
- "There's an error when I try to login"

### Example Interactions

**User:** "How do I reset my password?"  
**Agent:** Provides step-by-step instructions from the FAQ knowledge base

**User:** "I want a refund immediately"  
**Agent:** Escalates to human support with ticket ID and expected response time

**User:** "What did we discuss last time?"  
**Agent:** Retrieves and references your previous conversation history

## Project Structure
```
m4csagent/
├── env4a/                  # Virtual environment directory
├── app/
│   ├── app.py             # Main Streamlit application
│   └── requirements.txt   # Python dependencies
├── csagent.ipynb          # Jupyter notebook with development/testing
├── .env                   # Environment variables (API keys)
├── .gitignore             # Excludes env4a/, .env
└── README.md              # Project overview and setup
```

## Architecture

### Workflow Graph
The agent uses LangGraph to create a state machine with the following nodes:
1. **Intent Detection** - Classifies user input and sets HITL flag
2. **FAQ Handler** - Responds to common questions using knowledge base
3. **History Handler** - Retrieves and personalizes based on user history
4. **Human-in-the-Loop** - Escalates high-risk queries
5. **Fallback** - Handles unclear or unknown requests

### State Management
Uses TypedDict-based `AgentState` to maintain:
- User metadata (user_id, thread_id)
- Current input and detected intent
- Short-term memory (last 5 messages)
- Long-term memory (persistent user history)
- HITL flag for escalation

### Memory System
- **MemorySaver**: In-memory storage for cross-session persistence
- **State Reducer**: Filters irrelevant messages and manages context window
- Automatic trimming to prevent context overflow

## Notes

- **LLM Model**: Uses Groq's `llama-3.3-70b-versatile` for fast, accurate responses
- **Debugging**: Check terminal logs for intent detection and routing decisions
- **Security**: Store API keys in `.env` to avoid hardcoding
- **Rate Limits**: Monitor Groq API usage based on your plan tier

## Troubleshooting

- **Verify API key**: Ensure `GROQ_API_KEY` is set correctly in `.env`
- **Intent misclassification**: Check logs to see detected intent; adjust keywords if needed
- **Dependencies**: Upgrade pip if installation issues arise:
  ```bash
  python -m pip install --upgrade pip
  ```
- **Port conflicts**: If port 8501 is in use, specify different port:
  ```bash
  streamlit run app/app.py --server.port 8502
  ```

## Development

To modify the agent behavior:
- **Add new intents**: Update `detect_intent()` function and add corresponding nodes
- **Customize FAQ responses**: Modify `faq_context` in `faq_query_node()`
- **Adjust HITL keywords**: Update `high_risk_keywords` list in `detect_intent()`
- **Change memory limits**: Adjust trimming logic in `state_reducer()`

## License

This project is for educational and demonstration purposes.