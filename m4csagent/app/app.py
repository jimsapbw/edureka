# AI Customer Support Agent
# Built with LangGraph, Groq LLM, and Streamlit

# Step 1 Import necessary libraries

import streamlit as st
import asyncio
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from typing import TypedDict, Optional, Dict, List
import re
from dotenv import load_dotenv
import os
import requests
import logging

# Step 2 Load environment variables and configure logging

# === CONFIG ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 3 Define constants
# INTENT_DETECTION_NODE = "Intent Detection": Defines a constant string that will be used as the name of the intent detection node in the LangGraph. This improves readability and makes it easier to refer to this node.
INTENT_DETECTION_NODE = "Intent Detection"

# Step 4 State definition.
# class AgentState(TypedDict): Defines a typed dictionary called AgentState.
# This class specifies the structure of the agent's memory and metadata
# that will be passed between nodes in the LangGraph.


class AgentState(TypedDict): # Define the structure of the agent's state
    # Metadata
    user_id: str  # Unique identifier for the user
    thread_id: str  # Identifier for the current conversation/thread
    
    # Core input/output
    user_input: str  # Latest input from the user
    intent: Optional[str]  # Detected intent of the user's input (e.g., "query", "action")
    data: Optional[dict]  # Dictionary to store the output/response from a specific node
    
    # Memory
    short_term_memory: Optional[List[Dict[str, str]]]  
    # In-session memory: list of message dictionaries (e.g., {"role": "user", "content": "..."})
    # Captures the immediate conversational context
    
    long_term_memory: Optional[Dict[str, List[Dict[str, str]]]]  
    # Cross-session memory: dictionary keyed by categories (e.g., "user_history")
    # Each category stores a list of past queries and resolutions
    
    # Flags
    hitl_flag: Optional[bool]  
    # Human-in-the-loop flag for high-risk queries

#Step 5 === LLM ===This comment indicates the initialization of the language model.
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")

# Step 6 Define MemorySaver for persistent memory storage

class MemorySaver: # Simple in-memory storage for demonstration purposes
    def __init__(self): # Initialize the memory store
        self.store = {} # Dictionary to hold user memories

    def load(self, user_id: str): # Load memory for a given user_id
        return self.store.get(user_id) # Return the memory if it exists, else None

    def save(self, user_id: str, memory: dict): # Save memory for a given user_id
        self.store[user_id] = memory # Store the memory in the dictionary

# Step 7 Initialize MemorySaver instance
memory_saver = MemorySaver()

# Step 8 Define state reducer for message trimming and filtering

def state_reducer(state: AgentState, new_message: Dict[str, str]) -> AgentState: # Define a reducer function to update state with new messages
    """
    Reducer to update state with new messages while managing context window.
    
    Features:
    - Filters out irrelevant messages (greetings, common pleasantries)
    - Trims short-term memory to last 5 messages to manage context window size
    
    Args:
        state: Current AgentState
        new_message: Dictionary with "role" (user/assistant) and "content" keys
    
    Returns:
        Updated AgentState with filtered and trimmed short-term memory
    """
    # Get existing short-term memory
    messages = state.get("short_term_memory", [])
    
    # Filter out irrelevant messages (simple heuristic)
    irrelevant_phrases = ["hello", "hi", "thanks", "goodbye", "hey", "thank you"]
    if any(new_message["content"].lower().strip().startswith(p) for p in irrelevant_phrases):
        logger.info(f"Filtered out irrelevant message: {new_message['content'][:20]}...")
        return state
    
    # Append new message
    messages.append(new_message)
    
    # Trim to last 5 messages to manage context window
    if len(messages) > 5:
        messages = messages[-5:]
        logger.info("Trimmed short-term memory to last 5 messages")
    
    return {**state, "short_term_memory": messages}

# Step 9 === USER HISTORY FETCH ===
# This node retrieves past queries/resolutions from long-term memory
# and enriches the current response with personalization.

async def fetch_user_history(state: AgentState) -> AgentState: # Async function to fetch user history and personalize response
    user_input = state['user_input'] # Get the latest user input
    user_id = state.get('user_id') #    Get user ID from state
    
    # Load memory from MemorySaver
    saved_memory = memory_saver.load(user_id) # Load saved memory for the user
    if saved_memory: # If memory exists, retrieve long-term memory
        long_term_memory = saved_memory.get('long_term_memory', {}) # Get long-term memory
    else:
        long_term_memory = state.get('long_term_memory', {}) # Fallback to state memory
    
    user_history = long_term_memory.get('user_history', []) # Get user history list
    
    try:
        # Construct a prompt to personalize response
        prompt = (
            f"User asked: {user_input}\n"
            f"Here is their past history: {user_history}\n"
            f"Provide a helpful response that references relevant past queries if applicable. "
            f"Keep tone empathetic and clear."
        ) # Create prompt for LLM
        
        # Use ainvoke with proper message format
        response = await llm.ainvoke([{"role": "user", "content": prompt}]) # Call LLM asynchronously
        message = response.content.strip() # Extract response message
        
        # Append current query to user history
        user_history.append({"query": user_input, "resolution": message}) # Add new entry
        
        # Keep only last 5 entries to prevent unbounded growth
        if len(user_history) > 5:
            user_history = user_history[-5:] # Retain last 5 entries only
        
        long_term_memory["user_history"] = user_history # Update long-term memory
        
        # Save updated memory to MemorySaver
        memory_saver.save(user_id, {"long_term_memory": long_term_memory}) # Save memory
        
        logger.info(f"Fetched user history for user_id: {user_id}") # Log success
        
        return {
            **state,
            "long_term_memory": long_term_memory,
            "data": {"response": message}
        } # Return updated state with response
    
    except Exception as e:
        logger.error(f"Error in fetch_user_history: {e}") # Log any errors
        return {
            **state,
            "data": {"response": "I apologize, but I encountered an error retrieving your history."} # Return error response
        }

# Step 10
# === INTENT DETECTION FUNCTION ===
# Defines an asynchronous function that takes the current AgentState
# and returns an updated AgentState with the detected intent.

async def detect_intent(state: AgentState) -> AgentState:
    user_input = state['user_input']  # Extract latest user input
    short_term_memory = state.get('short_term_memory', [])  # Should be List, not Dict
    long_term_memory = state.get('long_term_memory', {})

    # Construct prompt for LLM classification
    prompt = (
        "Classify the user's intent into one of: "
        "'faq', 'account_action', 'history', 'human_in_the_loop', or 'unknown'.\n"
        f"User input: {user_input}\n"
        f"Previous messages: {short_term_memory[-3:] if short_term_memory else 'none'}\n"
        f"Long-term context: {long_term_memory.get('user_history', 'none')}\n"
        "Respond with only the intent name."
    )

    # Call LLM with proper message format
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    content = response.content.strip().lower()

    # Match against predefined intents
    match = re.search(r"(faq|account_action|history|human_in_the_loop)", content) # Regex to find intent
    intent = match.group(1) if match else "unknown" # Default to "unknown" if no match

    # Define high-risk keywords for HITL escalation
    high_risk_keywords = [
        "error", "not working", "complaint", "refund", "charge dispute",
        "escalate", "manager", "lawsuit"
    ] # Keywords indicating high-risk queries
    hitl_flag = any(keyword in user_input.lower() for keyword in high_risk_keywords) # Check for high-risk keywords

    logger.info(f"Detected intent: {intent}, HITL flag: {hitl_flag}")

    # Use state_reducer to add user message to short-term memory
    updated_state = state_reducer(state, {"role": "user", "content": user_input}) # Update state with new user message

    return {
        **updated_state,
        "intent": intent,
        "hitl_flag": hitl_flag
    } # Return updated state with intent and HITL flag

# Step 11 Define FAQ Query Node

async def faq_query_node(state: AgentState) -> AgentState:
    """
    FAQ Query Node - Handles frequently asked questions using LLM.
    
    Responds to common queries like:
    - Password reset
    - Account setup
    - Billing questions
    - Feature information
    
    Args:
        state: Current AgentState with user_input
    
    Returns:
        Updated AgentState with FAQ response in data field
    """
    user_input = state['user_input']
    short_term_memory = state.get('short_term_memory', []) # Get short-term memory
    
    try:
        # Define common FAQ knowledge base
        faq_context = """
        Common FAQs:
        - Password Reset: Go to Settings > Security > Reset Password. Click 'Forgot Password' and follow email instructions.
        - Account Setup: Navigate to Profile > Complete Setup Wizard. Ensure all required fields are filled.
        - Billing: View invoices under Billing > History. Update payment method in Billing > Payment Methods.
        - Feature Access: Premium features require subscription upgrade. Go to Plans > Upgrade.
        - Contact Support: Email support@company.com or use in-app chat for urgent issues.
        """ # FAQ knowledge base
        
        # Construct prompt with FAQ context and conversation history
        conversation_context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in short_term_memory[-3:]]
        ) if short_term_memory else "No previous context" # Last 3 messages
        
        prompt = (
            f"{faq_context}\n\n"
            f"Previous conversation:\n{conversation_context}\n\n"
            f"User question: {user_input}\n\n"
            f"Provide a clear, helpful answer based on the FAQ knowledge above. "
            f"If the question is not covered in FAQs, politely suggest contacting support. "
            f"Keep response concise and actionable."
        )
        
        # Call LLM to generate FAQ response
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        message = response.content.strip()
        
        logger.info(f"FAQ Query processed: {user_input[:50]}...")
        
        # Update state with assistant's response
        updated_state = state_reducer(state, {"role": "assistant", "content": message})
        
        return {
            **updated_state,
            "data": {"response": message, "source": "faq"}
        }
    
    except Exception as e:
        logger.error(f"Error in faq_query_node: {e}")
        return {
            **state,
            "data": {"response": "I apologize, but I encountered an error processing your FAQ query. Please try again or contact support."}
        }

# Step 12 Define Human-in-the-Loop Node

async def human_in_the_loop(state: AgentState) -> AgentState: # Define HITL node for high-risk queries
    """
    Human-in-the-Loop Node - Handles high-risk queries that require human review.
    
    Triggered when:
    - hitl_flag is True (queries containing error, complaint, refund, etc.)
    - Intent is explicitly 'human_in_the_loop'
    
    Args:
        state: Current AgentState with user_input and hitl_flag
    
    Returns:
        Updated AgentState with escalation message in data field
    """
    user_input = state['user_input']
    user_id = state.get('user_id', 'unknown')
    thread_id = state.get('thread_id', 'unknown')
    
    try:
        # Construct escalation message
        prompt = (
            f"The query '{user_input}' has been flagged as high-risk and requires human review.\n\n"
            f"A support specialist will review your case shortly. "
            f"Your ticket ID is: {thread_id}\n\n"
            f"In the meantime, you can:\n"
            f"- Email us at support@company.com with your ticket ID\n"
            f"- Check our status page for any ongoing issues\n"
            f"- Review our help documentation\n\n"
            f"We appreciate your patience and will respond within 2 business hours."
        )
        
        message = prompt
        
        # Log the escalation for human review
        logger.warning(
            f"HITL Escalation - User: {user_id}, Thread: {thread_id}, "
            f"Query: {user_input[:100]}..."
        ) # Log HITL escalation
        
        # Update state with assistant's escalation message
        updated_state = state_reducer(state, {"role": "assistant", "content": message}) #   Update state with escalation message
        
        return {
            **updated_state,
            "data": {
                "response": message,
                "escalated": True,
                "ticket_id": thread_id
            }
        } # Return updated state with escalation info
    
    except Exception as e:
        logger.error(f"Error in human_in_the_loop: {e}")
        return {
            **state,
            "data": {
                "response": "Your query has been escalated to our support team. Please contact support@company.com for immediate assistance.",
                "escalated": True
            }
        } # Return updated state with escalation info

# Step 13 Define Fallback Node

async def fallback(state: AgentState) -> AgentState:
    """
    Fallback Node - Handles unknown or unclassified intents.
    
    Triggered when:
    - Intent is 'unknown'
    - No other node matches the user's request
    
    Provides helpful guidance on what the agent can assist with.
    
    Args:
        state: Current AgentState with user_input
    
    Returns:
        Updated AgentState with fallback message in data field
    """
    user_input = state['user_input']
    
    try:
        # Construct helpful fallback message
        message = (
            "🤔 I'm not sure I understood your request.\n\n"
            "I can help you with:\n"
            "• **FAQs** - Password reset, account setup, billing questions\n"
            "• **Account Actions** - Update profile, manage settings\n"
            "• **History** - Review your past queries and interactions\n"
            "• **Complex Issues** - Escalate to human support\n\n"
            "Please rephrase your question or try asking about one of these topics."
        )
        
        logger.info(f"Fallback triggered for input: {user_input[:50]}...")
        
        # Update state with assistant's fallback message
        updated_state = state_reducer(state, {"role": "assistant", "content": message})
        
        return {
            **updated_state,
            "data": {"response": message, "source": "fallback"}
        }
    
    except Exception as e:
        logger.error(f"Error in fallback: {e}")
        return {
            **state,
            "data": {"response": "I apologize, but I encountered an error. Please try again or contact support."}
        }

# Step 14 Build & Compile Graph for Customer Support App

# Define routing function to determine next node based on intent
def route_to_node(state: AgentState) -> str:
    """
    Determines the next node to execute based on the current state.
    
    Routing logic:
    1. If hitl_flag is True -> escalate to human support
    2. If intent is valid -> route to corresponding handler
    3. Otherwise -> fallback node
    
    Args:
        state: Current AgentState with intent and hitl_flag
    
    Returns:
        Name of the next node to execute
    """
    # Check for human escalation first
    if state.get("hitl_flag", False):
        return "human_in_the_loop"
    
    # Route based on intent
    intent = state.get("intent", "unknown")
    
    # Map account_action to faq (they're similar)
    if intent == "account_action":
        return "faq"
    
    # Define valid intents that map to specific nodes
    valid_intents = ["faq", "history"]
    
    if intent in valid_intents:
        return intent
    
    # Default to fallback for unknown intents
    return "fallback"


# Initialize the StateGraph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node(INTENT_DETECTION_NODE, detect_intent)
workflow.add_node("faq", faq_query_node)
workflow.add_node("history", fetch_user_history)
workflow.add_node("human_in_the_loop", human_in_the_loop)
workflow.add_node("fallback", fallback)

# Set entry point
workflow.set_entry_point(INTENT_DETECTION_NODE)

# Add conditional edges from intent detection to appropriate handlers
workflow.add_conditional_edges(
    INTENT_DETECTION_NODE,
    route_to_node,
    {
        "faq": "faq",
        "history": "history",
        "human_in_the_loop": "human_in_the_loop",
        "fallback": "fallback"
    }
)

# Add finish edges (all handler nodes go to END)
workflow.add_edge("faq", "__end__")
workflow.add_edge("history", "__end__")
workflow.add_edge("human_in_the_loop", "__end__")
workflow.add_edge("fallback", "__end__")

# Compile the graph
app = workflow.compile()

logger.info("Customer support graph compiled successfully")

# Step 15 Build Streamlit UI for Customer Support App

st.set_page_config(page_title="🎧 Support Agent", page_icon="💬", layout="centered")
st.title("🎧 AI Customer Support Agent")
st.caption("Get instant help with FAQs, account issues, and personalized support based on your history.")

# Initialize session state for messages, user_id, and thread_id
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_001"  # Default user ID (can be customized)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "thread_001"  # Default thread ID (can be customized)

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_input := st.chat_input("Type your message..."):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            # Initialize state for the agent
            state = {
                "user_id": st.session_state.user_id,
                "thread_id": st.session_state.thread_id,
                "user_input": user_input,
                "intent": None,
                "data": None,
                "short_term_memory": [],
                "long_term_memory": {},
                "hitl_flag": False
            }
            
            # Invoke the compiled graph
            final_state = asyncio.run(app.ainvoke(state))
            bot_reply = final_state['data']['response']
            
            # Display the assistant's reply
            st.markdown(bot_reply)
    
    # Append assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
