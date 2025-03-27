import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun  # Updated import
from langchain.tools import tool
import yfinance as yf

# Streamlit page configuration
st.set_page_config(
    page_title="Financial Agent Chatbot",
    page_icon="💹",
    layout="wide"
)

# Add logo
LOGO_PATH = r"C:\Users\Asus\OneDrive\Desktop\New_berry\image001.png"  # Replace with your logo path
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image(LOGO_PATH, width=200)

st.title("Financial Agent Chatbot")
st.markdown("---")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

GEMINI_API_KEY = "AIzaSyCcUFY04YwiLbCdYFvjXzWg-ze0LOtYKmY" # Replace with your actual API key, or better, use environment variables

# Define tools for the agent
search_tool = DuckDuckGoSearchRun()

@tool
def get_stock_price(ticker_symbol: str) -> str:
    """Useful for getting the current stock price of a company.
    Input should be the stock ticker symbol (e.g., AAPL for Apple)."""
    try:
        data = yf.Ticker(ticker_symbol)
        todays_data = data.history(period='1d')
        if todays_data.empty:
            return "Could not retrieve stock price for {}".format(ticker_symbol)
        current_price = todays_data['Close'][0]
        return f"The current stock price of {ticker_symbol} is ${current_price:.2f}"
    except Exception as e:
        return f"Error retrieving stock price for {ticker_symbol}: {e}"

@tool
def get_analyst_recommendations(ticker_symbol: str) -> str:
    """Useful for getting analyst recommendations for a company.
    Input should be the stock ticker symbol (e.g., AAPL for Apple)."""
    try:
        data = yf.Ticker(ticker_symbol)
        recommendations_df = data.recommendations
        if recommendations_df.empty:
            return f"No analyst recommendations found for {ticker_symbol}"
        return f"Analyst recommendations for {ticker_symbol}:\n{recommendations_df.to_string()}"
    except Exception as e:
        return f"Error retrieving analyst recommendations for {ticker_symbol}: {e}"

@tool
def get_stock_fundamentals(ticker_symbol: str) -> str:
    """Useful for getting fundamental data for a company.
    Input should be the stock ticker symbol (e.g., AAPL for Apple)."""
    try:
        data = yf.Ticker(ticker_symbol)
        fundamental_info = data.info #  'info' contains a wide range of fundamentals
        if not fundamental_info:
            return f"No fundamental data found for {ticker_symbol}"
        # Format fundamental info into a readable string (can be improved)
        formatted_info = "\n".join([f"{key}: {value}" for key, value in fundamental_info.items()])
        return f"Fundamental data for {ticker_symbol}:\n{formatted_info}"
    except Exception as e:
        return f"Error retrieving fundamental data for {ticker_symbol}: {e}"

@tool
def get_company_news(ticker_symbol: str) -> str:
    """Useful for getting the latest news headlines about a company.
    Input should be the stock ticker symbol (e.g., AAPL for Apple)."""
    try:
        data = yf.Ticker(ticker_symbol)
        news_list = data.news
        if not news_list:
            return f"No news found for {ticker_symbol}"

        headlines = [] # Use a list to collect headlines
        for news_item in news_list:
            if 'title' in news_item:
                headlines.append(f"- {news_item['title']}") # Add title if available
            else:
                headlines.append("- (No title available)") # Handle missing title

        if not headlines: # Check if any headlines were collected (in case all items lacked titles)
            return f"Could not retrieve news headlines (titles missing in data) for {ticker_symbol}"

        return f"Latest news headlines for {ticker_symbol}:\n" + "\n".join(headlines)

    except Exception as e:
        return f"Error retrieving news for {ticker_symbol}: {e}"

def initialize_agent_and_tools():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-thinking-exp-01-21",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7
    )
    
    agent_tools = [
        search_tool,
        get_stock_price,
        get_analyst_recommendations,
        get_stock_fundamentals,
        get_company_news,
    ]
    
    return initialize_agent(
        agent="zero-shot-react-description",
        llm=llm,
        tools=agent_tools,
        verbose=True,
        handle_parsing_errors=True
    )

# Initialize agent
agent = initialize_agent_and_tools()

# Chat interface
st.markdown("### Chat Interface")
chat_container = st.container()

# Display chat history
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about stocks..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent.run(prompt)
                st.write(response)
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add a sidebar with information
with st.sidebar:
    st.markdown("### About")
    st.write("This chatbot can help you with:")
    st.write("- Getting stock prices")
    st.write("- Checking analyst recommendations")
    st.write("- Viewing company fundamentals")
    st.write("- Finding latest company news")
    
    st.markdown("### Example Questions")
    st.write("- What's the current price of AAPL stock?")
    st.write("- Show me the fundamentals for MSFT")
    st.write("- Get me the latest news about GOOGL")
    
    # Add clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()