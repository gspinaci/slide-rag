import time
import gradio as gr
from search_slides import SlideSearcher

# Initialize the searcher once at module level for better performance
searcher = SlideSearcher(
    collection_name="slide_chunks",
    host="localhost",
    port=8000,
    use_llm=True,
    llm_model="gemini-2.5-flash-lite"
)

def chat_with_slides(message, history):
    """
    Chat function that connects to the slide search system.
    
    Args:
        message (str): User message
        history (list): Chat history
        
    Yields:
        str: Streaming response
    """
    if not message.strip():
        return
    
    try:
        # Generate answer using the pre-initialized searcher
        answer = searcher.generate_answer(message, k=5)
        
        # Stream the response character by character for the typing effect
        response = f"{answer}"
        for i in range(len(response)):
            time.sleep(0.005)
            yield response[: i + 1]
            
    except Exception as e:
        error_response = f"Error: {str(e)}"
        for i in range(len(error_response)):
            time.sleep(0.05)
            yield error_response[: i + 1]

demo = gr.ChatInterface(
    chat_with_slides,
    type="messages",
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    save_history=True,
)

if __name__ == "__main__":
    demo.launch()