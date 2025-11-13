# import os
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage
# from typing import Optional

# # --- Configuration (you should replace these with your actual model names) ---
# # NOTE: The models you use must be available via the Gemini API and compatible 
# # with ChatGoogleGenerativeAI. 'gemini-2.5-flash' is a good, fast choice.
# model3_name = "gemini-2.5-flash"  
# model4_name = "gemini-2.5-pro"    
# # ---------------------------------------------------------------------------

# def _init_gemini_llm(model_type: int, user_api_key: Optional[str] = None):
#     """Initializes the Gemini LLM client."""
    
#     # ⚠️ IMPORTANT: Replace 'api_key' with your actual variable or mechanism 
#     # for getting the default API key if user_api_key is None.
#     # For simplicity, we'll try to get it from an environment variable first.
#     default_api_key = "AIzaSyDZFbYEY21RsziTT1Tvn9WkhjLe0rWh_eA"
#     api_key = default_api_key # This line assumes 'api_key' was meant to be the default key variable
    
#     final_api_key = api_key if user_api_key is None else user_api_key
    
#     # --- Model Selection ---
#     if model_type == 4:
#         model_name = model4_name
#     elif model_type == 3:
#         model_name = model3_name
#     else:
#         raise ValueError("model_type must be 3 or 4.")
    
#     llm = ChatGoogleGenerativeAI(
#         model=model_name, 
#         google_api_key=final_api_key, 
#         temperature=0.0
#     )
#     return llm

# def check_llm_response(model_type: int, user_api_key: Optional[str] = None):
#     """Initializes the LLM and sends a simple prompt to check for a response."""
    
#     llm = None
#     try:
#         # 1. Initialize the LLM client
#         llm = _init_gemini_llm(model_type, user_api_key)
        
#         # 2. Define a simple test prompt
#         test_message = "What is the capital of France? Respond in a single word."
        
#         # In LangChain, chat models take a list of message objects
#         messages = [HumanMessage(content=test_message)]
        
#         print(f"--- Testing LLM Model: {llm.model} ---")
#         print(f"Prompt: '{test_message}'")
        
#         # 3. Send the prompt using the .invoke() method
#         # This is a synchronous call that waits for the response.
#         response = llm.invoke(messages)
        
#         # 4. Check and print the response
#         if response and response.content:
#             print("\n✅ LLM Connection SUCCESS!")
#             print(f"Response Content: {response.content.strip()}")
#             print(f"Model Used: {llm.model}")
#             return True, response.content.strip()
#         else:
#             print("❌ LLM Connection FAILURE: Received an empty response.")
#             return False, None
            
#     except Exception as e:
#         print(f"\n❌ LLM Connection FAILURE: An error occurred.")
#         print(f"Error: {e}")
#         # Hint for common error: check your API key environment variable (GOOGLE_API_KEY)
#         return False, str(e)

# if __name__ == "__main__":
#     # --- USAGE ---
    
#     # NOTE: Ensure your GOOGLE_API_KEY is set as an environment variable, 
#     # or pass your key directly to the function (not recommended for production).
    
#     # Test with model type 3 (gemini-2.5-flash)
#     success, result = check_llm_response(model_type=3)
    
#     # Test with model type 4 (gemini-2.5-pro) - useful if you want to test both
#     # success_pro, result_pro = check_llm_response(model_type=4)












import os
import json
import re
import yaml

from typing import Optional, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate

# Load environment variables from a .env file (optional, but good practice)

# --- 1. MOCK CONFIGURATION (You need to replace this with your actual config) ---
# The LLM is instructed to return a JSON object with a 'target' key.


# --- 2. LLM Initialization Function (Using the one you provided) ---
config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
# Define model names used in your original logic
model3_name = "gemini-2.5-flash"  
model4_name = "gemini-2.5-pro"    

def _init_gemini_llm(model_type: int, user_api_key: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Initializes the Gemini LLM client."""
    
    # Prioritize user_api_key, then environment variable.
    final_api_key = "AIzaSyDZFbYEY21RsziTT1Tvn9WkhjLe0rWh_eA"
    
    if not final_api_key:
        raise ValueError(
            "API key not found. Please set the GOOGLE_API_KEY environment variable "
            "or pass user_api_key to the function."
        )

    if model_type == 4:
        model_name = model4_name
    elif model_type == 3:
        model_name = model3_name
    else:
        # Default to the faster model if an invalid type is passed
        print(f"Warning: Invalid model_type {model_type}. Defaulting to {model4_name}.")
        model_name = model4_name 
    
    llm = ChatGoogleGenerativeAI(
        model=model_name, 
        google_api_key=final_api_key, 
        temperature=0.0
    )
    return llm

# --- 3. YOUR ORIGINAL FUNCTION ---

def decide_target_attribute(attributes: List[str], types_info: Dict[str, str], head_info: str, model_type: int = 4, user_api_key: Optional[str] = None) -> str:
    """
    Determines the target attribute for modeling based on dataset attributes and characteristics.
    """
    try:
        print(f"--- Running decide_target_attribute (Model: {model4_name if model_type==4 else model3_name}) ---")
        
        # 1. Initialize the LLM
        llm = _init_gemini_llm(model_type, user_api_key)
        
        # 2. Format the prompt
        template = config["decide_target_attribute_template"]
        prompt_template = PromptTemplate(input_variables=["attributes", "types_info", "head_info"], template=template)
        summary_prompt = prompt_template.format(attributes=attributes, types_info=types_info, head_info=head_info)

        print(f"Sending prompt to LLM...")

        # 3. Invoke the LLM
        # LangChain's Chat models expect a list of BaseMessage objects (like HumanMessage)
        llm_answer = llm.invoke([HumanMessage(content=summary_prompt)])
        
        # 4. Extract JSON from the response content
        response_content = llm_answer.content
        
        if '```json' in response_content:
            # Use regex to safely extract the JSON content enclosed in triple backticks
            match = re.search(r'```json\n(.*?)```', response_content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                # Fallback if the JSON structure is slightly off
                json_str = response_content
        else:
            # Assume the entire content is the JSON string (based on the prompt)
            json_str = response_content

        # 5. Parse the JSON and return the 'target'
        result = json.loads(json_str)
        
        print("\n✅ LLM Response Received and Parsed.")
        print(f"Full LLM Output: \n{response_content[:150]}...")
        
        return result["target"]
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to get or parse target attribute.")
        # Re-raise the exception after printing for better debugging
        raise Exception(f"Error in decide_target_attribute: {e}")

# --- 4. SAMPLE DATA FOR TESTING ---

sample_attributes = [
    "CustomerID", 
    "Age", 
    "AnnualIncome", 
    "SpendingScore_1_100", 
    "Gender", 
    "SubscriptionStatus"
]

sample_types_info = {
    "CustomerID": "Integer/ID", 
    "Age": "Integer/Numerical", 
    "AnnualIncome": "Float/Numerical", 
    "SpendingScore_1_100": "Integer/Numerical", 
    "Gender": "String/Categorical", 
    "SubscriptionStatus": "Boolean/Target"
}

sample_head_info = """
   CustomerID  Age  AnnualIncome  SpendingScore_1_100  Gender SubscriptionStatus
0         1   19         15.00                   39  Female                 True
1         2   21         15.00                   81    Male                 True
2         3   20         16.00                    6  Female                False
3         4   23         16.00                   77  Female                 True
4         5   31         17.00                   40  Female                False
"""

# --- 5. EXECUTION ---
if __name__ == "__main__":
    
    # You can pass your key here if you don't want to use an environment variable:
    # my_api_key = "YOUR_API_KEY_HERE" 
    my_api_key = None 

    try:
        target_attribute = decide_target_attribute(
            attributes=sample_attributes,
            types_info=sample_types_info,
            head_info=sample_head_info,
            model_type=3, # Use the faster gemini-2.5-flash for a quick test
            user_api_key=my_api_key
        )
        print(f"\n=======================================================")
        print(f"🎉 Recommended Target Attribute: **{target_attribute}**")
        print(f"=======================================================")

    except Exception as e:
        print(f"Test failed: {e}")
        print("\nEnsure your GOOGLE_API_KEY is correctly set and the necessary libraries are installed.")