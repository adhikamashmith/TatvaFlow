import os
import yaml
import json
import re
import streamlit as st
from typing import Optional, List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
 
config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
# model4_name = config["model4_name"]
# model3_name = config["model3_name"]
# api_key = config["openai_api_key"]
model3_name = "llm model"  
# model4_name = ""    
model4_name = "llm model"    
# api_key = ""
api_key = "useapikey"

def _init_gemini_llm(model_type,user_api_key):
    """Initializes the Gemini LLM client."""

    print("init gemini")
    # Prioritize user_api_key, then environment variable.
    final_api_key = "AIzaSyBykfqy8DtlQa9I9OOqu2Kn6sUQr3mFTgM"
    
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
#-----------------------------------------------
def decide_encode_type(attributes, data_frame_head, model_type = 4, user_api_key = None):
    """
    Decides the encoding type for given attributes using a language model via the OpenAI API.

    Parameters:
    - attributes (list): A list of attributes for which to decide the encoding type.
    - data_frame_head (DataFrame): The head of the DataFrame containing the attributes. This parameter is expected to be a representation of the DataFrame (e.g., a string or a small subset of the actual DataFrame) that gives an overview of the data.
    - model_type (int, optional): Specifies the model to use. The default model_type=4 corresponds to a predefined model named `model4_name`. Another option is model_type=3, which corresponds to `model3_name`.
    - user_api_key (str, optional): The user's OpenAI API key. If not provided, a default API key `api_key` is used.

    Returns:
    - A JSON object containing the recommended encoding types for the given attributes. Please refer to prompt templates in config.py for details.

    Raises:
    - Exception: If there is an issue accessing the OpenAI API, such as an invalid API key or a network connection error, the function will raise an exception with a message indicating the problem.
    """
    try:
        print("decide_encode_type")
        llm = _init_gemini_llm(model_type,user_api_key)
        template = config["numeric_attribute_template"]
        prompt_template = PromptTemplate(input_variables=["attributes", "data_frame_head"], template=template)
        summary_prompt = prompt_template.format(attributes=attributes, data_frame_head=data_frame_head)
        
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
        return json.loads(json_str)
    except Exception as e:
        st.error("Cannot access the Gemini API. Please check your API key or network connection.")
        st.stop()

def decide_fill_null(attributes, types_info, description_info, model_type = 4, user_api_key = None):
    """
    Decides the best encoding type for given attributes using an AI model via OpenAI API.

    Parameters:
    - attributes (list): List of attribute names to consider for encoding.
    - data_frame_head (DataFrame or str): The head of the DataFrame or a string representation, providing context for the encoding decision.
    - model_type (int, optional): The model to use, where 4 is the default. Can be customized to use a different model.
    - user_api_key (str, optional): The user's OpenAI API key. If None, a default key is used.

    Returns:
    - dict: A JSON object with recommended encoding types for the attributes. Please refer to prompt templates in config.py for details.

    Raises:
    - Exception: If there is an issue accessing the OpenAI API, such as an invalid API key or a network connection error, the function will raise an exception with a message indicating the problem.
    """
    try:
        print("decide_fill_null")
        llm = _init_gemini_llm(model_type,user_api_key)
        template = config["null_attribute_template"]
        prompt_template = PromptTemplate(input_variables=["attributes", "types_info", "description_info"], template=template)
        summary_prompt = prompt_template.format(attributes=attributes, types_info=types_info, description_info=description_info)
        
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
        return json.loads(json_str)
    except Exception as e:
        st.error("Cannot access the Gemini API. Please check your API key or network connection.")
        st.stop()

def decide_model(shape_info, head_info, nunique_info, description_info, model_type = 4, user_api_key = None):
    """
    Decides the most suitable machine learning model based on dataset characteristics.

    Parameters:
    - shape_info (dict): Information about the shape of the dataset.
    - head_info (str or DataFrame): The head of the dataset or its string representation.
    - nunique_info (dict): Information about the uniqueness of dataset attributes.
    - description_info (str): Descriptive information about the dataset.
    - model_type (int, optional): Specifies which model to consult for decision-making.
    - user_api_key (str, optional): OpenAI API key for making requests.

    Returns:
    - dict: A JSON object containing the recommended model and configuration. Please refer to prompt templates in config.py for details.

    Raises:
    - Exception: If there is an issue accessing the OpenAI API, such as an invalid API key or a network connection error, the function will raise an exception with a message indicating the problem.
    """
    try:
        llm = _init_gemini_llm(model_type,user_api_key)
        template = config["decide_model_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info", "head_info", "nunique_info", "description_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, head_info=head_info, nunique_info=nunique_info, description_info=description_info)

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
        return json.loads(json_str)
    except Exception as e:
        st.error("Cannot access the Gemini API. Please check your API key or network connection.")
        st.stop()

def decide_cluster_model(shape_info, description_info, cluster_info, model_type = 4, user_api_key = None):
    """
    Determines the appropriate clustering model based on dataset characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - cluster_info: Additional information relevant to clustering.
    - model_type (int, optional): The model type to use for decision making (default 4).
    - user_api_key (str, optional): The user's API key for OpenAI.

    Returns:
    - A JSON object with the recommended clustering model and parameters. Please refer to prompt templates in config.py for details.

    Raises:
    - Exception: If unable to access the OpenAI API or another error occurs.
    """
    try:
        llm = _init_gemini_llm(model_type,user_api_key)
        template = config["decide_clustering_model_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info", "description_info", "cluster_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, description_info=description_info, cluster_info=cluster_info)

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
        return json.loads(json_str)
    except Exception as e:
        st.error("Cannot access the Gemini API. Please check your API key or network connection.")
        st.stop()

def decide_regression_model(shape_info, description_info, Y_name, model_type = 4, user_api_key = None):
    """
    Determines the appropriate regression model based on dataset characteristics and the target variable.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - Y_name: The name of the target variable.
    - model_type (int, optional): The model type to use for decision making (default 4).
    - user_api_key (str, optional): The user's API key for OpenAI.

    Returns:
    - A JSON object with the recommended regression model and parameters. Please refer to prompt templates in config.py for details.

    Raises:
    - Exception: If unable to access the OpenAI API or another error occurs.
    """
    try:
        llm = _init_gemini_llm(model_type,user_api_key)
        template = config["decide_regression_model_template"]

        prompt_template = PromptTemplate(input_variables=["shape_info", "description_info", "Y_name"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, description_info=description_info, Y_name=Y_name)
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
        return json.loads(json_str)

    except Exception as e:
        st.error("Cannot access the Gemini API. Please check your API key or network connection.")
        st.stop()

# def decide_target_attribute(attributes, types_info, head_info, model_type = 4, user_api_key = None):
def decide_target_attribute(attributes: List[str], types_info: Dict[str, str], head_info: str, model_type: int = 4, user_api_key: Optional[str] = None) -> str:

    """
    Determines the target attribute for modeling based on dataset attributes and characteristics.

    Parameters:
    - attributes: A list of dataset attributes.
    - types_info: Information about the data types of the attributes.
    - head_info: A snapshot of the dataset's first few rows.
    - model_type (int, optional): The model type to use for decision making (default 4).
    - user_api_key (str, optional): The user's API key for OpenAI.

    Returns:
    - The name of the recommended target attribute. Please refer to prompt templates in config.py for details.

    Raises:
    - Exception: If unable to access the OpenAI API or another error occurs.
    """
    try:
        print("decide_target_attribute")
        print(attributes)
        print(types_info)
        print(head_info)
        print(model_type)
        print(user_api_key)


        llm = _init_gemini_llm(model_type,user_api_key)
        print("llm returned")
        template = config["decide_target_attribute_template"]
        print(template)
        prompt_template = PromptTemplate(input_variables=["attributes", "types_info", "head_info"], template=template)
        summary_prompt = prompt_template.format(attributes=attributes, types_info=types_info, head_info=head_info)
        print(f"Sending prompt to LLM...")
        llm_answer = llm.invoke([HumanMessage(content=summary_prompt)])
        response_content = llm_answer.content
        if '```json' in response_content:
            match = re.search(r'```json\n(.*?)```', response_content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = response_content
        else:
            json_str = response_content
        result = json.loads(json_str)
        
        print("\n✅ LLM Response Received and Parsed.")
        print(f"Full LLM Output: \n{response_content[:150]}...")
        
        return result["target"]
    
    except Exception as e:
        st.error("Cannot access the Gemini API. Please check your API key or network connection.")
        st.stop()

def decide_test_ratio(shape_info, model_type = 4, user_api_key = None):
    """
    Determines the appropriate train-test split ratio based on dataset characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.
    - model_type (int, optional): The model type to use for decision making (default 4).
    - user_api_key (str, optional): The user's API key for OpenAI.

    Returns:
    - The recommended train-test split ratio as a float. Please refer to prompt templates in config.py for details.

    Raises:
    - Exception: If unable to access the OpenAI API or another error occurs.
    """
    try:
        llm = _init_gemini_llm(model_type,user_api_key)
        template = config["decide_test_ratio_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info)
        print(f"Sending prompt to LLM...")
        llm_answer = llm.invoke([HumanMessage(content=summary_prompt)])
        response_content = llm_answer.content
        if '```json' in response_content:
            match = re.search(r'```json\n(.*?)```', response_content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = response_content
        else:
            json_str = response_content
        return json.loads(json_str)["test_ratio"]
    except Exception as e:
        st.error("Cannot access the Gemini API. Please check your API key or network connection.")
        st.stop()

def decide_balance(shape_info, description_info, balance_info, model_type = 4, user_api_key = None):
    """
    Determines the appropriate method to balance the dataset based on its characteristics.

    Parameters:
    - shape_info: Information about the dataset shape.
    - description_info: Descriptive statistics or information about the dataset.
    - balance_info: Additional information relevant to dataset balancing.
    - model_type (int, optional): The model type to use for decision making (default 4).
    - user_api_key (str, optional): The user's API key for OpenAI.

    Returns:
    - The recommended method to balance the dataset. Please refer to prompt templates in config.py for details.

    Raises:
    - Exception: If unable to access the OpenAI API or another error occurs.
    """
    try:
        llm = _init_gemini_llm(model_type,user_api_key)
        template = config["decide_balance_template"]
        prompt_template = PromptTemplate(input_variables=["shape_info", "description_info", "balance_info"], template=template)
        summary_prompt = prompt_template.format(shape_info=shape_info, description_info=description_info, balance_info=balance_info)
        print(f"Sending prompt to LLM...")
        llm_answer = llm.invoke([HumanMessage(content=summary_prompt)])
        response_content = llm_answer.content
        if '```json' in response_content:
            match = re.search(r'```json\n(.*?)```', response_content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = response_content
        else:
            json_str = response_content
        return json.loads(json_str)["method"]
    except Exception as e:
        st.error("Cannot access the Gemini API. Please check your API key or network connection.")
        st.stop()
