import logging
import time
from datetime import datetime
# import google.generativeai as google_ai  # Not needed for hot path
from concurrent.futures import ThreadPoolExecutor, as_completed
# from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM  # Not needed for hot path
# import torch  # Not needed for hot path
import json
# import gc  # Not needed for hot path
import os
from tqdm.auto import tqdm
from openai import OpenAI
# import anthropic  # Not needed for hot path
# from together import Together  # Not needed for hot path
from dotenv import load_dotenv

load_dotenv()

# Initialize API clients with environment variables
OAI_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# ANT_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))  # Not needed for hot path
# TOGETHER_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))  # Not needed for hot path

MODEL_TO_COST = {"gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
                    "gpt-4o": {"input": 0.0000025, "output": 0.00001},
                    "gpt-4": {"input": 0.000030, "output": 0.000060},
                    "gpt-5-mini": {"input": 0.00000025, "output": 0.000002},  # placeholder pricing
                    "o1-mini": {"input": 0.000003, "output": 0.000012},
                    "o1": {"input": 0.000015, "output": 0.00006},
                    "o3-mini": {"input": 0.0000011, "output": 0.0000044},
                    "claude-3-7-sonnet-20250219": {"input": 0.000003, "output": 0.000015},
                    "claude-3-5-sonnet-20241022": {"input": 0.000003, "output": 0.000015},
                    "claude-3-5-haiku-20241022": {"input": 0.0000008, "output": 0.000004},
                    }


# google_ai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))  # Not needed for hot path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_MODEL_PATH_MAP = {
    "meta-llama/Llama-2-70b-chat-hf": "/vast/work/public/ml-datasets/llama-2/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "/vast/work/public/ml-datasets/llama-3.1/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "/vast/work/public/ml-datasets/llama-3.1/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct": "/vast/work/public/ml-datasets/llama-3.1/Meta-Llama-3.1-405B-Instruct"
}

def GET_API_COST(model, response):
    # Handle fine-tuned models by extracting the base model name
    # Fine-tuned models have format: ft:base-model:org:name:id
    if model.startswith("ft:"):
        # Extract base model (e.g., "gpt-4o-mini-2024-07-18" from fine-tuned name)
        # For fine-tuned gpt-4o-mini, use gpt-4o-mini pricing
        if "gpt-4o-mini" in model:
            base_model = "gpt-4o-mini"
        elif "gpt-4o" in model:
            base_model = "gpt-4o"
        elif "gpt-4" in model:
            base_model = "gpt-4"
        else:
            base_model = "gpt-4o-mini"  # default fallback
    else:
        base_model = model
    
    if 'gpt' in model or 'o1' in model or 'o3' in model:
        return (
            response.usage.prompt_tokens * MODEL_TO_COST[base_model]["input"]
            + response.usage.completion_tokens * MODEL_TO_COST[base_model]["output"]
        )
    elif 'claude' in model:
        return (
            response.usage.input_tokens * MODEL_TO_COST[base_model]["input"]
            + response.usage.output_tokens * MODEL_TO_COST[base_model]["output"]
        )

def generate_OAI_response(
    prompt, system_prompt="", model_name="gpt-4o-mini", cost=0, max_tokens=500, temperature=0
):
    """Make one OAI inference with error handling, retry logic, 
    and graceful handling of invalid prompts by checking the exception message."""
    while True:
        try:
            if model_name in ['o1', 'o1-mini', "o3-mini", "gpt-5-mini"]:
                response = OAI_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
            else:
                response = OAI_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant." + " " + system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            # Calculate and update the cost of the API call
            openai_cost = GET_API_COST(model_name, response)
            cost += openai_cost

            # Extract and return the response content
            response_content = response.choices[0].message.content
            return response_content, cost

        except Exception as e:
            # Convert the exception to a string so we can inspect its message
            error_str = str(e)
            print(error_str)
            # Check if the error message contains "invalid_prompt"
            if "violating" in error_str:
                return "Sorry, I can't help with this.", cost

            # Otherwise, print the error and retry after 10 seconds
            print(f"API error occurred: {e}. Retrying in 10 seconds...")
            time.sleep(10)


def document_api_cost(total_cost):

    curr_loc = os.getcwd()

    # Find the most recent folder containing 'SafeSysGen'
    while 'SafeSysGen' not in os.path.basename(curr_loc) and os.path.dirname(curr_loc) != curr_loc:
        curr_loc = os.path.dirname(curr_loc)

    if 'SafeSysGen' not in os.path.basename(curr_loc):
        raise FileNotFoundError("SafeSysGen folder not found in the current directory structure.")

    file_path = os.path.join(curr_loc, "oai_costs.json")
    print(file_path)
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # Check if the file exists, if not create an empty file with an empty dictionary
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            json.dump({}, file)
        print(f"Created {file_path} to document API cost.")
    
    # Read and handle the file content safely
    try:
        with open(file_path, "r") as file:
            content = file.read().strip()  # Read and strip whitespace
            data = json.loads(content) if content else {}  # Load or initialize
    except (json.JSONDecodeError, FileNotFoundError):
        data = {}  # Reset to an empty dictionary if the file is invalid
    
    # Add the new entry
    data[datetime.now().strftime("%Y-%m-%d %H:%M:%S")] = total_cost

    # Write back the updated data
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def generate_OAI_responses_threaded(
    prompts, system_prompt="", model_name="gpt-4o-mini", max_tokens=300, temperature=0, max_workers=20
):
    """Generate responses using multithreading with a specified number of workers and progress bar.

    Returns:
        Tuple of (results dict, total_cost float)
    """
    results = {}
    total_prompts = len(prompts)
    total_cost = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prompt = {
            executor.submit(
                generate_OAI_response, prompt, system_prompt, model_name, 0, max_tokens, temperature
            ): prompt
            for prompt in prompts
        }

        with tqdm(total=total_prompts, desc="Generating responses") as pbar:
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results[prompt], cost = result
                    total_cost += cost
                except Exception as e:
                    print(f"Error processing prompt: {prompt}")
                    print(f"Error: {e}")
                finally:
                    pbar.update(1)

    return results, total_cost

def make_mutithreaded_inference_by_model(prompts, model_name, system_prompt="You are a helpful assistant.", max_tokens=2000, temperature=0, max_workers=30):
    """Run multi-threaded inference. Currently only supports OpenAI models (gpt-*, o1-*, o3-*).

    Returns:
        Tuple of (responses dict, total_cost float)
    """
    if 'gpt' in model_name or 'o1' in model_name or 'o3' in model_name:
        responses, cost = generate_OAI_responses_threaded(
            prompts, system_prompt=system_prompt, model_name=model_name, max_tokens=max_tokens, temperature=0, max_workers=max_workers
        )
    else:
        raise ValueError(f"Model {model_name} not supported. Only OpenAI models (gpt-*, o1-*, o3-*) are supported in this version.")
    return responses, cost


# ============================================================================
# The following functions are commented out as they are not needed for the
# hot path (OpenAI/gpt-5-mini). Uncomment and add the required imports if needed.
# ============================================================================

# def generate_TOGETHER_response(...)
# def generate_TOGETHER_responses_threaded(...)
# def generate_ANT_response(...)
# def generate_ANT_responses_threaded(...)
# def get_response_with_retry(...)
# def get_response_from_google_model(...)
# def generate_gemini_responses_threaded(...)
# def generate_gemini_responses_threaded_modified(...)
# def free_memory(...)
# def load_hf_model_and_tokenizer(...)
# def get_response_from_hf_model(...)
# def load_model_and_tokenizer(...)
# def batch_HF_inference(...)