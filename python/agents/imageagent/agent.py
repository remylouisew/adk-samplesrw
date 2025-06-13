#conda activate agents

import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from dotenv import load_dotenv
import base64
from google import adk
import requests
import subprocess
import json


import os
import uuid
from google import genai
from dotenv import load_dotenv
import logging
import asyncio
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent, SequentialAgent
from google.genai import types
from google.adk.runners import InMemoryRunner
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools import FunctionTool, ToolContext
from typing import AsyncGenerator, Optional
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService # Or GcsArtifactService
from google.adk.agents import LlmAgent # Any agent
from google.adk.sessions import InMemorySessionService

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# ---- Constants -----
APP_NAME = "image_agent" # New App Name
USER_ID = "dev_user_01"
SESSION_ID_BASE = "loop_exit_tool_session" # New Base Session ID
GEMINI_MODEL = "gemini-2.5-flash"

#STATE_INITIAL_PROMPT = "initial_prompt"


# --- State Keys ---
STATE_CURRENT = "current_session"
STATE_CRITICISM = "criticism"
STATE_CURRENT_IMAGE = "artifact_and_prompt"
# Define the exact phrase the Critic should use to signal completion
COMPLETION_PHRASE = "No major issues found."

'''
session_service = InMemorySessionService()
session = await session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=s,
    state={"initial_prompt": "null"}
)
print(f"Initial state: {session.state}")
'''
#instatiate artifact runner
artifact_service = InMemoryArtifactService() # Choose an implementation
session_service = InMemorySessionService()

runner = Runner(
    agent=initial_image_agent,
    app_name="my_artifact_app",
    session_service=session_service,
    artifact_service=artifact_service # Provide the service instance here
)


def exit_loop(tool_context: ToolContext):
  """Call this function ONLY when the critique indicates no further changes are needed, signaling the iterative process should end."""
  print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
  tool_context.actions.escalate = True
  # Return empty dict as tools should typically return JSON-serializable output
  return {}

async def call_imagen_tool(
    context: ToolContext,
    prompt: str,
    artifact_filename: str,
    aspect_ratio: str
    ):
    """Tool to generate an image from a prompt using Imagen, and save it as an artifact.
    
    Args:
        context: The ToolContext provided by the ADK Runner, used to save artifacts.
        prompt (str): The text prompt to be sent to the Imagen model.
        artifact_filename (str): The unique name for the artifact (e.g., 'report.png').
        aspect_ratio (str): Optional. The aspect ratio of the generated image (e.g., '1:1', '16:9', '9:16'). Defaults to '1:1'.


    Returns:
        A string that is the path to the artifact file where the image has been saved, or none if image generation failed
    """
    try:

        load_dotenv()   

        PROJECT_ID="remy-sandbox"
        location="global"
        gcs_bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET", "ml-demo-rw")
#savec to GCS but that's not used as of now
        #output_gcs_uri=f"gs://{gcs_bucket_name}/imageagent/"+ uuid.uuid4().hex +".png"


        # Initialize the client for the Generative AI APIs

        #vertexai.init(project=PROJECT_ID, location="us-central1")
        #model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

        print("Getting access token from gcloud...")
        token_process = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True
        )
        access_token = token_process.stdout.strip()
        print("Successfully retrieved access token.")

        # --- 2. Define the API endpoint and headers ---
        api_endpoint = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/imagen-3.0-generate-002:predict"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        # --- 3. Construct the JSON payload ---
        payload = {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": {
                "sampleCount": 1,
                "language": "en",
                "aspectRatio": aspect_ratio,
                "personGeneration": "allow_adult"
            }
        }

        # --- 4. Make the REST API call using the 'requests' library ---
        print(f"Sending request to Imagen API for prompt: '{prompt}'")
        response = requests.post(api_endpoint, headers=headers, json=payload)

        # Raise an exception if the API returned an error (e.g., 4xx or 5xx)
        response.raise_for_status()
        #response[0].save(location=output_gcs_uri, include_generation_parameters=False)
        #response.images[0].show()
#print(json.dumps(response, indent=2))

        logging.info("Imagen job successfully completed.")

        response = response.json()

#save inline image as file
        base64_image_data = response['predictions'][0]['bytesBase64Encoded']
        image_bytes = base64.b64decode(base64_image_data)
        print(f"Decoded image data ({len(image_bytes)} bytes).")

        #response[0].save(location=artifact_filename, include_generation_parameters=False)

        # 3. --- Create an ADK Artifact Part (as per documentation) ---
        # The data is wrapped in a google.genai.types.Part object.
        image_artifact = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        
        print(f"Created types.Part artifact with MIME type: {image_artifact.inline_data.mime_type}")

        # 4. --- Save the Artifact using the ToolContext ---
        # This is the core interaction with the ADK artifact system.
        print(f"Saving artifact with filename: '{artifact_filename}'")
        version = await context.save_artifact(
            filename=artifact_filename,
            artifact=image_artifact
        )
        
        result_message = f"Successfully saved image to artifact '{artifact_filename}' (version {version})."
        print(result_message)

        return artifact_filename
        

    except Exception as e:
        logging.error(f"An error occurred during image generation: {e}", exc_info=True)
        # Depending on the desired error handling for the agent, you might want to
        # re-raise the exception or just return None.
        return None
    
call_imagen = FunctionTool(
    func=call_imagen_tool
)

async def artifact_to_inline(
    context: ToolContext,
    artifact_filename: str
):
    """Tool to load an image artifact for an LLM agent to use.
    
    Args:
        context: The ToolContext provided by the ADK Runner, used to load artifacts.
        artifact_filename: The unique filename of the artifact, used to locate it. This will be given verbatim as an input, do not alter the given filename in any way.

    Returns:
       types.Part : The inline image file, 
    """
    try:
        print(f"Loading artifact '{artifact_filename}'...")
        image_artifact = await context.load_artifact(filename=artifact_filename)

        return image_artifact
    
    except Exception as e:
        logging.error(f"An error occurred during image loading: {e}", exc_info=True)
        # Depending on the desired error handling for the agent, you might want to
        # re-raise the exception or just return None.
        return None


# Configure logging
logger = logging.getLogger(__name__)

initial_image_agent = LlmAgent(
    name="InitialImageAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    
    instruction=f"""You are an expert prompt engineering assistant for Google's Imagen 3 text-to-image AI. You have two tasks:
   1. Ask the user what kind of image they would like to generate.
  2. Based on the user's input, determine an appropriate Imagen prompt, a unique artifact_filename (e.g., "my_image_description.png"), and an aspect_ratio.
  3. Call the `call_imagen` tool with the generated prompt, artifact_filename, and aspect_ratio.
  4. After the `call_imagen` tool returns the `artifact_filename` (which should be the same as the one you provided to the tool if successful):
     Your final output for this turn MUST be a string representing a Python list containing exactly two string elements:
     the `artifact_filename` returned by the tool, and the `prompt` you used for the `call_imagen` tool.
     Example of correct final output format: `["my_image_description.png", "the full prompt text used"]`
     Do NOT add any other text, conversation, or explanation before or after this list string.


""",
    description="Generates an imagen image based on user feedback to the prompt.",
    output_key=STATE_CURRENT_IMAGE,
    tools=[save_artifact, call_imagen]
)

#need to have the agent take the URI as input!
critic_agent_in_loop = LlmAgent(
    name="CriticAgent",
    model=GEMINI_MODEL,
    include_contents='none',
    # MODIFIED Instruction: More nuanced completion criteria, look for clear improvement paths.
    instruction=f""" You are a critic of AI generated images. You will look at an image that has been generated by AI and judge whether it adequately depicts what is described in the prompt.

    *** Current Image artifact filename and previous Prompt ***
    ```
    {{artifact_and_prompt}}
    ```

    **Task:**

    Load the image using the artifact_to_inline tool. Show it to the user.

    Review the image for adherence to the artistic style, composition, subject and color described in the previous prompt. Ask the user for their opinion on whether the image is acceptable or not.

    IF you identify 1-2 *clear and actionable* ways the image could be improved to better capture the prompt or enhance user satisfaction (e.g., "The cat needs to be the main subject", "The sky should be purple"):
    Provide these specific suggestions concisely. Output *only* the critique text.

    ELSE IF the image has the correct subject, composition, artistic qualities, and accurately depicts what is described in the prompt, AND the user is satisfied with the image:
    Respond *exactly* with the phrase "{COMPLETION_PHRASE}" and nothing else. It doesn't need to be perfect, just functionally complete for this stage. Avoid suggesting purely subjective stylistic preferences if the core is sound.

    Do not add explanations. Output only the critique OR the exact completion phrase.
""",
    description="Reviews the current image, providing critique if clear improvements are needed, otherwise signals completion.",
    output_key=STATE_CRITICISM
)

refiner_agent_in_loop = LlmAgent(
    name="RefinerAgent",
    model=GEMINI_MODEL,
    # Relies solely on state via placeholders
    include_contents='none',
    instruction=f"""You are an AI image generation Assistant refining an image prompt based on feedback OR exiting the process.
 *** Current Image artifact filename and previous Prompt ***
    ```
     {{artifact_and_prompt}}
    ```
    **Critique/Suggestions:**
    {{criticism}}

    **Task:**
    Analyze the 'Critique/Suggestions'.
    IF the critique is *exactly* "{COMPLETION_PHRASE}":
    You MUST call the 'exit_loop' function. Do not output any text.
    ELSE (the critique contains actionable feedback):
    Carefully apply the suggestions to improve the previous prompt. Create a better prompt and submit it to Imagen to generate the new image.

    Do not add explanations. Either output the refined image OR call the exit_loop function.
""",
    description="Refines the image based on critique, or calls exit_loop if critique indicates completion.",
    tools=[exit_loop], # Provide the exit_loop tool
    output_key=STATE_CURRENT_IMAGE # Overwrites state['current_document'] with the refined version
)

# STEP 2: Refinement Loop Agent
refinement_loop = LoopAgent(
    name="RefinementLoop",
    # Agent order is crucial: Critique first, then Refine/Exit
    sub_agents=[
        critic_agent_in_loop,
        refiner_agent_in_loop,
    ],
    max_iterations=5 # Limit loops
)

# STEP 3: Overall Sequential Pipeline
# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = SequentialAgent(
    name="IterativeWritingPipeline",
    sub_agents=[
        initial_image_agent, # Run first to create initial doc
        refinement_loop       # Then run the critique/refine loop
    ],
    description="Generates an initial image and then iteratively refines it with critique using an exit tool."
)


''' --- example agents 

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city."
    ),
    tools=[get_weather, get_current_time],
)

'''