
import asyncio
import pprint
import time # Note: Your example used time.sleep, but the original script is async. Sticking to asyncio.sleep.
             # If this script is not run in an async context, time.sleep would be appropriate.

from google import genai
from google.genai import types
import requests

from dotenv import load_dotenv

import os
import uuid

image_gcs_uri="gs://ml-demo-rw/imagenagent/529a8373aa644619b3e634784fe66ebc/1748902438838/sample_0.png"
image_mime_type="image-png"

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

'''



def gcs_to_inline (
        image_gcs_uri: str,
        image_mime_type: str,

):
    """tool description"""

    image_path=image_gcs_uri
    image_bytes = requests.get(image_path).content
    image = types.Part.from_bytes(
        data=image_bytes, mime_type=image_mime_type
)
    print(image)
    return image   

gcs_to_inline(image_gcs_uri,image_mime_type)

    
    '''