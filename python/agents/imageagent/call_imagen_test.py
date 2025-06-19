# version with vertexai endpoint
#maybe try REST instead

#        aspect_ratio (str): Optional. The aspect ratio of the generated image (e.g., '1:1', '16:9', '9:16'). Defaults to '1:1'.
#  Returns:
     #   str: The GCS URI of the generated image, or None if generation failed.

#conda install google-adk
#conda install google-genai

import os
import uuid
from google.genai import types as types
from dotenv import load_dotenv
import logging
import asyncio
import base64
from google import adk
import requests
import subprocess
import json

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
#import google.genai as types
from google.adk.tools import FunctionTool, ToolContext

async def call_imagen_tool(
    tool_context: ToolContext,
    prompt: str,
    artifact_filename: str,
    version: str,
    aspect_ratio: str
    ):
    """Tool to generate an image from a prompt using Imagen, and save it as an artifact.
    
    Args:
        tool_context: The ToolContext provided by the ADK Runner, used to save artifacts.
        prompt (str): The text prompt to be sent to the Imagen model.
        artifact_filename (str): The unique name for the artifact (e.g., 'report.png').
        version (str): The version (e.g v1, v2) of the generated artifact.
        aspect_ratio (str): Optional. The aspect ratio of the generated image (e.g., '1:1', '16:9', '9:16'). Defaults to '1:1'.


    Returns:
        A string that is the path to the artifact file where the image has been saved, or none if image generation failed
    """
    try:

        load_dotenv()   

        PROJECT_ID="remy-sandbox"

#save to GCS but that's not used as of now
        #location="global"
        #gcs_bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET", "ml-demo-rw")
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
        full_image_filename = f"{version}:{artifact_filename}"
        print(f"Saving artifact with filename: '{full_image_filename}'")
        await tool_context.save_artifact(
            filename=full_image_filename,
            artifact=image_artifact
        )

        #save the prompt as an artifact
        full_prompt_filename = f"{version}.prompt_fulltext"
        await tool_context.save_artifact(
        filename=full_prompt_filename,
        artifact=types.Part(text=prompt),
        )
        
        result_message = f"Successfully saved image to artifact '{full_image_filename}' with prompt '{prompt}'."
        print(result_message)

        return result_message
        

    except Exception as e:
        logging.error(f"An error occurred during image generation: {e}", exc_info=True)
        # Depending on the desired error handling for the agent, you might want to
        # re-raise the exception or just return None.
        return None

'''   
# Wrap the Python function in a FunctionTool to make it available to an agent.
call_imagen = FunctionTool(
    func=image_generation_tool,
    description="Generates an image based on a text prompt and saves it as a file (artifact) for later use."
)
 '''   


    # --- Test Runner Code ---

async def main():
    """A simple example function to demonstrate running the tool."""
    
    # In a real ADK application, the Runner provides the context.
    # For this example, we'll create a simple mock object that prints what it's doing.
    class MockToolContext:
        async def save_artifact(self, filename: str, artifact: types.Part) -> int:
            """Mock function to simulate saving the artifact."""
            print(f"--- MOCK CONTEXT: Saving artifact '{filename}' ---")
            # For demonstration, we'll save the image to the local directory
            # so you can see the output.
            local_filename = filename.split(":")[-1]  # remove 'user:' prefix
            if artifact.text is not None:
                with open(local_filename, "w", encoding="utf-8") as f:
                    f.write(artifact.text)
                print(f"--- MOCK CONTEXT: Text artifact saved locally to '{local_filename}' ---")
            elif artifact.inline_data and artifact.inline_data.data is not None:
                with open(local_filename, "wb") as f:
                    f.write(artifact.inline_data.data)
                print(f"--- MOCK CONTEXT: Binary artifact (e.g., image) saved locally to '{local_filename}' ---")
            else:
                print(f"--- MOCK CONTEXT: Artifact '{filename}' has no text or inline_data to save. ---")
            return 0  # Return a mock version number

    # Set up the arguments for the tool call
    mock_context = MockToolContext()
    prompt = "A watercolor painting of a red panda sleeping on a cherry blossom tree"
    filename = "red_panda.png"
    version = "v1"
    aspect_ratio = "1:1"

    print("--- Starting Tool Execution Example ---")
    # Execute the tool
    result = await call_imagen_tool(
        tool_context=mock_context,
        prompt=prompt,
        artifact_filename=filename,
        version=version,
        aspect_ratio=aspect_ratio
    )
    print(f"--- Tool finished with result: {result} ---")


if __name__ == "__main__":
    # Configure logging to see the tool's output
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # To run this example, you need a .env file with your GCP details, like this:
    # GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
    # GOOGLE_CLOUD_BUCKET="your-gcs-bucket-name"
    #
    # You also need to be authenticated with GCP. Run `gcloud auth application-default login`.
    load_dotenv()
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        logging.error("ERROR: GOOGLE_CLOUD_PROJECT environment variable not set. Please create a .env file.")
    else:
        asyncio.run(main())



'''
        # Create the GenerateVideosConfig object
        generate_video_config = types.GenerateVideosConfig(
            duration_seconds=duration_seconds,
            number_of_videos=1,
            output_gcs_uri=output_gcs_uri,
            person_generation="allow_adult",
            enhance_prompt=True,
        )

        # Create an operation to generate a video
        operation =  client.models.generate_videos(
            model="veo-2.0-generate-001",
            prompt=prompt,
            config=generate_video_config,
        )
        # Wait for video generation to complete
        while not operation.done:
            await asyncio.sleep(5) # Polling interval (e.g., 15 seconds)
            operation = client.operations.get(operation)
        pprint.pprint(operation)
        
        return operation.response
        
        
    except Exception as e:
        # Catch any other general exceptions that might occur during the process
        return f"Error generating video: {str(e)}"
        #raise e









#Example of how you might register this tool (conceptual)


import asyncio
import random
from marshmallow import fields
import logging

from google.adk.tools import *

from google.cloud import aiplatform # For Vertex AI client
from vertexai.preview.vision_models import ImageGenerationModel, ImageGenerationResponse # Specific for Imagen
from google.api_core import exceptions as google_exceptions # For handling API errors

logger = logging.getLogger(__name__)

''' '''
class CallImagenInputSchema(ToolInputSchema):
    """Input schema for the CallImagenTool."""
    prompt = fields.String(
        required=True,
        description="The text prompt to send to Imagen for image generation."
    )
    negative_prompt = fields.String(
        required=False,
        description="Optional. A description of what you want to omit in the generated image."
    )
    aspect_ratio = fields.String(
        required=False,
        description="Optional. The aspect ratio of the generated image (e.g., '1:1', '16:9', '9:16'). Defaults to '1:1'."
    )
    # You can add more Imagen parameters here if needed, e.g., seed, guidance_scale

class CallImagenOutputSchema(ToolOutputSchema):
    """Output schema for the CallImagenTool."""
    image_uri = fields.String(
        required=True,
        description="The GCS URI of the generated image."
    )
    prompt_used = fields.String(
        required=True,
        description="The exact prompt that was submitted to Imagen."
    )
    model_used = fields.String(
        required=True,
        description="The Imagen model version used for generation."
    )
'''

'''

    def __init__(self, model_id: str = "imagen-3.0-generate-002", project: str = None, location: str = None, **kwargs):
        """
        Initializes the CallImagenTool.

        Args:
            model_id: The ID of the Imagen model to use (e.g., "imagegeneration@006" for Imagen 2,
                      or a specific ID for Imagen 3 if available).
            project: Your Google Cloud project ID. If None, attempts to infer from environment.
            location: Your Google Cloud region. If None, attempts to infer from environment.
        """
        super().__init__(**kwargs)
        self.model_id = model_id
        self.project = project
        self.location = location
        
        # Initialize Vertex AI. If project/location are not provided,
        # aiplatform.init() will try to infer them from the environment.
        # This is done once when the tool is instantiated.
        try:
            aiplatform.init(project=self.project, location=self.location)
            logger.info(
                f"Vertex AI initialized for project: {aiplatform.gca_config.project}, "
                f"location: {aiplatform.gca_config.location}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)
            # Depending on ADK's error handling, you might re-raise or handle this.
            # For now, we'll let it proceed, and _run will fail if initialization was crucial and failed.


    async def _run(
        self,
        prompt: str,
        negative_prompt: str = None,
        aspect_ratio: str = "1:1",
        **kwargs
    ) -> dict:
        """
        The core logic of the tool.
        Receives a prompt, calls Imagen on Vertex AI, and returns the image URI.
        """
        logger.info(f"Tool '{self.name}' invoked with prompt: '{prompt}'")
        logger.info(f"Using Imagen model ID: {self.model_id}")
        if negative_prompt:
            logger.info(f"Negative prompt: '{negative_prompt}'")
        logger.info(f"Aspect ratio: '{aspect_ratio}'")

        try:
            generation_model = ImageGenerationModel.from_pretrained(self.model_id)

            # The generate_images method might be synchronous.
            # To avoid blocking the asyncio event loop in ADK, run it in a separate thread.
            response: ImageGenerationResponse = await asyncio.to_thread(
                generation_model.generate_images,
                prompt=prompt,
                number_of_images=1,  # Agent instruction implies one image
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                #output_gcs_uri=
                # You can expose other parameters like seed, guidance_scale via input_schema
            )

            if not response.images:
                logger.error("Imagen API call successful but returned no images.")
                raise Exception("Imagen API did not return any images.")

            # The URI is typically available on the image object.
            # For images generated by the service and stored in GCS, this is common.
            # The exact attribute might vary slightly with SDK updates, verify with documentation.
            generated_image_uri = response.images[0].uri
            
            if not generated_image_uri:
                logger.error("Imagen API returned an image object without a URI.")
                raise Exception("Generated image has no URI.")

            logger.info(f"Imagen generated image URI: {generated_image_uri}")

            return {
                "image_uri": generated_image_uri,
                "prompt_used": prompt,
                "model_used": self.model_id
            }

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Imagen API call failed with GoogleAPIError: {e}", exc_info=True)
            # Provide a more user-friendly error message if possible
            error_message = f"Imagen API error: {e.message}" if hasattr(e, 'message') else str(e)
            raise Exception(error_message) from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during Imagen image generation: {e}", exc_info=True)
            raise Exception(f"An unexpected error occurred: {str(e)}") from e


if __name__ == '__main__':
    # This is for local testing of the tool, not for ADK deployment
    # You'd need to set up your GOOGLE_APPLICATION_CREDENTIALS or be in a GCP env
    # and provide your project and location.
    PROJECT_ID = "remy-sandbox"  # Replace with your Project ID
    LOCATION = "us-central1"      # Replace with your Region

    logging.basicConfig(level=logging.INFO)

    async def test_tool_locally():
        try:
            # Initialize Vertex AI for the test
            # aiplatform.init(project=PROJECT_ID, location=LOCATION) # Done in constructor now

            # Instantiate the tool
            # For Imagen 3, you might need a specific model_id.
            # "imagegeneration@006" is a stable Imagen 2 model.
            # Check Vertex AI documentation for the latest Imagen 3 model IDs.
            imagen_tool = CallImagenTool(model_id="imagegeneration@006", project=PROJECT_ID, location=LOCATION)

            # Test with a sample prompt
            result = await imagen_tool.run(
                prompt="A majestic snow leopard perched on a rocky outcrop, overlooking a vast mountain range at sunset, hyperrealistic, detailed fur.",
                negative_prompt="cartoon, blurry, text, watermark",
                aspect_ratio="16:9"
            )
            print("\nTool run successful:")
            print(result)

        except Exception as e:
            print(f"\nTool run failed: {e}")

    asyncio.run(test_tool_locally())



    ---- old pytho SDK imagen call

            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

        response = model.generate_images(
            prompt=prompt,
            # Optional parameters
            number_of_images=1,
            language="en",
            # You can't use a seed value and watermark at the same time.
            # add_watermark=False,
            # seed=100,
            # output_gcs_uri=output_gcs_uri,
            aspect_ratio=aspect_ratio,
            safety_filter_level="block_some",
            person_generation="allow_adult"
        )


    '''