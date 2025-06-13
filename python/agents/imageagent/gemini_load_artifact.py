# -*- coding: utf-8 -*-
"""
Defines an ADK Tool that loads an image from an Artifact and uses a
multimodal model to judge its quality.
"""
import asyncio

# ADK and GenAI libraries
import google.generativeai as genai
import google.genai.types as types
from google.adk.tools import FunctionTool, ToolContext

# Assume the previous tool 'generate_and_save_image' exists and is available.
# from adk_imagen_tool import generate_and_save_image, imagen_artifact_tool

# --- Configuration ---
# In a real ADK application, the API key would be managed securely.
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

async def judge_image_quality(
    context: ToolContext,
    artifact_filename: str,
    criteria: str = "technical quality, composition, and aesthetics",
) -> str:
    """
    An ADK Tool that loads an image from an artifact and judges its quality.

    Args:
        context: The ToolContext provided by the ADK Runner, used to load artifacts.
        artifact_filename: The unique name of the image artifact to load.
        criteria: The criteria on which to judge the image.

    Returns:
        A string containing the model's quality assessment.
    """
    print(f"Executing tool 'judge_image_quality' for artifact: '{artifact_filename}'")
    try:
        # 1. --- Load the Artifact using the ToolContext ---
        # This is the core method for retrieving artifact data.
        print(f"Loading artifact '{artifact_filename}'...")
        image_artifact = await context.load_artifact(filename=artifact_filename)

        if not image_artifact or not image_artifact.inline_data:
            not_found_msg = f"Error: Artifact '{artifact_filename}' not found or is empty."
            print(not_found_msg)
            return not_found_msg

        print(f"Successfully loaded artifact. MIME Type: {image_artifact.inline_data.mime_type}")

        # 2. --- Use a Multimodal LLM to Judge the Image ---
        # Configure the Gemini client
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro-vision')

        # The prompt includes the image artifact part and a text instruction.
        prompt_parts = [
            f"Please judge this image based on the following criteria: {criteria}. "
            "Provide a concise, one-paragraph assessment.",
            image_artifact, # Pass the loaded artifact Part directly
        ]

        print("Sending image and prompt to Gemini for analysis...")
        response = await model.generate_content_async(prompt_parts)
        print("Received analysis from Gemini.")

        return response.text

    except ValueError as ve:
        # This can be raised if the artifact service isn't configured in the Runner.
        error_message = f"Error loading artifact: {ve}. Is the ArtifactService configured correctly?"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        return error_message

# --- Tool Definition ---
image_judging_tool = FunctionTool(
    func=judge_image_quality,
    description="Loads a saved image artifact and uses an LLM to provide a quality assessment."
)

