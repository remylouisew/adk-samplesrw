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


# --- Example Usage Simulation ---
# This simulates an agent that first creates an image, then judges it.
class MockArtifactService:
    """A mock service to simulate artifact storage for testing."""
    def __init__(self):
        self._store = {}
    async def save(self, name, artifact):
        if name not in self._store:
            self._store[name] = []
        self._store[name].append(artifact)
        version = len(self._store[name]) - 1
        return version
    async def load(self, name, version=None):
        if name in self._store:
            # Load latest if version is None
            idx = -1 if version is None else version
            try:
                return self._store[name][idx]
            except IndexError:
                return None
        return None

async def main():
    """Main function to simulate a multi-tool agent run."""
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        print("ERROR: Please update the GEMINI_API_KEY in the script before running.")
        return

    # In a real app, the Runner provides the context. We mock it here.
    mock_artifact_service = MockArtifactService()
    mock_context = ToolContext(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        save_artifact_func=mock_artifact_service.save,
        load_artifact_func=mock_artifact_service.load,
    )
    
    # --- This part would be done by the code in the Canvas ---
    print("--- (AGENT STEP 1 - From Canvas) Generating Image... ---")
    # Simulate having a pre-generated image artifact for this example
    # In a real flow, the other tool would run first.
    dummy_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82' # 1x1 black pixel PNG
    image_artifact = types.Part.from_bytes(data=dummy_image_bytes, mime_type="image/png")
    artifact_filename = "user:test_image.png"
    await mock_context.save_artifact(filename=artifact_filename, artifact=image_artifact)
    print(f"Image artifact '{artifact_filename}' saved for test.\n")
    # --- End of Canvas part ---

    print("--- (AGENT STEP 2) Judging Image Quality... ---")
    judgement = await image_judging_tool.func(
        context=mock_context,
        artifact_filename=artifact_filename
    )
    print("\n--- Tool Execution Result (Image Judgement) ---")
    print(judgement)
    print("-------------------------------------------\n")


if __name__ == "__main__":
    asyncio.run(main())
