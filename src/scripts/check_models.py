import os
from google import genai
from dotenv import load_dotenv

def check_available_models():
    load_dotenv()
    api_key = os.getenv("API_KEY")

    if not api_key:
        print("Error: API_KEY not found in .env file.")
        return

    print(f"Checking models with API Key: {api_key[:5]}...{api_key[-4:]}")

    try:

        versions = ['v1', 'v1beta']
        for version in versions:
            print(f"\n--- Testing API Version: {version} ---")
            client = genai.Client(api_key=api_key, http_options={'api_version': version})
            try:
                models = list(client.models.list())
                if not models:
                    print(f"No models found for version {version}.")
                else:
                    print(f"Found {len(models)} models:")
                    for m in models:
                        print(f" - {m.name} (Supported: {getattr(m, 'supported_methods', 'Unknown')})")
            except Exception as e:
                print(f"Failed to list models for {version}: {e}")

    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    check_available_models()

