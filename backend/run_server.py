"""Entry point for running the MemoryMind backend server with startup activation prompt."""

import os
import sys


def prompt_activation_if_needed() -> None:
    from app import config

    if not config.is_active():
        print("=" * 60)
        print(" MemoryMind Backend - Provider API Key Required")
        print("=" * 60)
        print(f"Current LLM Provider: {config.LLM_PROVIDER}")
        print("No API key was detected for this provider.")
        print("To activate the service, please enter your API key below.")
        print("(Press Enter without typing to skip and activate later via the extension UI)")
        print("-" * 60)

        try:
            # Check if running interactively
            if sys.stdin.isatty():
                entered_key = input(f"Enter {config.LLM_PROVIDER.upper()} API Key: ").strip()
                if entered_key:
                    config.set_provider_key(entered_key)
                    print(f"✓ Provider '{config.LLM_PROVIDER}' activated successfully and saved to backend/.env!")
                else:
                    print(f"Skipped for now. You can activate provider '{config.LLM_PROVIDER}' anytime from the extension UI.")
            else:
                print(f"Running non-interactively. You can activate provider '{config.LLM_PROVIDER}' anytime from the extension UI.")
        except (EOFError, KeyboardInterrupt):
            print("\nStartup continuing without API key.")
        print("=" * 60)
        print()


if __name__ == "__main__":
    prompt_activation_if_needed()

    import uvicorn
    from app import config

    uvicorn.run(
        "app.main:app",
        host=config.BACKEND_HOST,
        port=config.BACKEND_PORT,
        reload=False,
    )

