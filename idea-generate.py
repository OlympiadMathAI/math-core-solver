import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

def generate_math_solution(problem_statement: str) -> str:
    """Sends a math problem to Gemini and returns a step-by-step proof/solution."""
    
    # Initialize client (reads GEMINI_API_KEY from environment automatically)
    client = genai.Client()

    system_instruction = (
        "You are an expert mathematical reasoner and Olympiad coach. "
        "Provide a clear, rigorous, step-by-step proof or solution for the given problem. "
        "Use LaTeX for mathematical expressions."
    )

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=problem_statement,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )

    return response.text

if __name__ == "__main__":
    test_problem = input("Enter Problem:")
    print(test_problem)
    print("\n--- Generating Solution ---")
    solution = generate_math_solution(test_problem)
    print(solution)