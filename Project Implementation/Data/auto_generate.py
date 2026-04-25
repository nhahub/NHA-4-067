from google import genai
import time

client = genai.Client(api_key="AIzaSyAWWHxkMmNiSMaMl7n_PLyHUHuv3ljobIU")

SYSTEM_INSTRUCTIONS = """
You are a specialized CSV data generator. 
Your ONLY job is to output raw CSV rows. 
NEVER include conversational text, headers, or explanations. 
Every single line MUST follow this exact pattern: 
,,,,,,,,"[MESSAGE]",,,,,,,,,,ISSUE_TYPE
"""

MODEL_NAME = "gemini-2.5-flash-lite"

prompt = """
Generate 50 NEW support tickets. 

Follow these examples EXACTLY:
,,,,,,,,"I can't access my dashboard because my account is locked.",,,,,,,,,,account_locked
,,,,,,,,"The system is very slow today.",,,,,,,,,,performance_slow

Categories to use: account_locked, bug_error, billing_invoice_issue, performance_timeout, feature_request_new.
"""

csv_path = r"c:\Users\mohamed\OneDrive\Documents\GitHub\Intelligent-Support-Ticket\Project Implementation\Data\Demo.csv"

with open(csv_path, "a", encoding="utf-8") as file:
    batch = 1
    while batch <= 140:
        print(f"Generating batch {batch}/140...")
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                config={
                    'system_instruction': SYSTEM_INSTRUCTIONS,
                    'temperature': 0.9 # Higher temperature for more variety
                },
                contents=prompt
            )
            
            raw_text = response.text.strip()
            lines = [line.strip() for line in raw_text.split('\n') if line.strip() and line.count(',') >= 10]
            
            if lines:
                file.write("\n" + "\n".join(lines))
                print(f"Success! Batch {batch} added correctly.")
                batch += 1
            
            time.sleep(12)
            
        except Exception as e:
            print(f"Error: {e}. Waiting...")
            time.sleep(60)

print("\nFinished!")
