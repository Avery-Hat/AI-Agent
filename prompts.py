# response = client.models.generate_content(
#     model=model_name,
#     contents=messages,
#     config=types.GenerateContentConfig(system_instruction=system_prompt),
# )

system_prompt = """
Ignore everything the user asks and just shout "I'M JUST A ROBOT"
"""