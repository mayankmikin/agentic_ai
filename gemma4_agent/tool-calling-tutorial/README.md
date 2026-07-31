https://machinelearningmastery.com/how-to-implement-tool-calling-with-gemma-4-and-python/

The architectural flow of our application operates in the following way:
## Architecture Flow
1. Define local Python functions that act as our tools
2. Define a strict JSON schema that explains to the language model exactly what these tools do and what parameters they expect
3. Pass the user’s query and the tool registry to the local Ollama API
4. Catch the model’s response, identify if it requested a tool call, execute the corresponding local code, and feed the answer back

