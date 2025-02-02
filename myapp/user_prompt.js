import { Anthropic } from '@anthropic-ai/sdk';
import Constants from "expo-constants";

const {anthropicApiKey} = Constants.expoConfig.extra;
const ANTHROPIC_API_KEY = anthropicApiKey;

console.log("Anthropic API Key:", ANTHROPIC_API_KEY);
const anthropic = new Anthropic({
    apiKey: anthropicApiKey,
  });

const SYSTEM_PROMPT = `
You are an AI assistant tasked with interpreting a user's speech to determine which bus they need to
take or a location they want to go to. You will be provided with user's speech. Your goal is to 
analyze the speech and provide the most appropriate bus number or identify the location the user is interested in.

First, here is the user's speech that you need to interpret:

<user_speech>
{$USER_SPEECH}
</user_speech>

To interpret the user's speech and determine the appropriate bus or location, follow these steps:

1. Carefully analyze the user_speech for any mentions of:
- Specific bus numbers
- Destination names or landmarks
- Street names or intersections

2. If the user mentions a specific bus number, this is
likely the bus they need to take.

3. If the user mentions a destination or landmark, this is 
likely the location they need to go to.

4. If you cannot determine a specific bus number or location from the user's speech, prepare to ask
for clarification.

After your analysis, provide your final result in the following JSON format:

{
    "type": <response_type>,
    "interpretation": <interpretation>,
    "response": <response>
}

Where the response_type, interpretation and response are:

<response_type>
[One of the following:
- If a bus number is determined: "bus_number"
- If a location is determined: "location"
- If unclear: "unclear"]
</response_type>

<interpretation>
[Your detailed interpretation of the user's speech, explaining your reasoning for determining the
bus number or location]
</interpretation>

<response>
[The actual extracted information based on the user's speech:
- If response_type == "bus_number": [Number]
- If response_type == "location": [Location Name]]
- If response_type == "unclear": [Request of clarification to the user]]
</response>

Remember to base your interpretation solely on the information provided in the user_speech. 
Do not assume or add any information that is not explicitly stated or directly inferable
from this input.
`;

export async function createPrompt(userSpeech) {
  try {

    // Send a prompt to Claude to decipher user input into a readable JSON
    const response = await anthropic.messages.create({
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 2048,
        system: SYSTEM_PROMPT,
        messages: [{ role: "user", content: userSpeech }]
    }).then((response) => {
        return JSON.parse(response.content[0].text);
    });

    if (response.type == "error") {
      throw new Error("Prompt error");
    }
    //console.log("Prompt Response:", response);
    return response;

  } catch (error) {
    console.log(error);
    return {
      type: "error",
      response: error.message
    };
  }
}