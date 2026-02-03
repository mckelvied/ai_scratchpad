from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel
import asyncio
from dotenv import load_dotenv

load_dotenv()

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems in the style of Isaac Newton. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly in the style of a viking storyteller.",
)

science_tutor_agent = Agent(
    name="Science Tutor",
    handoff_description="Specialist agent for science questions",
    instructions="You provide assistance with science queries. Explain important events and context clearly in the style of a mad scientist.",
)


async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info={"Guardrail triggered on input:": input_data, " for reason:": final_output.reasoning},
        tripwire_triggered=not final_output.is_homework, # Only answer qustions related to homework
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent, science_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

async def main():
    
    # science example
    try:
        result = await Runner.run(triage_agent, "Why do we feel heat from the sun?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e.guardrail_result.output.output_info)

    # History question
    try:
        result = await Runner.run(triage_agent, "who was the first president of the united states?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e.guardrail_result.output.output_info)

    # General/philosophical question
    try:
        result = await Runner.run(triage_agent, "Where can I get a car key cut?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail blocked this input:", e.guardrail_result.output.output_info)


if __name__ == "__main__":
    asyncio.run(main())