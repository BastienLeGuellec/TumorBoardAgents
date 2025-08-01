
import asyncio
from typing import List, Dict, Any, Sequence
import re
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import BaseChatMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 1. Mock EHR Data
mock_ehr = {
    "patient_id": "PAT12345",
    "demographics": {
        "name": "John Doe",
        "age": 65,
        "sex": "Male",
    },
    "medical_history": "History of hypertension, managed with lisinopril. Non-smoker.",
    "presenting_complaint": "Persistent cough and unexplained weight loss over the last 3 months.",
    "imaging_report": {
        "type": "CT Chest",
        "date": "2025-07-15",
        "findings": "A 3.5 cm mass is identified in the upper lobe of the right lung. Associated mediastinal lymphadenopathy is noted. No distant metastases are visible.",
    },
    "pathology_report": {
        "type": "Biopsy of right lung mass",
        "date": "2025-07-20",
        "findings": "Histopathology confirms Non-Small Cell Lung Cancer (NSCLC), adenocarcinoma subtype. Grade 2 differentiation. Immunohistochemistry is positive for TTF-1 and Napsin A. PD-L1 expression is 60%. EGFR and ALK mutations are negative.",
    },
}

# 2. Mock Tools for Specialists
def get_patient_summary() -> str:
    """Retrieves and summarizes the patient's basic information and history."""
    ehr = mock_ehr
    return f"""
    Patient: {ehr['demographics']['name']}, {ehr['demographics']['age']}yo {ehr['demographics']['sex']}.
    History: {ehr['medical_history']}
    Complaint: {ehr['presenting_complaint']}
    """

def get_imaging_findings() -> str:
    """Retrieves and returns the findings from the imaging report."""
    report = mock_ehr["imaging_report"]
    return f"Imaging ({report['type']} on {report['date']}): {report['findings']}"

def get_pathology_diagnosis() -> str:
    """Retrieves and returns the diagnosis from the pathology report."""
    report = mock_ehr["pathology_report"]
    return f"Pathology ({report['type']} on {report['date']}): {report['findings']}"

# 3. Setup Agents
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=
                                          os.getenv("OPENAI_API_KEY"))


oncologist = AssistantAgent(
    name="Oncologist",
    description="The lead of the tumor board, specializes in cancer treatment.",
    model_client=model_client,
    system_message="""You are a medical oncologist and the lead of this Tumor Board. Your primary role is to lead the discussion, gather information from specialists, and formulate a comprehensive treatment plan.
    Your workflow is as follows:
    1.  Request the patient summary from the EHR_Analyst.
    2.  Request the imaging findings from the Radiologist.
    3.  Request the pathology diagnosis from the Pathologist.
    4.  After gathering initial information, create a concise case summary.
    5.  Request a surgical evaluation from the Surgeon.
    6.  Request input on radiation therapy options from the Radiation_Therapist.
    7.  After all specialists have provided their initial evaluations, go around the table and ask each specialist (EHR_Analyst, Radiologist, Pathologist, Surgeon, Radiation_Therapist) if they have any additional comments or recommendations. Address each specialist by name.
    8.  Finally, synthesize all findings and the discussion into a final, consolidated treatment plan.

    At each step, you must explicitly state which specialist you are addressing and what information you need. For example, say: "EHR_Analyst, please provide the patient summary."
    When the final plan is ready, respond with 'FINAL PLAN:'.
    """,
)

ehr_analyst = AssistantAgent(
    name="EHR_Analyst",
    description="Summarizes the patient's Electronic Health Record.",
    tools=[get_patient_summary],
    model_client=model_client,
    system_message="You are an EHR Analyst in a Tumor Board. When asked, provide the patient summary using your tool. If asked for additional comments, state 'No further comments.' or provide relevant insights.",
)

radiologist = AssistantAgent(
    name="Radiologist",
    description="Interprets medical imaging reports.",
    tools=[get_imaging_findings],
    model_client=model_client,
    system_message="You are a Radiologist in a Tumor Board. When asked, provide the imaging findings using your tool. If asked for additional comments, state 'No further comments.' or provide relevant insights.",
)

pathologist = AssistantAgent(
    name="Pathologist",
    description="Analyzes tissue samples and pathology reports.",
    tools=[get_pathology_diagnosis],
    model_client=model_client,
    system_message="You are a Pathologist in a Tumor Board. When asked, provide the pathology diagnosis using your tool. If asked for additional comments, state 'No further comments.' or provide relevant insights.",
)

surgeon = AssistantAgent(
    name="Surgeon",
    description="Evaluates the patient for surgical options.",
    model_client=model_client,
    system_message="You are a Surgeon in a Tumor Board. When the Oncologist provides a case summary and asks for your evaluation, provide your reasoning based on your knowledge. If asked for additional comments, state 'No further comments.' or provide relevant insights.",
)

radiation_therapist = AssistantAgent(
    name="Radiation_Therapist",
    description="Evaluates the patient for radiation therapy options.",
    model_client=model_client,
    system_message="You are a Radiation Therapist in a Tumor Board. When the Oncologist provides a case summary and asks for your evaluation, provide your reasoning based on your knowledge regarding radiation therapy. If asked for additional comments, state 'No further comments.' or provide relevant insights.",
)

# 4. Create the Team and Run the Simulation
def custom_speaker_selection(messages: Sequence[BaseChatMessage]) -> str:
    last_message = messages[-1]
    
    # If the last message is from the Oncologist
    if last_message.source == "Oncologist":
        # Check for explicit addressing for initial evaluations
        match = re.search(r"\b(EHR_Analyst|Radiologist|Pathologist|Surgeon|Radiation_Therapist)\b,", last_message.content)
        if match:
            return match.group(1)
        
        # Check if the Oncologist is asking for additional comments
        if "additional comments or recommendations" in last_message.content:
            specialist_order = ["EHR_Analyst", "Radiologist", "Pathologist", "Surgeon", "Radiation_Therapist"]
            
            # Find the last specialist who spoke (excluding Oncologist)
            last_specialist_index = -1
            for i in range(len(messages) - 2, -1, -1): # Iterate backwards from second to last message
                if messages[i].source in specialist_order:
                    last_specialist_index = specialist_order.index(messages[i].source)
                    break
            
            # Determine the next specialist to ask for comments
            next_specialist_index = (last_specialist_index + 1) % len(specialist_order)
            return specialist_order[next_specialist_index]

    # Default to returning control to the Oncologist to continue the workflow
    return "Oncologist"

termination_condition = TextMentionTermination("FINAL PLAN:")

team = SelectorGroupChat(
    participants=[oncologist, ehr_analyst, radiologist, pathologist, surgeon, radiation_therapist],
    termination_condition=termination_condition,
    model_client=model_client,
    selector_func=custom_speaker_selection,
)

async def main():
    # The Oncologist will now drive the conversation.
    initial_message = "A new case is available for review. Please begin the evaluation process."
    
    stream = team.run_stream(task=initial_message)
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
