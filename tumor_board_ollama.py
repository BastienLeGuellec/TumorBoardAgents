
import asyncio
from typing import List, Dict, Any, Sequence
import re
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import BaseChatMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage

# Install pypdf if not already installed
try:
    import pypdf
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "pypdf"])
    import pypdf

# 1. Mock EHR Data for High-Grade Glioma
mock_ehr = {
    "patient_id": "PAT67890",
    "demographics": {
        "name": "Jane Smith",
        "age": 58,
        "sex": "Female",
    },
    "medical_history": "No significant past medical history.",
    "presenting_complaint": "Sudden onset of seizure, associated with persistent headaches and some cognitive changes (difficulty with word-finding).",
    "neurological_exam": "Alert and oriented, but with mild expressive aphasia. Cranial nerves II-XII intact. Motor strength 5/5 throughout. Mild pronator drift on the right.",
    "imaging_report": {
        "type": "MRI Brain with and without contrast",
        "date": "2025-07-18",
        "findings": "A 4.2 cm ring-enhancing mass in the left frontal lobe, involving the Broca's area, with significant surrounding vasogenic edema. The features are highly suspicious for a high-grade glioma.",
    },
    "pathology_report": {
        "type": "Stereotactic biopsy of left frontal lobe mass",
        "date": "2025-07-22",
        "findings": "Histopathology confirms Glioblastoma, IDH-wildtype (WHO Grade IV). MGMT promoter is methylated.",
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

def get_neurological_exam() -> str:
    """Retrieves the neurological exam findings."""
    return f"Neurological Exam: {mock_ehr['neurological_exam']}"

def get_imaging_findings() -> str:
    """Retrieves and returns the findings from the imaging report."""
    report = mock_ehr["imaging_report"]
    return f"Imaging ({report['type']} on {report['date']}): {report['findings']}"

def get_pathology_diagnosis() -> str:
    """Retrieves and returns the diagnosis from the pathology report."""
    report = mock_ehr["pathology_report"]
    return f"Pathology ({report['type']} on {report['date']}): {report['findings']}"

async def get_clinical_guidelines() -> str:
    """Retrieves and summarizes the clinical practice guidelines for High-Grade Gliomas."""
    try:
        with open("clinical_guidelines_HGG.pdf", "rb") as f:
            reader = pypdf.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        
        if not text.strip():
            return "Could not extract any text from the PDF."

        # Summarize the text using the main model client
        response = await model_client.create(
            messages=[
                SystemMessage(content="You are an expert at summarizing clinical guidelines. Summarize the following text, focusing on treatment recommendations for Glioblastoma, IDH-wildtype, with MGMT promoter methylation."),
                UserMessage(content=text, source="system")
            ]
        )
        summary = response.content
        return summary
    except FileNotFoundError:
        return "Clinical guidelines PDF not found. Please ensure 'clinical_guidelines_HGG.pdf' is in the correct directory."
    except Exception as e:
        return f"An error occurred while processing the PDF: {str(e)}"

# 3. Setup Agents
model_client = OllamaChatCompletionClient(model="your-ollama-model-name")

oncologist_manager = AssistantAgent(
    name="Oncologist_Manager",
    description="The lead oncologist who manages the tumor board meeting.",
    model_client=model_client,
    system_message='''You are an experienced neuro-oncologist and the manager of this tumor board. Your role is to facilitate the discussion, summarize findings, and ensure a comprehensive treatment plan is formulated.

    Your workflow is as follows:
    1.  Start the meeting by stating the case.
    2.  Call on the EHR_Analyst for the patient summary. After they respond, thank them and summarize their findings.
    3.  Call on the Radiologist for imaging findings. After they respond, thank them and summarize their findings.
    4.  Call on the Pathologist for the diagnosis. After they respond, thank them and summarize their findings.
    5.  Call on the Neurologist for a clinical perspective on symptoms. After they respond, thank them and summarize their findings.
    6.  After initial data gathering, ask the Medical_Oncologist, Surgeon, and Radiation_Therapist for their preliminary treatment recommendations.
    7.  Initiate a final round of comments, calling on each specialist **one by one** for their final thoughts.
    8.  Finally, ask the Medical_Oncologist to synthesize all information into a final, consolidated treatment plan.

    You must address each specialist by name to give them the floor. For example: "EHR_Analyst, please provide the patient summary."
    When the final plan is presented, you will end the meeting by saying "END OF DISCUSSION".
    ''',
)

ehr_analyst = AssistantAgent(
    name="EHR_Analyst",
    description="Summarizes the patient's Electronic Health Record.",
    tools=[get_patient_summary, get_clinical_guidelines],
    model_client=model_client,
    reflect_on_tool_use=True,
    system_message="You are an EHR Analyst. When called upon, use your tool to provide the patient summary. You can also consult the clinical guidelines to inform your comments. Do not suggest treatment.",
)

neurologist = AssistantAgent(
    name="Neurologist",
    description="Provides clinical perspective on neurological symptoms.",
    tools=[get_neurological_exam, get_clinical_guidelines],
    model_client=model_client,
    reflect_on_tool_use=True,
    system_message="You are a Neurologist. When called upon, use your tool to retrieve the neurological exam findings and provide your expert interpretation. You can also consult the clinical guidelines to inform your comments. Do not recommend treatment.",
)

radiologist = AssistantAgent(
    name="Radiologist",
    description="Interprets medical imaging reports.",
    tools=[get_imaging_findings, get_clinical_guidelines],
    model_client=model_client,
    reflect_on_tool_use=True,
    system_message="You are a Neuroradiologist. When called upon, use your tool to provide the imaging findings and your expert interpretation. You can also consult the clinical guidelines to inform your comments. Do not recommend treatment.",
)

pathologist = AssistantAgent(
    name="Pathologist",
    description="Analyzes tissue samples and pathology reports.",
    tools=[get_pathology_diagnosis, get_clinical_guidelines],
    model_client=model_client,
    reflect_on_tool_use=True,
    system_message="You are a Neuropathologist. When called upon, use your tool to provide the pathology diagnosis and comment on its significance. You can also consult the clinical guidelines to inform your comments. Do not recommend treatment.",
)

medical_oncologist = AssistantAgent(
    name="Medical_Oncologist",
    description="Specializes in chemotherapy and systemic treatments.",
    tools=[get_clinical_guidelines],
    model_client=model_client,
    system_message="You are a Medical Oncologist specializing in chemotherapy. When asked for your recommendation, propose a systemic treatment plan, consulting the clinical guidelines if necessary. At the end of the meeting, you will be asked to synthesize all specialist inputs into a final, consolidated treatment plan. Respond with 'FINAL PLAN:' before detailing the plan.",
)

surgeon = AssistantAgent(
    name="Surgeon",
    description="Evaluates the patient for surgical options.",
    tools=[get_clinical_guidelines],
    model_client=model_client,
    system_message="You are a Neurosurgeon. When asked for your evaluation, provide your reasoning on the resectability of the tumor and potential surgical approaches, consulting the clinical guidelines if necessary.",
)

radiation_therapist = AssistantAgent(
    name="Radiation_Therapist",
    description="Evaluates the patient for radiation therapy options.",
    tools=[get_clinical_guidelines],
    model_client=model_client,
    system_message="You are a Radiation Therapist. When asked for your evaluation, provide your reasoning on the role and type of radiation therapy, consulting the clinical guidelines if necessary.",
)

# 4. Create the Team and Run the Simulation
def custom_speaker_selection(messages: Sequence[BaseChatMessage]) -> str:
    last_message = messages[-1]

    # If the last message is from the Oncologist_Manager, find out who it is addressed to.
    if last_message.source == "Oncologist_Manager":
        # Find all mentions of specialists in the message
        mentions = re.findall(r"\b(EHR_Analyst|Radiologist|Pathologist|Neurologist|Medical_Oncologist|Surgeon|Radiation_Therapist)\b", last_message.content)
        if mentions:
            # The last mentioned specialist is the one being addressed
            return mentions[-1]

    # Otherwise, it is the Oncologist_Manager's turn to speak.
    return "Oncologist_Manager"

termination_condition = TextMentionTermination("END OF DISCUSSION")

team = SelectorGroupChat(
    participants=[oncologist_manager, ehr_analyst, radiologist, pathologist, neurologist, medical_oncologist, surgeon, radiation_therapist],
    termination_condition=termination_condition,
    model_client=model_client,
    selector_func=custom_speaker_selection,
)

async def main():
    initial_message = "A new case is available for review: Jane Smith, a 58-year-old female with a new diagnosis of a brain tumor. Please begin the evaluation process."
    
    stream = team.run_stream(task=initial_message)
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
