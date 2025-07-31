# Tumor Board Simulation using Multi-Agent Systems

This project simulates a multi-disciplinary tumor board meeting for an oncology patient using a team of AI agents powered by AutoGen.

## Overview

The simulation involves several specialist agents collaborating to create a treatment plan for a patient with Non-Small Cell Lung Cancer. The agents and their roles are:

- **Oncologist**: The lead agent that orchestrates the discussion, queries other specialists, and synthesizes the final treatment plan.
- **EHR_Analyst**: Provides a summary of the patient's Electronic Health Record.
- **Radiologist**: Interprets medical imaging reports.
- **Pathologist**: Analyzes tissue samples and pathology reports.
- **Surgeon**: Evaluates the patient for surgical options based on their knowledge and the provided case summary.

## Setup

1.  **Install Dependencies**: Make sure you have Python 3 installed. Then, install the necessary libraries:

    ```bash
    pip install autogen-agentchat autogen-ext
    ```

2.  **API Key**: The simulation uses an OpenAI model. You must have an OpenAI API key. The key is currently hardcoded in the `tumor_board_simulation.py` file. You should replace the placeholder with your actual key.

    *Note: For production applications, it is strongly recommended to use environment variables to manage API keys securely.*

## How to Run

To run the simulation, execute the following command in your terminal:

```bash
python3 tumor_board_simulation.py
```

The script will print the entire conversation between the agents to the console, culminating in a final, consolidated treatment plan.