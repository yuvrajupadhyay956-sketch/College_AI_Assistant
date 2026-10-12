import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ==========================================
# LOAD HUGGING FACE TOKEN
# ==========================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("Error: HF_TOKEN not found in .env")
    exit()


# ==========================================
# CREATE INFERENCE CLIENT
# ==========================================

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)


# ==========================================
# READ JSON DATA
# ==========================================

def load_students():

    with open("students.json", "r") as file:
        return json.load(file)


def save_students(data):

    with open("students.json", "w") as file:
        json.dump(data, file, indent=4)


# ==========================================
# TOOL 1 - STUDENT DATA TOOL
# ==========================================

def student_data_tool(name):

    data = load_students()

    for student in data["students"]:

        if student["name"].lower() == name.lower():
            return student

    return None


# ==========================================
# TOOL 2 - CALCULATOR TOOL
# ==========================================

def calculator_tool(attended, total):

    if total == 0:
        return "Total classes cannot be zero."

    percentage = (attended / total) * 100

    return f"{percentage:.2f}%"


# ==========================================
# TOOL 3 - ATTENDANCE TOOL
# ==========================================

def attendance_tool(name, status):

    data = load_students()

    for student in data["students"]:

        if student["name"].lower() == name.lower():

            student["attendance_status"] = status

            save_students(data)

            return (
                f"{name}'s attendance has been "
                f"marked as {status}."
            )

    return "Student not found."


# ==========================================
# LLM TOOL SELECTION
# ==========================================

def choose_tool(user_input):

    prompt = f"""
You are a College AI Assistant.

Your job is to understand the student's request
and decide which tool should be used.

AVAILABLE TOOLS:

1. student_data

Use when student information is required.

Example:
"Show Yuvraj's information."

Return:

{{
    "tool": "student_data",
    "name": "Yuvraj"
}}


2. calculator

Use when the user asks for attendance percentage.

Example:
"Calculate Yuvraj's attendance percentage."

Return:

{{
    "tool": "calculator",
    "name": "Yuvraj"
}}


3. attendance

Use when the user wants to mark or change attendance.

Example:
"Mark Shankar present."

Return:

{{
    "tool": "attendance",
    "name": "Shankar",
    "status": "Present"
}}


4. general

Use when none of the available tools are required.

Return:

{{
    "tool": "general"
}}


IMPORTANT RULES:

Return ONLY valid JSON.

Do not include markdown.

Do not include explanations.

Do not calculate attendance yourself.

The calculator tool must perform calculations.

The attendance tool must modify attendance.

Student request:

{user_input}
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=150
    )

    return response.choices[0].message.content


# ==========================================
# MAIN PROGRAM
# ==========================================

print("\n===================================")
print("      COLLEGE AI ASSISTANT")
print("===================================")

print("Type 'exit' to close the program.")


while True:

    user_input = input("\nStudent: ")

    if user_input.lower() in ["exit", "quit"]:

        print("\nCollege AI Assistant closed.")

        break


    try:

        # ----------------------------------
        # ASK LLM TO SELECT TOOL
        # ----------------------------------

        llm_response = choose_tool(user_input)

        decision = json.loads(llm_response)

        tool = decision.get("tool")


        # ==================================
        # STUDENT DATA TOOL
        # ==================================

        if tool == "student_data":

            print("\nTool Used: Student Data Tool")

            name = decision.get("name")

            if not name:

                print("Assistant: Student name is required.")

                continue


            student = student_data_tool(name)


            if student:

                print("\nAssistant:")

                print("Name:", student["name"])

                print(
                    "Subject:",
                    student["subject"]
                )

                print(
                    "Classes Attended:",
                    student["attended"]
                )

                print(
                    "Total Classes:",
                    student["total_classes"]
                )

                print(
                    "Attendance Status:",
                    student["attendance_status"]
                )

            else:

                print(
                    "Assistant: Student not found."
                )


        # ==================================
        # CALCULATOR TOOL
        # ==================================

        elif tool == "calculator":

            print(
                "\nTool Used: Student Data Tool"
            )

            name = decision.get("name")


            if not name:

                print(
                    "Assistant: Student name is required."
                )

                continue


            # First tool call
            student = student_data_tool(name)


            if not student:

                print(
                    "Assistant: Student not found."
                )

                continue


            print(
                "Tool Used: Calculator Tool"
            )


            # Second tool call
            percentage = calculator_tool(

                student["attended"],

                student["total_classes"]
            )


            print(
                f"\nAssistant: {name}'s attendance "
                f"percentage is {percentage}"
            )


        # ==================================
        # ATTENDANCE TOOL
        # ==================================

        elif tool == "attendance":

            print(
                "\nTool Used: Attendance Tool"
            )


            name = decision.get("name")

            status = decision.get("status")


            if not name or not status:

                print(
                    "Assistant: Name and attendance "
                    "status are required."
                )

                continue


            result = attendance_tool(
                name,
                status
            )


            print(
                "\nAssistant:",
                result
            )


        # ==================================
        # GENERAL REQUEST
        # ==================================

        elif tool == "general":

            print(
                "\nTool Used: No external tool"
            )

            print(
                "Assistant: I can help with student "
                "information, attendance percentage "
                "and attendance marking."
            )


        # ==================================
        # UNKNOWN TOOL
        # ==================================

        else:

            print(
                "\nAssistant: Unknown tool selected."
            )


    # ======================================
    # INVALID JSON
    # ======================================

    except json.JSONDecodeError:

        print(
            "\nAssistant: LLM returned invalid JSON."
        )


    # ======================================
    # OTHER ERRORS
    # ======================================

    except Exception as error:

        print(
            "\nError:",
            error
        )
