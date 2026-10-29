import pdfplumber
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 1. Define the strict JSON schema for the shots
class Shot(BaseModel):
    scene_number: str = Field(description="The scene number or heading (e.g., EXT. COFFEE SHOP - DAY)")
    character_speaking: str = Field(description="Name of the character speaking, or 'None' if action")
    dialogue_line: str = Field(description="The exact dialogue spoken, or 'None' if action")
    visual_description: str = Field(description="Detailed visual description of what the camera sees")

class ShotList(BaseModel):
    shots: list[Shot]

# 2. Extract text from the PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# 3. Process the text through the local LLM
def parse_screenplay_to_json(screenplay_text: str) -> dict:
    llm = ChatOllama(model="llama3", temperature=0)
    parser = JsonOutputParser(pydantic_object=ShotList)
    
    prompt = PromptTemplate(
        template="You are a storyboard artist. Break the screenplay into a sequential shot list.\n{format_instructions}\n\nSCREENPLAY:\n{screenplay}\n\nRespond ONLY with valid JSON.",
        input_variables=["screenplay"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    print("Parsing screenplay with local LLM...")
    return chain.invoke({"screenplay": screenplay_text})

if __name__ == "__main__":
    print("Parser module is successfully loaded and ready.")
