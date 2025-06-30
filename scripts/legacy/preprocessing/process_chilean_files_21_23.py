# %%
import sys, pymupdf  # import the bindings
import re
import os
import json
import glob
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
import io
from dotenv import load_dotenv
from fdllm import get_caller
from fdllm.sysutils import register_models, list_models
from fdllm.llmtypes import LLMMessage, LLMImage
from fdllm.chat import ChatController
from fdllm.extensions import general_query

#from process_chilean_files import segment_paragraphs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
dotenv_path = ROOT / ".env"
load_dotenv(dotenv_path, override=True)

import openai
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional

try:
    register_models(Path.home() / ".fdllm" / "custom_models.yaml")
except:
    pass

homepath = Path.home()

# %%
def run_ocr_on_image_gen_query(image_object_list):
    caller = get_caller("gpt-4o-mini")
    # caller = get_caller("fw-llama-v3p2-90b-vision-instruct")
    # caller = get_caller("claude-3-5-sonnet-20241022")

    jsonin = {"page_text": "ENCODED ON IMAGE"}
    jsonout = {"extracted_text::Extract the text found in the image": None}

    output = general_query(
        jsonin,
        jsonout,
        caller=caller,
        role="user",
        images=image_object_list,
        detail="high",
    )
    return output["extracted_text"]


def run_ocr_on_image(image_object_list):
    #prompt = """
    #You are an expert in extracting text from images. Can you help me extract text from these images? 
    #Only output the text in the image. Do not add uncessary comments nor modify the text.
    #"""
    prompt = """
    You are a specialized OCR system trained to deliver highly accurate text extraction. Please extract all visible 
    text in each image with precision, including details like formatting, punctuation, and capitalization. 
    Maintain the text's exact sequence as shown in the image without adding or omitting any content. 
    Provide only the text as output—no additional comments, descriptions, or formatting adjustments.
    """
    # prompt = '''
    # Extract all the text that is present in the image provided. Do not translate or modify the text.
    #'''

    # caller = get_caller("gpt-4o")
    caller = get_caller("gpt-4o-mini")
    #caller = get_caller("fw-llama-v3p2-90b-vision-instruct")

    message = LLMMessage(
        Role="user",
        Message=prompt,
        Images=LLMImage.list_from_images(
            image_object_list, detail="high"
        ),  # needs to be a list in input
    )
    output = caller.call([message])
    return output


# extract all text from the pdf with llm image and store it in a list


def extract_text_from_pdf(path_pdf, name_pdf, year, range_pages=None):
    # transform all pages of the pdf into images
    doc = pymupdf.open(path_pdf)  # open document
    # create a folder to store the images
    os.makedirs(
        f"data/Chile/ECEP {year}/folder_pdf2im/images_{name_pdf}", exist_ok=True
    )

    for page in doc:  # iterate through the pages
        pix = page.get_pixmap()  # render page to an image
        pix.save(
            f"data/Chile/ECEP {year}/folder_pdf2im/images_{name_pdf}/page-%i.png"
            % page.number
        )  # save image as PNG file

    if range_pages:
        images_pdf = []
        for i in range_pages:
            images_pdf.append(
                Image.open(
                    f"data/Chile/ECEP {year}/folder_pdf2im/images_{name_pdf}/page-{i}.png"
                )
            )
    else:
        images_pdf = [
            Image.open(f)
            for f in Path(f"folder_pdf2im/images_{name_pdf}").glob("*.png")
        ]

    print(
        f"Done transforming pdf into images. Number of images to be processed: {len(images_pdf)}"
    )

    try:
        # extract text from the images
        pdf_text = dict()
        for im, page_number in zip(images_pdf, range_pages):
            print(f"Extracting text from page {page_number} with OCR...")
            # if using the general query
            #text_from_image = run_ocr_on_image_gen_query([im])
            # if using normal ocr
            output = run_ocr_on_image([im])
            text_from_image = output.Message
            pdf_text[f"Page {page_number}"] = text_from_image
    except:
        pass
    finally:
        # close the images
        for im in images_pdf:
            im.close()

    # doc.close()

    return pdf_text


def segment_text2para(text):
    # Regular expression to match each numbered paragraph
    pattern = r"(?s)(\d+\..*?D.*?(?=\d+\.|$))"

    # Find all matching segments
    segments = re.findall(pattern, text)

    # Clean and print each paragraph
    paragraphs = [segment.strip() for segment in segments]

    return paragraphs

# code from process_chilean_files.py
# functioons:
# - check_end_paragraph
# - segment_paragraphs

def check_end_paragraph(line, Q_number_prev):
    #print(f"Q_number_prev: {Q_number_prev}")
    Q_number_next = str(int(Q_number_prev) + 1)
    print(f"Q_number_next: {Q_number_next}")
    print(f"Line: {line}")
    # if Q_number is 1 digit
    if len(Q_number_next) == 1:
        if len(line) != 0:
            if line[0] == Q_number_next:
        #if line[0] == Q_number_next:
                return True
    # if Q_number is 2 digits
    elif len(Q_number_next) == 2:
        if line[0] == Q_number_next[0] and line[1] == Q_number_next[1]:
            return True
    return False

def segment_paragraphs(text):
    # Split the text into lines
    lines = text.split("\n")

    # Check whether the page starts with a digit
    # if not, skip to next page because the questions will not be independent
    if not lines[0][0].isdigit():
        print("     Page does not start with a digit, skipping to next page")
        print(f"Text:\n{text}")
        return []

    # Initialize variables
    paragraphs = []
    current_paragraph = []
    Q_number_prev = []

    # add a correction factor to account for the fact that the answer D can be on multiple lines
    correction_factor = 0
    # Iterate through the lines
    for idx, line in enumerate(lines):
        idx_corrected = idx + correction_factor
        if idx_corrected >= len(lines):
            break
        line = lines[idx_corrected]

        # Check if the line starts with a "D"
        if line.startswith("D"):
            # append all the lines from the D line to the next line starting with a digit
            current_paragraph.append(line)
            for line in lines[idx_corrected + 1:]:
                #if line[0].isdigit() :
                if check_end_paragraph(line, Q_number_prev):
                    #print("End of paragraph detected with D line")
                    break
                correction_factor += 1
                current_paragraph.append(line)
            paragraphs.append("\n".join(current_paragraph))
            # Reset the current paragraph
            current_paragraph = []
            Q_number_prev = []
            #print("Q_number_prev reset\n")
        else:
            current_paragraph.append(line)
            # initialize the Q_number_prev if it is empty
            if len(Q_number_prev) == 0:
                #print("Q_number_prev initialization")
                #print(line)
                # ierate through the line and append to Q_number_prev the first digits found
                for char in line:
                    if char.isdigit():
                        Q_number_prev.append(char)
                    else:
                        break
                Q_number_prev = "".join(Q_number_prev)
                #print(f"Q_number_prev: {Q_number_prev}")

    # Add the last paragraph if it exists
    if current_paragraph:
        paragraphs.append("\n".join(current_paragraph).strip())

    return paragraphs


def extract_question_number(paragraph):
    pattern = r"(\d+\.)"
    question_number = re.search(pattern, paragraph)

    # if question_number:
    #    # remove the whole line of the question number
    #    paragraph = re.sub(pattern, "", paragraph)
    if question_number:
        question_number = int(question_number.group(0).replace(".", ""))
        return question_number
    else:
        return None
    


class Answers(BaseModel):
    A: Optional[str] = None
    B: Optional[str] = None
    C: Optional[str] = None
    D: Optional[str] = None


class MCQ_both_languages(BaseModel):
    question_spanish: Optional[str] = None
    choices_spanish: Answers = Field(default_factory=dict)
    question_english: Optional[str] = None
    choices_english: Answers = Field(default_factory=dict)


def process_paragraph_with_LLM(paragraph):

    prompt = f"""
    You are a highly skilled translator and text processor. I have a text containing paragraphs of educational questions and answers in Spanish. I want you to do the following:

    1. **Segment** each paragraph into "Question" and "Answers":
    - "Question" should contain the main body of the paragraph that describes the situation and the instruction. Basically, the part before the list of choices. If there is a paragraph of text before the actual question, include that as well. Do not include the question number at the beginning of the paragraph if there is one.
    - "Answers" should include the list of choices (e.g., A, B, C, D) provided after the question.

    2. **Translate** the segmented text into English.

    The text is as follows:

    -----
    {paragraph}
    -----

    1. Please provide the segmented output in the following JSON format:
    {{
        "question": "<Segmented Question>",
        "A": "<Segmented Choice A>",
        "B": "<Segmented Choice B>",
        "C": "<Segmented Choice C>",
        "D": "<Segmented Choice D>"
    }}

    2. Please provide the translated output in the following JSON format:
    {{
        "question": "<Translated Question>",
        "A": "<Translated Choice A>",
        "B": "<Translated Choice B>",
        "C": "<Translated Choice C>",
        "D": "<Translated Choice D>"
    }}
    """
    # print(prompt)

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        #model="gpt-4o-2024-08-06",
        model = "gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        # tools=[
        #    openai.pydantic_function_tool(MCQ_both_languages),
        # ],
        response_format=MCQ_both_languages,
    )

    # output = completion.choices[0].message.tool_calls[0].function.parsed_arguments
    return completion


# %%
### PATH TO PDF HERE
path_mcqs = ""
###################
year = 2023
range_pages = range(1, 29)
N_qu_expected =60
name_file = "sc_am23"

# %%
# Transform the pdf into images and extract text from them all at once
pdf_text = extract_text_from_pdf(path_mcqs, name_file, year, range_pages=range_pages)

# %%
import pprint

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(pdf_text)

# %%
# save as a json
with open(f"data/Chile/ECEP {year}/folder_pdf2im/pdf_text_{name_file}.json", "w") as f:
    json.dump(pdf_text, f, indent=4)
# %%

# read another json file (to avoid running the previous cells)
with open(f"data/Chile/ECEP {year}/folder_pdf2im/pdf_text_{name_file}.json", "r") as f:
    pdf_text = json.load(f)

# %%

df_english = pd.DataFrame(
    columns=[
        "Question number",
        "Question",
        "Answer A",
        "Answer B",
        "Answer C",
        "Answer D",
    ]
)
df_spanish = pd.DataFrame(
    columns=[
        "Question number",
        "Question",
        "Answer A",
        "Answer B",
        "Answer C",
        "Answer D",
    ]
)

Q_number_prev = 0

for page, text in pdf_text.items():
    # print(text)
    page_nb = re.findall(r"\d+", page)[0]
    print(f"Processing page {page_nb}...")
    paragraphs = segment_text2para(text)
    #paragraphs = segment_paragraphs(text) # uses the segmentation of years 2017-2020

    if len(paragraphs) == 0:
        print(
            f"   Paragraphs segmentation did not work in page {page_nb}. Skipping to next page..."
        )
        continue

    for paragraph in paragraphs:

        Q_number = extract_question_number(paragraph)

        if Q_number is None:
            print(
                f"Question number not found in page {page_nb}, skipping to next question...\n"
            )
            print(f"Paragraph: -------------------------\n{paragraph}\n-------------------------\n\n")
            continue

        if Q_number not in range(1, N_qu_expected + 1):
            print(
                f"Question number {Q_number} is greater than expected number of questions {N_qu_expected}. There might be an error in paragraph segmentation. Skipping to next question..."
            )
            print(f"Paragraph: -------------------------\n{paragraph}\n-------------------------\n\n")
            continue

        #if Q_number != Q_number_prev + 1:
        #    print(
        #        f"Question number {Q_number} is not the expected number {Q_number_prev + 1}. There might be an error in paragraph segmentation. Skipping to next question..."
        #    )
        #    print(f"\nQuestion number {Q_number_prev} might have been cut off.")
        #    print(f"Removing the last row of the dataframe... (question {Q_number_prev})")
        #    print(f"Paragraph: -------------------------\n{paragraph}\n-------------------------\n\n")
        #    # remove the row with value Q_number_prev in column "Question number" if there is one
        #    if Q_number_prev in df_spanish["Question number"].values:
        #        df_spanish = df_spanish[df_spanish["Question number"] != Q_number_prev].reset_index(drop=True)
        #        df_english = df_english[df_english["Question number"] != Q_number_prev].reset_index(drop=True)
        #    continue

        # extract data with LLM
        completion = process_paragraph_with_LLM(paragraph)

        # if completion.choices[0].message and len(completion.choices[0].message.tool_calls) > 0:
        if completion.choices[0].message.parsed:
            parsed_arguments = completion.choices[0].message.parsed

            # spanish
            question_spanish = parsed_arguments.question_spanish
            question_spanish = question_spanish.replace(
                "\n", " "
            )  # remove new lines in the question
            answers_spanish = parsed_arguments.choices_spanish
            # if answers not empty, remove the last character if it is a point
            # if len(answers_spanish.A) > 0 and len(answers_spanish.B) > 0 and len(answers_spanish.C) > 0 and len(answers_spanish.D) > 0:
            if (
                answers_spanish.A is not None
                and len(answers_spanish.A) > 0
                and answers_spanish.B is not None
                and len(answers_spanish.B) > 0
                and answers_spanish.C is not None
                and len(answers_spanish.C) > 0
                and answers_spanish.D is not None
                and len(answers_spanish.D) > 0
            ):

                ans_spanish_A = (
                    answers_spanish.A[:-1]
                    if answers_spanish.A[-1] == "."
                    else answers_spanish.A
                )
                ans_spanish_B = (
                    answers_spanish.B[:-1]
                    if answers_spanish.B[-1] == "."
                    else answers_spanish.B
                )
                ans_spanish_C = (
                    answers_spanish.C[:-1]
                    if answers_spanish.C[-1] == "."
                    else answers_spanish.C
                )
                ans_spanish_D = (
                    answers_spanish.D[:-1]
                    if answers_spanish.D[-1] == "."
                    else answers_spanish.D
                )

                # english
                question_english = parsed_arguments.question_english
                question_english = question_english.replace(
                    "\n", " "
                )  # remove new lines in the question
                answers_english = parsed_arguments.choices_english
                # if answers not empty, remove the last character if it is a point
                # if len(answers_english.A) > 0 and len(answers_english.B) > 0 and len(answers_english.C) > 0 and len(answers_english.D) > 0:
                if (
                    answers_english.A is not None
                    and len(answers_english.A) > 0
                    and answers_english.B is not None
                    and len(answers_english.B) > 0
                    and answers_english.C is not None
                    and len(answers_english.C) > 0
                    and answers_english.D is not None
                    and len(answers_english.D) > 0
                ):

                    ans_english_A = (
                        answers_english.A[:-1]
                        if answers_english.A[-1] == "."
                        else answers_english.A
                    )
                    ans_english_B = (
                        answers_english.B[:-1]
                        if answers_english.B[-1] == "."
                        else answers_english.B
                    )
                    ans_english_C = (
                        answers_english.C[:-1]
                        if answers_english.C[-1] == "."
                        else answers_english.C
                    )
                    ans_english_D = (
                        answers_english.D[:-1]
                        if answers_english.D[-1] == "."
                        else answers_english.D
                    )

                    # copy in dataframes if all answers are not empty
                    df_spanish.loc[len(df_spanish)] = [
                        Q_number,
                        question_spanish,
                        ans_spanish_A,
                        ans_spanish_B,
                        ans_spanish_C,
                        ans_spanish_D,
                    ]
                    df_english.loc[len(df_english)] = [
                        Q_number,
                        question_english,
                        ans_english_A,
                        ans_english_B,
                        ans_english_C,
                        ans_english_D,
                    ]
                else:
                    print("Processing failed (english answers format)")
                    print(f"Completion: {completion}")
                    print(f"Paragraph: {paragraph}\n")
            else:
                print("Processing failed (spanish answers format)")
                print(f"Completion: {completion}")
                print(f"Paragraph: {paragraph}\n")
        else:
            print("Processing failed (completion format)")
            print(f"Completion: {completion}")
            print(f"Paragraph: {paragraph}\n")

        # update Q_number_prev
        #Q_number_prev = Q_number

# %%
print(df_spanish.shape)
print(df_english.shape)
print(df_spanish.loc[0, "Question"])
print(df_spanish["Question number"].unique())
# check that all unique question number only appears once
if len(df_spanish["Question number"].unique()) != len(df_spanish):
    print("⚠️ There are duplicate question numbers in the dataframe. See indices:")
    # print the index of the duplicate question numbers
    print(
        df_spanish[
            df_spanish["Question number"].duplicated(keep=False)
        ].index.tolist()
    )
# check that question number are ordered in ascending order
if not df_spanish["Question number"].is_monotonic_increasing:
    print("⚠️ Question numbers are not ordered in ascending order.")

print(df_spanish.head(), df_english.head())

# %%
df_spanish.tail(10)

# %%
# remove some rows based on rows index
idx_remove = [32,33]
df_spanish = df_spanish.drop(idx_remove).reset_index(drop=True)
df_english = df_english.drop(idx_remove).reset_index(drop=True)

# %%
print(f"Reading answers_{name_file}.csv")

answers_csv = pd.read_csv(
    f"./data/Chile/ECEP {year}/answers_{name_file}.csv"
)  # change the path
answers_csv.head()


# %%
# add the correct answer to the dataframes
def add_correct_answer(df, answers_csv):
    for i, row in df.iterrows():
        question_number = row["Question number"]
        correct_letter = answers_csv[
            answers_csv["Question number"] == int(question_number)
        ]["Correct answer"].values[0]
        df.loc[i, "Correct answer"] = correct_letter
    return df


df_spanish = add_correct_answer(df_spanish, answers_csv)
df_english = add_correct_answer(df_english, answers_csv)
print(df_spanish.shape)
print(df_english.shape)
print(df_spanish.head(), df_english.head())

# %%
# save the dataframe to an excel file
print(f"Saving to mcqs_SP_{name_file}.xlsx and mcqs_EN_{name_file}.xlsx")
df_spanish.to_excel(
    f"./data/Chile/ECEP {year}/mcqs_SP_{name_file}.xlsx", index=False
)  # change path
df_english.to_excel(
    f"./data/Chile/ECEP {year}/mcqs_EN_{name_file}.xlsx", index=False
)  # change path


# %%


def merge_excels_with_origin(folders_list, pattern):
    # Create an empty list to store all dataframes
    all_dfs = []

    # Loop through each folder in the list
    for folder in folders_list:
        print(f"Reading files from {folder}")
        # Find all Excel files ending with "pattern" in the folder
        # files = glob.glob(os.path.join(folder, '*edited*.xlsx'))  # Adjust if your files have different extensions
        files = glob.glob(os.path.join(folder, pattern))

        # Loop through each file found
        for file in files:
            # Read the Excel file into a dataframe
            df = pd.read_excel(file)

            # Add a new column "origin of the file" with the file path or name
            # df['origin of the file'] = file  # or os.path.basename(file) for just the filename
            df["origin of the file"] = file.split(
                "Questions\\processed files\\Chile\\"
            )[-1]

            # Append the dataframe to the list
            all_dfs.append(df)
        print(f"    {len(files)} files read")

    # Concatenate all dataframes in the list
    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df


year = "2021"

load_dotenv(override=True)
BENCHMARKS_DIR = Path(os.getenv("BENCHMARKS_DIR", "./"))

path_files = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / f"ECEP {year}"

path_output_file_SP = f"./data/Chile/ECEP {year}/merged_SP_ecep_{year}_all.xlsx"
path_output_file_EN = f"./data/Chile/ECEP {year}/merged_EN_ecep_{year}_all.xlsx"
pattern_edited = "*edited*.xlsx"
pattern_EN_all = f"mcqs_EN_*{year[-2:]}.xlsx"
pattern_SP_all = f"mcqs_SP_*{year[-2:]}.xlsx"
# %%
merged_df_SP = merge_excels_with_origin([path_files], pattern=pattern_SP_all)
merged_df_EN = merge_excels_with_origin([path_files], pattern=pattern_EN_all)
print(merged_df_SP.shape, merged_df_EN.shape)
print(merged_df_SP.head(), merged_df_EN.head())

# %%
# Save the merged dataframe to a new Excel file
# merged_df_SP.to_excel(path_output_file_SP, index=False)
# merged_df_SP.to_excel(path_files / f"merged_SP_ecep_{year}_all.xlsx", index=False)
# merged_df_EN.to_excel(path_output_file_EN, index=False)
# merged_df_EN.to_excel(path_files / f"merged_EN_ecep_{year}_all.xlsx", index=False)
# %%
