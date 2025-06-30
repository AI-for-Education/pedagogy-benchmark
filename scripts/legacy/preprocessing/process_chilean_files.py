# %%
# import packages
from PyPDF2 import PdfReader
import pdfplumber
import fitz # PyMuPDF
#import PyPDF2

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pprint
import os
import glob
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
import openai
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional

homepath = Path.home()

# %%
def extract_text_from_pdf(pdf_path, page_number):
    # Open the PDF file in read-binary mode
    with open(pdf_path, "rb") as file:
        # Create a PDF reader object
        pdf_reader = PdfReader(file)
        #print(f"Length of the pdf: {len(pdf_reader.pages)} pages")

        page = pdf_reader.pages[page_number]
        text = page.extract_text() 

        # split on new lines and remove the last line of the text, which is the number of the page
        text = text.split("\n")[:-1]

        # join the text back together
        text = "\n".join(text)
        
        return text

def extract_text_from_pdf_plumber(pdf_path, page_number):
    # Open the PDF file with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        text = page.extract_text()
        
        # Split on new lines and remove the last line, which is the page number
        text = text.split("\n")[:-1]

        # Join the text back together
        text = "\n".join(text)
        
        return text

def extract_text_from_pdf_pymupdf(pdf_path, page_number):
    # Open the PDF with PyMuPDF
    pdf_document = fitz.open(pdf_path)
    
    page = pdf_document.load_page(page_number)  # page_number is zero-based
    text = page.get_text("text")
    
    # Split on new lines and remove the last line, which is the page number
    text = text.split("\n")[:-1]

    # Join the text back together
    text = "\n".join(text)
    
    return text

def check_end_paragraph(line, Q_number_prev):
    #print(f"Q_number_prev: {Q_number_prev}")
    Q_number_next = str(int(Q_number_prev) + 1)
    #print(f"Q_number_next: {Q_number_next}")
    #print(f"Line: {line}")
    # if Q_number is 1 digit
    if len(Q_number_next) == 1:
        if line[0] == Q_number_next:
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
    - "Question" should contain the main body of the paragraph that describes the situation and the instruction. Basically, the part before the list of choices. If there is a paragraph of text before the actual question, include that as well.
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
    #print(prompt)

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
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
        #tools=[
        #    openai.pydantic_function_tool(MCQ_both_languages),
        #],
        response_format=MCQ_both_languages,
    )

    #output = completion.choices[0].message.tool_calls[0].function.parsed_arguments
    return completion

# %%
########################
# PARAMETERS TO SET:
########################
# - range_pages: range of pages to process in the pdf (be careful to python indexing: substract 1 to the firts page number, do nothing for end of page because range(N) goes from 0 to N-1) 
# - pdf_path: path to the pdf file
########################

range_pages = range(1, 5)
################################
## PATH TO PDFS HERE
path_mcqs = ""
###################################
save_format = "em_fp23"

print(f"Path to MCQs: {path_mcqs}")

# check the path=
with open(path_mcqs, "rb") as file:
    pdf_reader = PdfReader(file)
    print(f"Length of the pdf: {len(pdf_reader.pages)} pages")

# %%
df_english = pd.DataFrame(columns=['Question number', 'Question', 'Answer A', 'Answer B', 'Answer C', 'Answer D'])
df_spanish = pd.DataFrame(columns=['Question number', 'Question', 'Answer A', 'Answer B', 'Answer C', 'Answer D'])

for idx in range_pages:
    print(f"Processing page {idx}...")
    # extract text from pdf one page at a time
    text = extract_text_from_pdf_plumber(path_mcqs, idx)

    # remove the whole first line if it starts with "◦"
    #lines = text.splitlines()
    #if lines and lines[0].strip() == "◦":
    #    lines = lines[1:]
    #text = "\n".join(lines)

    print(f"Text page {idx}:\n{text}\n")
    continue

    #if idx == 8 or idx == 20:
    #    continue

    # segment the text into paragraphs (1 paragraph = 1 question)
    paragraphs = segment_paragraphs(text)

    #for paragraph in paragraphs:
    #    print(f"Paragraph: \n{paragraph}")
    #    print("\n")
    #continue

    for paragraph in paragraphs:
        # extract number of the question and remove it from the paragraph
        Q_number = []
        N_digits = 0
        #print(f"Paragraph: \n{paragraph}")

        while paragraph[0].isdigit():
            Q_number.append(paragraph[0])
            paragraph = paragraph[1:]
            N_digits += 1
        Q_number = "".join(Q_number)
        #print(f"Question number: {Q_number}")
        # remove it from the paragraph
        paragraph = paragraph[N_digits-1:]

        # extract data with LLM
        completion = process_paragraph_with_LLM(paragraph)

        #if completion.choices[0].message and len(completion.choices[0].message.tool_calls) > 0:
        if completion.choices[0].message.parsed:
            #parsed_arguments = completion.choices[0].message.tool_calls[0].function.parsed_arguments
            parsed_arguments = completion.choices[0].message.parsed

            # spanish
            question_spanish = parsed_arguments.question_spanish
            answers_spanish = parsed_arguments.choices_spanish
            # if answers not empty, remove the last character if it is a point
            #if len(answers_spanish.A) > 0 and len(answers_spanish.B) > 0 and len(answers_spanish.C) > 0 and len(answers_spanish.D) > 0:
            if (answers_spanish.A is not None and len(answers_spanish.A) > 0 and
                answers_spanish.B is not None and len(answers_spanish.B) > 0 and
                answers_spanish.C is not None and len(answers_spanish.C) > 0 and
                answers_spanish.D is not None and len(answers_spanish.D) > 0):

                ans_spanish_A = answers_spanish.A[:-1] if answers_spanish.A[-1] == "." else answers_spanish.A
                ans_spanish_B = answers_spanish.B[:-1] if answers_spanish.B[-1] == "." else answers_spanish.B
                ans_spanish_C = answers_spanish.C[:-1] if answers_spanish.C[-1] == "." else answers_spanish.C
                ans_spanish_D = answers_spanish.D[:-1] if answers_spanish.D[-1] == "." else answers_spanish.D

                # english
                question_english = parsed_arguments.question_english
                answers_english = parsed_arguments.choices_english
                # if answers not empty, remove the last character if it is a point
                #if len(answers_english.A) > 0 and len(answers_english.B) > 0 and len(answers_english.C) > 0 and len(answers_english.D) > 0:
                if (answers_english.A is not None and len(answers_english.A) > 0 and
                    answers_english.B is not None and len(answers_english.B) > 0 and
                    answers_english.C is not None and len(answers_english.C) > 0 and
                    answers_english.D is not None and len(answers_english.D) > 0):

                    ans_english_A = answers_english.A[:-1] if answers_english.A[-1] == "." else answers_english.A
                    ans_english_B = answers_english.B[:-1] if answers_english.B[-1] == "." else answers_english.B
                    ans_english_C = answers_english.C[:-1] if answers_english.C[-1] == "." else answers_english.C
                    ans_english_D = answers_english.D[:-1] if answers_english.D[-1] == "." else answers_english.D

                    # copy in dataframes if all answers are not empty
                    df_spanish.loc[len(df_spanish)] = [Q_number, question_spanish, ans_spanish_A, ans_spanish_B, ans_spanish_C, ans_spanish_D]
                    df_english.loc[len(df_english)] = [Q_number, question_english, ans_english_A, ans_english_B, ans_english_C, ans_english_D]
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

# %%
print(df_spanish.shape)
print(df_english.shape)
print(df_spanish.loc[0, "Question"])
print(df_spanish.head(), df_english.head())

# %%
print(f"Reading answers_{save_format}.csv")

answers_csv = pd.read_csv(f"./data/ECEP 2020/answers_{save_format}.csv") # change the path
answers_csv.head()
# %%
# add the correct answer to the dataframes
def add_correct_answer(df, answers_csv):
    for i, row in df.iterrows():
        question_number = row['Question number']
        correct_letter = answers_csv[answers_csv['Question number'] == int(question_number)]['Correct answer'].values[0]
        df.loc[i, 'Correct answer'] = correct_letter
    return df

df_spanish = add_correct_answer(df_spanish, answers_csv)
df_english = add_correct_answer(df_english, answers_csv)
print(df_spanish.shape)
print(df_english.shape)
print(df_spanish.head(), df_english.head())

# %% 
# save the dataframe to an excel file
print(f"Saving to mcqs_SP_{save_format}.xlsx and mcqs_EN_{save_format}.xlsx")
df_spanish.to_excel(f"./data/ECEP 2020/mcqs_SP_{save_format}.xlsx", index=False) # change path
df_english.to_excel(f"./data/ECEP 2020/mcqs_EN_{save_format}.xlsx", index=False) # change path

# %%

#############################
# Post-processing functions #
#############################

# function to prepare the csv file to run inference on LLM

def prepare_csv_for_inference(df, filename, columns_to_drop):
    df.insert(6, "Answer E", None)
    df.insert(7, "Answer F", None)
    df.insert(8, "Answer G", None)
    df.drop(columns = columns_to_drop, inplace=True)
    #df["Adaptation Needed"] = None
    #df["Category"] = "GenPK"
    df.to_csv(filename, index=False)
    return df

# function to merge all the excel files in a folder and add a column with the origin of the file

def merge_excels_with_origin(folders_list, pattern):
    # Create an empty list to store all dataframes
    all_dfs = []
    
    # Loop through each folder in the list
    for folder in folders_list:
        print(f"Reading files from {folder}")
        # Find all Excel files ending with "pattern" in the folder
        #files = glob.glob(os.path.join(folder, '*edited*.xlsx'))  # Adjust if your files have different extensions
        files = glob.glob(os.path.join(folder, pattern))
        
        # Loop through each file found
        for file in files:
            # Read the Excel file into a dataframe
            df = pd.read_excel(file)
            
            # Add a new column "origin of the file" with the file path or name
            #df['origin of the file'] = file  # or os.path.basename(file) for just the filename
            df['origin of the file'] = file.split("Questions\\processed files\\Chile\\")[-1]
            
            # Append the dataframe to the list
            all_dfs.append(df)
        print(f"    {len(files)} files read")
    
    # Concatenate all dataframes in the list
    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

year = "2023"

path_files = ""

folders_list = [path_files / f"PC {year}", path_files / f"EP {year}", path_files / f"SC {year}", path_files / f"EM {year}", path_files / f"ED {year}", path_files / f"EA {year}", path_files / f"EMTP {year}"]
#folders_list = [path_files / f"SC {year}", path_files / f"EM {year}", path_files / f"ED {year}", path_files / f"EA {year}", path_files / f"EMTP {year}"]
for folder in folders_list:
    print(folder)
path_output_file_SP = f"./data/Chile/ECEP {year}/merged_SP_ecep_{year}_all.xlsx"
path_output_file_EN = f"./data/Chile/ECEP {year}/merged_EN_ecep_{year}_all.xlsx"
pattern_edited = "*edited*.xlsx"
pattern_EN_all = f"mcqs_EN_*{year[-2:]}.xlsx"
pattern_SP_all = f"mcqs_SP_*{year[-2:]}.xlsx"
# %%
merged_df_SP = merge_excels_with_origin(folders_list, pattern=pattern_SP_all)
merged_df_EN = merge_excels_with_origin(folders_list, pattern=pattern_EN_all)
print(merged_df_SP.shape, merged_df_EN.shape)
print(merged_df_SP.head(), merged_df_EN.head())
# %%
# Save the merged dataframe to a new Excel file
#merged_df_SP.to_excel(path_output_file_SP, index=False)
#merged_df_SP.to_excel(path_files / f"merged_SP_ecep_{year}_all.xlsx", index=False)
#merged_df_EN.to_excel(path_output_file_EN, index=False)
#merged_df_EN.to_excel(path_files / f"merged_EN_ecep_{year}_all.xlsx", index=False)






# %%
# Process the MCQs edited files
homepath = Path.home()
path_mcqs_2017 = ""
path_mcqs_2018 = ""
path_mcqs_2019 = ""
path_mcqs_2020 = ""


mcqs_2017_edited = pd.read_excel(path_mcqs_2017 / "merged_EN_ecep_2017_all_edited_finale.xlsx")
mcqs_2018_edited = pd.read_excel(path_mcqs_2018 / "merged_EN_ecep_2018_all_edited.xlsx")
mcqs_2019_edited = pd.read_excel(path_mcqs_2019 / "merged_EN_ecep_2019_all_without_duplicates_edited.xlsx")
mcqs_2020_edited = pd.read_excel(path_mcqs_2020 / "merged_EN_ecep_2020_all_without_duplicates_edited.xlsx")


print(mcqs_2017_edited.shape)
print(mcqs_2018_edited.shape)
print(mcqs_2019_edited.shape)
print(mcqs_2020_edited.shape)

# print columns
print(mcqs_2017_edited.columns.tolist())
print(mcqs_2018_edited.columns.tolist())
print(mcqs_2019_edited.columns.tolist())
print(mcqs_2020_edited.columns.tolist())

# add year before concatenating
mcqs_2017_edited["Year"] = 2017
mcqs_2018_edited["Year"] = 2018
mcqs_2019_edited["Year"] = 2019
mcqs_2020_edited["Year"] = 2020

merged_edited_mcqs = pd.concat([mcqs_2017_edited, 
                                mcqs_2018_edited, 
                                mcqs_2019_edited,
                                mcqs_2020_edited], ignore_index=True)


# %%
# clean merged MCQs
# 1. typos in subdomain columns
merged_edited_mcqs["Subdomain"] = merged_edited_mcqs["Subdomain"].str.replace("Education Theories", "Education theories")
merged_edited_mcqs["Subdomain"] = merged_edited_mcqs["Subdomain"].str.replace("Classroom Management", "Classroom management")
merged_edited_mcqs["Subdomain"] = merged_edited_mcqs["Subdomain"].str.replace("’", "'")
merged_edited_mcqs["Alternative Subdomain"] = merged_edited_mcqs["Alternative Subdomain"].str.replace("’", "'")
#merged_edited_mcqs["Subdomain.1"] = merged_edited_mcqs["Subdomain.1"].str.replace("’", "'")

# 2. remove white spaces in Correct answer column
merged_edited_mcqs["Correct answer"] = merged_edited_mcqs["Correct answer"].str.strip()

print(merged_edited_mcqs.shape)
merged_edited_mcqs.head(2)

# %%
def plot_value_counts(df, columns, height = 5):
    fig, axs = plt.subplots(len(columns), 1, figsize=(12, height*len(columns)))
    for i, column in enumerate(columns):
        sns.countplot(data=df, y=column, ax=axs[i], color = 'coral', order = df[column].value_counts().index)
        axs[i].set_ylabel(column, fontsize=18)
        axs[i].set_xlabel("Number of MCQs", fontsize=16)
        axs[i].grid(axis='x', linestyle='--', alpha=0.6)

    #plt.suptitle("Value counts of MCQs features", fontsize=18)
    plt.gca().set_axisbelow(True)
    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.show()

# %%
plot_value_counts(merged_edited_mcqs, ["Correct answer", "Category", "Alternative Category", "Subdomain", "Alternative Subdomain", "Age Group", "Year"])


# %%   

# Preprocess the merged edited files without the duplicates
homepath = Path.home()
path_chile_folder = ""

merged_edited_mcqs_wo_dupli = pd.read_excel(path_chile_folder / "merged_edited_mcqs_wo_dupli.xlsx")
print(merged_edited_mcqs_wo_dupli.shape)
merged_edited_mcqs_wo_dupli.head(1)

# %%

plot_value_counts(merged_edited_mcqs_wo_dupli, ["Correct answer", "Category", "Alternative Category", "Subdomain", "Alternative Subdomain", "Subdomain.1", "Age Group", "Year"])
# %%

# add origin of the file short
merged_edited_mcqs_wo_dupli["origin short"] = merged_edited_mcqs_wo_dupli["origin of the file"].apply(lambda x: x.split("\\")[1])

fig, ax = plt.subplots(figsize=(10, 8))
sns.countplot(data = merged_edited_mcqs_wo_dupli, y = "origin short", hue = "Year")
plt.xlabel("Number of MCQs", fontsize=15)
plt.ylabel("Origin of the file", fontsize=15)
plt.title("Number of MCQs per origin file", fontsize=18)

# %%
# understand the different Subdomain columns
hue = "Year"
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
sns.countplot(data = merged_edited_mcqs_wo_dupli, y = "Subdomain", hue = hue, ax = axs[0])
sns.countplot(data = merged_edited_mcqs_wo_dupli, y = "Alternative Subdomain", hue = hue, ax = axs[1])
sns.countplot(data = merged_edited_mcqs_wo_dupli, y = "Subdomain.1", hue = hue, ax = axs[2])
fig.align_ylabels(axs)
plt.tight_layout()
plt.show()

# %%
sns.countplot(data = merged_edited_mcqs_wo_dupli, y = "Alternative Category", hue = "Category")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

sns.countplot(data = merged_edited_mcqs_wo_dupli[merged_edited_mcqs_wo_dupli["Category"] == "Religion PCK"], y = "origin of the file")
plt.show()

# %% Remove Religion PCK and Special Educational Needs
merged_edited_mcqs_finale = merged_edited_mcqs_wo_dupli[merged_edited_mcqs_wo_dupli["Category"] != "Religion PCK"].reset_index(drop=True)
merged_edited_mcqs_finale = merged_edited_mcqs_finale[merged_edited_mcqs_finale["Alternative Category"] != "Special Educational Needs"].reset_index(drop=True)

print(merged_edited_mcqs_finale.shape)
merged_edited_mcqs_finale.head(1)
# %%
plot_value_counts(merged_edited_mcqs_finale, ["Correct answer", "Category", "Alternative Category", "Subdomain", "Alternative Subdomain", "Subdomain.1", "Age Group", "Year"])
# %%
# Remove columns "origin short", "Alternative Category", "Subdomain.1", "Alternative Subdomain" because less than 5 samples
columns_to_drop = ["origin short", "Alternative Category", "Subdomain.1", "Alternative Subdomain"]
merged_edited_mcqs_finale.drop(columns = columns_to_drop, inplace=True)
print(merged_edited_mcqs_finale.shape)
merged_edited_mcqs_finale.head(1)
# %%
# save
#merged_edited_mcqs_final.to_excel(path_chile_folder / "merged_edited_mcqs_final.xlsx", index=False)
#merged_edited_mcqs_final.to_excel("./data/Chile/merged_edited_mcqs_final.xlsx", index=False)


# %%
# Prepare the csv file for inference
# Columns should be: "Source", "Question", "A", "B", "C", "D", "E", "F", "G", "Correct answer", whatever...

df_inference = pd.read_excel(path_chile_folder / "merged_edited_mcqs_finale.xlsx")
#plot_value_counts(merged_edited_mcqs_finale, ["Correct answer", "Category", "Subdomain", "Age Group", "Year"])

df_inference.insert(0, "Source", "Chile ECEP 17-20")
df_inference.drop(columns = ["Question number"], inplace=True)
# insert the columns "E", "F", "G" with None values
df_inference.insert(6, "Answer E", None)
df_inference.insert(7, "Answer F", None)
df_inference.insert(8, "Answer G", None)

# print columns for check
print(df_inference.columns.tolist())
print(df_inference.shape)
# %%
#df_inference.to_csv(path_chile_folder / "merged_edited_mcqs_final.csv", index=False)
#df_inference.to_csv("./data/Chile/merged_edited_mcqs_final.csv", index=False)
# %%
df_inference['Category'].value_counts().index.tolist()

# %%
# merge years 2017, 2018, 2019, 2020 with 2021, 2022, 2023
homepath = Path.home()
path_chile_folder = ""
#mcqs_17_20 = pd.read_excel(path_chile_folder / "merged_edited_mcqs_final.xlsx")
#mcqs_21_23 = pd.read_excel(path_chile_folder / "set_mcqs_2021_2023_edited.xlsx")
#mcqs_21_23_bis = pd.read_excel(path_chile_folder / "set_mcqs_2021_2023_new_edited.xlsx")
#
#print(mcqs_17_20.shape, mcqs_21_23.shape, mcqs_21_23_bis.shape)
#print(mcqs_17_20.columns.tolist())
#print(mcqs_21_23.columns.tolist())
#print(mcqs_21_23_bis.columns.tolist())
#print(mcqs_17_20.head(1), mcqs_21_23.head(1), mcqs_21_23_bis.head(1))
mcqs_17_23 = pd.read_excel(path_chile_folder / "merged_edited_mcqs_2017_2023.xlsx")
mcqs_17_23_fp = pd.read_excel(path_chile_folder / "fp_duplicates_edited.xlsx")
print(mcqs_17_23.shape, mcqs_17_23_fp.shape)
print(mcqs_17_23.columns.tolist())
print(mcqs_17_23_fp.columns.tolist())
print(mcqs_17_23.head(1), mcqs_17_23_fp.head(1))


# %%
#merged_edited_mcqs = pd.concat([mcqs_17_20, mcqs_21_23, mcqs_21_23_bis], ignore_index=True)
merged_edited_mcqs = pd.concat([mcqs_17_23, mcqs_17_23_fp], ignore_index=True)
merged_edited_mcqs.info()
# %%
# clean merged MCQs
# 1. typos in subdomain columns
merged_edited_mcqs["Subdomain"] = merged_edited_mcqs["Subdomain"].str.replace("Education Theories", "Education theories")
merged_edited_mcqs["Subdomain"] = merged_edited_mcqs["Subdomain"].str.replace("Classroom Management", "Classroom management")
merged_edited_mcqs["Subdomain"] = merged_edited_mcqs["Subdomain"].str.replace("’", "'")
merged_edited_mcqs["Alternative Subdomain"] = merged_edited_mcqs["Alternative Subdomain"].str.replace("’", "'")
#merged_edited_mcqs["Subdomain.1"] = merged_edited_mcqs["Subdomain.1"].str.replace("’", "'")

# 2. remove white spaces in Correct answer column
merged_edited_mcqs["Correct answer"] = merged_edited_mcqs["Correct answer"].str.strip()

# 3. Check duplicates
# look at python file flag_duplicates.py
# %%
plot_value_counts(merged_edited_mcqs, ["Correct answer", "Category", "Alternative Category", "Subdomain", "Alternative Subdomain", "Age Group", "Year"])

# %%
