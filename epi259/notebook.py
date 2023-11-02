import os
import glob

# Specify the root folder where you want to search for .pdf files
root_folder = r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\miglab_asq\data\new_data\Relax Reports'

# Create a list to store the names of .pdf files
pdf_files = []

# Use a recursive search to find .pdf files in the root folder and its subdirectories
for foldername, subfolders, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith('.pdf'):
            pdf_file = os.path.splitext(filename)[0]  # Remove the .pdf extension
            pdf_files.append(pdf_file)

# Define the path for the output .txt file
output_txt_file = r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\Epidemiology_notebook\epi259\new_nox_reports.txt'

# Save the list of PDF filenames without extension to the .txt file
with open(output_txt_file, 'w') as txt_file:
    txt_file.write("\n".join(pdf_files))

print(f"List of PDF filenames without extension saved to {output_txt_file}")