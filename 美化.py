# app.py
import base64
import os
import random
import shutil
from collections import OrderedDict

import numpy as np
import pandas as pd
import streamlit as st
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pyuul import VolumeMaker
from pyuul import utils
from sklearn.cluster import KMeans

# Set a random seed
random_seed = 100
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True

# Constants for file paths
UPLOAD_FOLDER_PATH = r"C:\Users\ouc2023\Desktop\stream\lig"
TMP_FOLDER_PATH = r"C:\Users\ouc2023\Desktop\stream\tmp"
PDB_FOLDER_PATH = r"C:\Users\ouc2023\Desktop\stream\pdb"
RESULT_CSV_PATH = r"C:\Users\ouc2023\Desktop\stream\kmeans-2.csv"

# Device selection
device = "cuda"


# Helper functions

def list_files_in_folder(folder_path):
    return os.listdir(folder_path)


def add_files_to_folder(source_folder, destination_folder):
    file_list = os.listdir(source_folder)
    for filename in file_list:
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, destination_folder)


def copy_files(files, folder):
    all_files = os.listdir(folder)
    for filename in all_files:
        if filename not in files:
            file_path = os.path.join(folder, filename)
            os.remove(file_path)


def pyuul(folder, n_clusters):
    coords, atname, pdbname, pdb_num = utils.parsePDB(folder)
    atoms_channel = utils.atomlistToChannels(atname)
    radius = utils.atomlistToRadius(atname)

    PointCloudSurfaceObject = VolumeMaker.PointCloudVolume(device=device)
    coords = coords.to(device)
    radius = radius.to(device)
    atoms_channel = atoms_channel.to(device)

    SurfacePoitCloud = PointCloudSurfaceObject(coords, radius)
    feature = SurfacePoitCloud.view(pdb_num, -1).cpu()

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, init="k-means++", random_state=random_seed)
    y = kmeans.fit_predict(feature)

    pairs = zip(pdbname, y)
    result_dict = {key: value for key, value in pairs}
    ligend_class = result_dict['lig.pdb']

    sheet = []
    for key, value in result_dict.items():
        if value == ligend_class:
            sheet.append(key)
    return sheet


def kmeans_clustering(ligend_path, peptide_folder_path, pdb_folder, n_clusters, n_num):
    for i in range(1, n_num + 1):
        if i == 1:
            add_files_to_folder(ligend_path, pdb_folder)
            add_files_to_folder(peptide_folder_path, pdb_folder)
            output = pyuul(pdb_folder, n_clusters)
            copy_files(output, pdb_folder)
        else:
            if pdb_folder:
                output = pyuul(pdb_folder, n_clusters)
                copy_files(output, pdb_folder)

    data = OrderedDict()
    data['Name'] = output
    data = pd.DataFrame(data)
    data.to_csv(RESULT_CSV_PATH, index=False)


def get_binary_file_download_link(file_path, file_name):
    with open(file_path, "rb") as f:
        file_contents = f.read()
    b64 = base64.b64encode(file_contents).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">点击这里下载</a>'
def clear_folder(folder_path):
    file_list = os.listdir(folder_path)
    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Streamlit app
def main():
    st.title("文件夹文件下载示例")
    st.write("欢迎使用文件夹文件下载功能！在这里，您可以下载指定文件夹中的文件。")

    file_list = list_files_in_folder(PDB_FOLDER_PATH)

    if not file_list:
        st.warning("文件夹中没有文件。")
        return

    # Sidebar for file search and upload
    st.sidebar.subheader("搜索文件")
    search_term = st.sidebar.text_input("输入文件名关键字：")

    st.sidebar.subheader("上传文件")
    uploaded_files = st.sidebar.file_uploader("选择要上传的文件", accept_multiple_files=True)

    # Main content area
    st.subheader("文件列表")

    filtered_files = [filename for filename in file_list if search_term.lower() in filename.lower()]

    if not filtered_files:
        st.warning("找不到匹配的文件。")
        return

    files_per_page = 5
    num_pages = (len(filtered_files) - 1) // files_per_page + 1
    page_number = st.number_input("选择页码", min_value=1, max_value=num_pages, value=1)
    start_idx = (page_number - 1) * files_per_page
    end_idx = min(start_idx + files_per_page, len(filtered_files))

    for idx in range(start_idx, end_idx):
        filename = filtered_files[idx]
        st.write(f"**文件名:** {filename}")
        seq = filename[0:3]
        protein_analysis = ProteinAnalysis(seq)
        molecular_weight = protein_analysis.molecular_weight()
        st.write(f"**molecular_weight:** {molecular_weight}")
        isoelectric_point = protein_analysis.isoelectric_point()
        st.write(f"**isoelectric_point:** {isoelectric_point}")
        gravy_score = protein_analysis.gravy()
        st.write(f"**gravy_score:** {gravy_score}")

        file_path = os.path.join(PDB_FOLDER_PATH, filename)
        st.write(get_binary_file_download_link(file_path, filename), unsafe_allow_html=True)
        st.write("\n")

    if uploaded_files:

        if not os.path.exists(TMP_FOLDER_PATH):
            os.makedirs(TMP_FOLDER_PATH)

        for file in uploaded_files:
            file_path = os.path.join(TMP_FOLDER_PATH, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

        num_clusters = st.number_input("输入聚类簇的数量:", min_value=2, step=1, value=2)
        num_iterations = st.number_input("输入聚类迭代的次数:", min_value=1, step=1, value=2)

        if st.button("开始聚类", key="start_clustering"):
            kmeans_clustering(UPLOAD_FOLDER_PATH, PDB_FOLDER_PATH, TMP_FOLDER_PATH, num_clusters, num_iterations)
            if os.path.exists(RESULT_CSV_PATH):
                st.subheader("K-means Clustering Results")
                st.markdown(get_binary_file_download_link(RESULT_CSV_PATH, "kmeans-2.csv"), unsafe_allow_html=True)

    # Delete the temporary folder after processing
    clear_folder(TMP_FOLDER_PATH)


if __name__ == "__main__":
    main()
