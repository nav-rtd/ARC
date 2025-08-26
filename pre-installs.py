#make imports
import os
import gdown
import zipfile
import shutil

ASSIGNMENT_DIR = os.path.dirname(os.path.realpath(__file__))

#make data directory
DATA_DIR = os.path.join(ASSIGNMENT_DIR, "data")
if not(os.path.exists(DATA_DIR)):
    os.mkdir(DATA_DIR)

#make saved_models directory
SAVED_MODELS_DIR = os.path.join(ASSIGNMENT_DIR, "saved_models")
if not(os.path.exists(SAVED_MODELS_DIR)):
    os.mkdir(SAVED_MODELS_DIR)


def download_arc_data():
    """Download arc data into the data directory"""

    #change directory to DATA DIRECTORY
    os.chdir(DATA_DIR)

    #download data using gdown
    id = "11JCXVdq6sEsMnmmFW8xULOwOVFVds5fZ"
    output = "arc_data.zip"
    gdown.download(id = id, output = output)

    #extract the zip file
    TEMP_ZIP_FILE_PATH = os.path.join(DATA_DIR, output)
    with zipfile.ZipFile(TEMP_ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    #remove unnecessary files
    os.remove(TEMP_ZIP_FILE_PATH)
    shutil.rmtree(f"{DATA_DIR}/__MACOSX")

    return


def download_model_checkpoint():
    """Download model checkpoint from gdrive"""

    os.chdir(SAVED_MODELS_DIR)

    #download model checkpoint into the saved_models directory
    id = "1QoZg1oZoKqHbdL9hq6AexKxuSyPns1PB"
    output = "checkpoint-3000.zip"
    gdown.download(id=id, output=output)

    #extract the contents of the checkpoint zip file
    TEMP_ZIP_FILE_PATH = os.path.join(SAVED_MODELS_DIR, output)

    with zipfile.ZipFile(TEMP_ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(SAVED_MODELS_DIR)

    os.remove(TEMP_ZIP_FILE_PATH)
    return


def download_qwen():
    """Download qwen from gdrive"""

    os.chdir(SAVED_MODELS_DIR)

    #download qwen into the saved models directory
    id = "1UbJ9b0dHn8p7lHqVKh47Sfeq7AT_UZ2j"
    output = "Qwen2.5-0.5B-Instruct.zip"
    gdown.download(id = id, output= output)

    #extract the contents of the model zip file
    TEMP_ZIP_FILE_PATH = os.path.join(SAVED_MODELS_DIR, output)

    with zipfile.ZipFile(TEMP_ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(SAVED_MODELS_DIR)

    os.remove(TEMP_ZIP_FILE_PATH)
    return


def download_wheels():
    """Download wheels for the required dependencies"""

    #change into the data dir
    os.chdir(DATA_DIR)

    #populate requirements.txt with the dependencies
    os.system("echo torch > requirements.txt")
    os.system("echo transformers >> requirements.txt")
    os.system("echo peft >> requirements.txt")
    os.system("echo trl >> requirements.txt")
    os.system("echo bitsandbytes >> requirements.txt")
    os.system("echo datasets >> requirements.txt")
    os.system("echo matplotlib >> requirements.txt")
    os.system("echo termcolor >> requirements.txt")
    os.system("echo gdown >> requirements.txt")

    #download the wheels for the dependencies
    os.system("pip download -r requirements.txt")

    return


def install_deps():
    """Install the required dependencies using the downloaded wheels"""

    #build the dependencies
    os.system(f"pip install --no-index --find-links {DATA_DIR} --requirement {DATA_DIR}/requirements.txt")
    return


if __name__ == "__main__":

    #call functions to populate working directory with data and saved models and build the required dependencies
    download_wheels()
    install_deps()
    download_arc_data()
    download_model_checkpoint()
    download_qwen()