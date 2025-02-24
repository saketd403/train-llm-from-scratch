import gdown

# Replace with the Google Drive file ID or full link
file_id = '1i8eeP79dN2TwIK7H4qr_Y-ji1cB19SMU'
url = f'https://drive.google.com/uc?id={file_id}&export=download'

# Specify the output path where you want to save the file
output_path = './downloaded_file.zip'

# Download the file
gdown.download(url, output_path, quiet=False)