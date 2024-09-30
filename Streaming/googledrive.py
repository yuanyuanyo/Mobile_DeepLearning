import os
import mimetypes
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# 如果修改了范围，需要删除 token.json 重新认证
SCOPES = ['https://www.googleapis.com/auth/drive.file']


def authenticate_google_drive():
    creds = None
    # 如果已经存在 token.json ，则使用它进行身份验证
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # 如果没有（或 token 已失效），需要重新登录授权
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # 将 token 保存到文件中，以便后续使用
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


def upload_file_to_drive(service, file_path, folder_id=None):
    file_name = os.path.basename(file_path)
    mime_type = mimetypes.guess_type(file_path)[0]

    file_metadata = {'name': file_name}
    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(file_path, mimetype=mime_type)

    # 上传文件
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    print(f'File {file_name} uploaded with file ID: {file.get("id")}')


def upload_images_in_folder(folder_path, drive_service, drive_folder_id=None):
    # 仅上传 jpg、png、jpeg、gif 等图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in image_extensions:
            upload_file_to_drive(drive_service, file_path, drive_folder_id)


def main():
    # 第一步：身份验证
    creds = authenticate_google_drive()

    # 第二步：创建 Google Drive 服务对象
    service = build('drive', 'v3', credentials=creds)

    # 你要上传的本地文件夹路径
    local_folder = "output"

    # 如果有特定的Google Drive文件夹ID可以填写在此处, 否则会上传到根目录
    drive_folder_id = None

    # 第三步：上传文件夹中的图片
    upload_images_in_folder(local_folder, service, drive_folder_id)


if __name__ == '__main__':
    main()