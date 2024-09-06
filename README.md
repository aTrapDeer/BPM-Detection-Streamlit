# Python BPM Detection System
## Streamlit App - Easy Deployment
** Deploy this to your own AWS account to use it as a Streamlit app. **
### Requirements
- Python 3.x
- Streamlit
- boto3
- numpy
- scipy
- librosa
- pywt
- wave
- array
- math
- tempfile

### Setup
1. Create a new AWS account if you don't already have one.
2. Create a new S3 bucket in the same region as your AWS account.
3. Create a new IAM user with programmatic access and attach the following policy to it:
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::YOUR_BUCKET_NAME",
                "arn:aws:s3:::YOUR_BUCKET_NAME/*"
            ]
        }
    ]
}
```
4. Replace the placeholders in the .env file with your AWS credentials and S3 bucket name.
5. Run the app locally by running the following command in your terminal:
```
streamlit run app.py
```
6. Deploy the app to your AWS account by running the following command in your terminal:
```
streamlit run app.py --server.port 8080 --server.address 127.0.0.1 --server.headless true
```
7. Open your browser and navigate to http://127.0.0.1:8080 to access the app.

### Usage
1. Upload an audio file (MP3, FLAC, or WAV) to detect its BPM.
2. The app will upload the file to your S3 bucket and run the detection algorithm on it.
3. The app will display the detected BPM and provide a link to the file in your S3 bucket.
4. You can also download the file from your S3 bucket by clicking on the link.

### Limitations
- The app is designed to work with audio files that are less than 100 MB in size.
- The app is designed to work with MP3, FLAC, and WAV audio files.

### Future Improvements
- **Creating Pytorch Model For Detection using my dataset**
- Create a more user-friendly interface for the app.
- Add more audio file formats to the app.
- Add more detection algorithms to the app.
- Add more features to the app.
- Add more error handling to the app.
- Add more documentation to the app.
