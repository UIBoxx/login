import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Replace these with your email configuration
EMAIL_HOST = "smtp.office365.com"
EMAIL_PORT = 587
EMAIL_USERNAME = ""
EMAIL_PASSWORD = ""
SENDER_EMAIL = ""

def send_reset_email(to_email: str, reset_token: str):
    subject = "Password Reset Request"
    message = f"Click the link below to reset your password:\n\n"
    reset_link = f"http://your-frontend-url/reset-password/{reset_token}"
    message += reset_link

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(message, "plain"))

    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        server.quit()
        print("Reset email sent successfully")
    except Exception as e:
        print("Error sending reset email:", e)

send_reset_email('prabin.bhatt@braintip.ai',"xasdxhkgacvhacxbvac")