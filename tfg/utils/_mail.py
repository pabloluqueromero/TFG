
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime
import smtplib
import io

'''Class to send CSV by email to store results'''

class EmailSendCSV:
    def __init__(self, sender, receiver, password):
        self.sender_email = sender
        self.receiver_email = receiver
        self.password = password

    def send(self, subject, df, filename):
        # Create a multipart message
        msg = MIMEMultipart()

        msg['Subject'] = subject
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email

        stream = io.StringIO(df.to_csv(index=False, sep=","))
        msg.attach(MIMEApplication(stream.read(), Name=filename))

        # Create SMTP object
        smtp_obj = smtplib.SMTP("smtp.gmail.com", 587)
        # Login to the server
        smtp_obj.ehlo()
        smtp_obj.starttls()
        smtp_obj.login(self.sender_email, self.password)
        # Convert the message to a string and send it
        smtp_obj.sendmail(msg['From'], msg['To'], msg.as_string())
        smtp_obj.quit()

    def send_test(self):
        msg = MIMEMultipart()
        msg['Subject'] = "TEST-EMAIL"
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email

        # Create SMTP object
        smtp_obj = smtplib.SMTP("smtp.gmail.com", 587)
        # Login to the server
        smtp_obj.ehlo()
        smtp_obj.starttls()
        smtp_obj.login(self.sender_email, self.password)
        # Convert the message to a string and send it
        smtp_obj.sendmail(msg['From'], msg['To'], msg.as_string())
        smtp_obj.quit()


def send_results(algorithm, email_data, result):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M")
    email = EmailSendCSV(email_data["FROM"], email_data["TO"], email_data["PASSWORD"])
    email.send(f"[{algorithm}] -{dt_string}- {email_data['TITLE']}", result, email_data["FILENAME"])
