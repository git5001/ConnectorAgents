import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pydantic import BaseModel, EmailStr, Field

from atomic_agents.lib.base.base_tool import BaseToolConfig

from AgentFramework.ConnectedAgent import ConnectedAgent



class EmailAgentConfig(BaseToolConfig):
    """Configuration for Secure Email Sending."""
    smtp_server: str = Field(..., description="SMTP server address")
    smtp_port: int = Field(465, description="SMTP server port (default: 465 for SSL/TLS)")
    sender_email: EmailStr = Field(..., description="Sender's email address")
    password: str = Field(..., description="SMTP authentication password")
    html: bool = Field(False, description="HTML formatted email")


class EmailMessageInput(BaseModel):
    """Schema for an email message."""
    recipient_email: EmailStr = Field(..., description="Recipient's email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body content")


class EmailMessageOutput(BaseModel):
    """Schema for email sending results."""
    success: bool = Field(..., description="Indicates whether the email was sent successfully")
    message: str = Field(..., description="Success or error message")


class EmailAgent(ConnectedAgent):
    """
    Secure Email Agent using SSL/TLS (Port 465).
    Supports sending plaintext or HTML emails via SMTP.

    Attributes:
        smtp_server (str): SMTP server address.
        smtp_port (int): SMTP server port for SSL/TLS.
        sender_email (EmailStr): Sender's email address.
        password (str): Authentication password for SMTP.
        html (bool): Whether to send emails in HTML format.
    """
    input_schema = EmailMessageInput
    output_schema = EmailMessageOutput

    def __init__(self, config: EmailAgentConfig) -> None:
        """
        Initializes the Secure Email Agent.

        Args:
            config (EmailAgentConfig): Configuration containing SMTP server details.
        """
        super().__init__(config)
        self.smtp_server: str = config.smtp_server
        self.smtp_port: int = config.smtp_port
        self.sender_email: EmailStr = config.sender_email
        self.password: str = config.password
        self.html: bool = config.html

    def send_email(self, email_message: EmailMessageInput) -> EmailMessageOutput:
        """
        Sends an email using SSL/TLS encryption.

        Args:
            email_message (EmailMessageInput): The email details including recipient, subject, and body.

        Returns:
            EmailMessageOutput: Contains success status and a message.
        """
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = email_message.recipient_email
        msg["Subject"] = email_message.subject

        if self.html:
            msg.attach(MIMEText(email_message.body, "html"))
        else:
            msg.attach(MIMEText(email_message.body, "plain"))

        try:
            # Create a secure SSL context
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                server.login(self.sender_email, self.password)
                server.sendmail(self.sender_email, email_message.recipient_email, msg.as_string())

            return EmailMessageOutput(success=True,
                                      message=f"Email sent successfully to {email_message.recipient_email}")

        except Exception as e:
            return EmailMessageOutput(success=False, message=f"Error sending email: {e}")

    def run(self, params: EmailMessageInput) -> EmailMessageOutput:
        """
        Runs the email agent sending the email.

        Args:
            params (EmailMessageInput): The input parameters for the email.

        Returns:
            EmailMessageOutput: The output containing success status and message.
        """
        return self.send_email(params)  # Direct call, no extra thread