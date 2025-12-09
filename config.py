"""
Configuration for Alert System
Edit these values with your own credentials.
"""

# Email Alert Configuration
EMAIL_CONFIG = {
    "enabled": True,  # Set to True to enable email alerts
    "sender_email": "jashkaransaini10@gmail.com",  # Your Gmail address
    "sender_password": "pssl dumv rxco fohm",  # Gmail App Password (16 characters)
    "recipient_emails": ["jashkarnsingh10@gmail.com"],  # Who to notify
}


# Slack Configuration
SLACK_CONFIG = {
    "enabled": False,  # Set to True to enable Slack alerts
    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",  # Your Slack Webhook URL
}

# Default Detection Settings
DETECTION_CONFIG = {
    "confidence_threshold": 0.75,
    "imgsz": 320,
    "device": "auto",
    "frame_skip": 3,
}

# Alert Settings
ALERT_CONFIG = {
    "store_name": "Home",
    "alert_threshold": 1,  # Alert after N suspicious events in 1 minute
    "alert_cooldown": 90,  # Seconds between alerts (prevent spam)
}
