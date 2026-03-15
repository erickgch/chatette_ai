import datetime
import os
import pickle
import base64

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

# ========================
# Google API scopes
# ========================
SCOPES = [
    "https://www.googleapis.com/auth/calendar",  # read + write (replaces readonly)
    "https://www.googleapis.com/auth/gmail.readonly"
]

CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE")
TOKEN_FILE = os.getenv("GOOGLE_TOKEN_FILE")

EMAIL_DAYS_WINDOW = int(os.getenv("EMAIL_DAYS_WINDOW", 14))
CALENDAR_DAYS_AHEAD = int(os.getenv("CALENDAR_DAYS_AHEAD", 14))
CALENDAR_DAYS_BEHIND = int(os.getenv("CALENDAR_DAYS_BEHIND", 1))


# ========================
# Authentication
# ========================
def get_google_credentials():
    """Authenticate and return Google credentials."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    return creds


# ========================
# Calendar
# ========================
#CHECK EVENTS
def get_upcoming_events(days_ahead: int = None, days_behind: int = None) -> list:
    """Fetch calendar events within a time window."""

    if days_ahead is None:
        days_ahead = CALENDAR_DAYS_AHEAD

    if days_behind is None:
        days_behind = CALENDAR_DAYS_BEHIND

    creds = get_google_credentials()
    service = build("calendar", "v3", credentials=creds)

    now = datetime.datetime.now(datetime.UTC)

    time_min = (now - datetime.timedelta(days=days_behind)).isoformat()
    time_max = (now + datetime.timedelta(days=days_ahead)).isoformat()

    #print(f"📅 Fetching calendar events ({days_behind} days back, {days_ahead} ahead)...")

    events_result = service.events().list(
        calendarId="primary",
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    events = events_result.get("items", [])

    formatted_events = []

    for event in events:

        start = event["start"].get("dateTime", event["start"].get("date"))
        end = event["end"].get("dateTime", event["end"].get("date"))

        formatted_events.append({
            "title": event.get("summary", "No title"),
            "start": start,
            "end": end,
            "description": event.get("description", ""),
            "location": event.get("location", "")
        })

    #print(f"✅ Found {len(formatted_events)} events")

    return formatted_events

#CREATE EVENT
def create_calendar_event(title: str, start_datetime: str, end_datetime: str = None,
                          description: str = "", attendees: list = None) -> str:
    """Create a new event in Google Calendar."""
    creds = get_google_credentials()
    service = build("calendar", "v3", credentials=creds)

    if not end_datetime:
        from datetime import datetime, timedelta
        start_dt = datetime.fromisoformat(start_datetime)
        end_dt = start_dt + timedelta(hours=1)
        end_datetime = end_dt.isoformat()

    # Add Chatette signature to description
    signature = "\n\n---\nEvent created by AI Assistant Chatette"
    full_description = description + signature if description else signature.strip()

    event = {
        "summary": title,
        "description": full_description,
        "start": {
            "dateTime": start_datetime,
            "timeZone": "Europe/Berlin"
        },
        "end": {
            "dateTime": end_datetime,
            "timeZone": "Europe/Berlin"
        }
    }

    if attendees:
        event["attendees"] = [{"email": email} for email in attendees]
        event["guestsCanSeeOtherGuests"] = True

    created_event = service.events().insert(
        calendarId="primary",
        body=event,
        sendUpdates="all"
    ).execute()

    print(f"✅ Event created: {created_event.get('htmlLink')}")
    return created_event.get("id")

# ========================
# Gmail
# ========================
def get_recent_emails(max_results: int = 10, days_window: int = None) -> list:
    """Fetch recent emails within time window."""

    if days_window is None:
        days_window = EMAIL_DAYS_WINDOW

    creds = get_google_credentials()
    service = build("gmail", "v1", credentials=creds)

    after_date = (
        datetime.datetime.now(datetime.UTC) -
        datetime.timedelta(days=days_window)
    )

    after_timestamp = int(after_date.timestamp())

    #print(f"📧 Fetching last {max_results} emails (last {EMAIL_DAYS_WINDOW} days)...")

    results = service.users().messages().list(
        userId="me",
        maxResults=max_results,
        labelIds=["INBOX"],
        q=f"after:{after_timestamp}"
    ).execute()

    messages = results.get("messages", [])

    formatted_emails = []

    for msg in messages:

        msg_detail = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="full"
        ).execute()

        headers = msg_detail["payload"]["headers"]

        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No subject")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")
        date = next((h["value"] for h in headers if h["name"] == "Date"), "Unknown")
        body = ""

        payload = msg_detail["payload"]

        if "parts" in payload:

            for part in payload["parts"]:

                if part["mimeType"] == "text/plain":

                    data = part["body"].get("data")

                    if data:
                        body = base64.urlsafe_b64decode(data).decode(
                            "utf-8",
                            errors="ignore"
                        )
                        break

        elif "body" in payload:

            data = payload["body"].get("data")

            if data:
                body = base64.urlsafe_b64decode(data).decode(
                    "utf-8",
                    errors="ignore"
                )

        body = body.strip()[:1000] if body else "No body content"

        formatted_emails.append({
            "subject": subject,
            "from": sender,
            "date": date,
            "body": body
        })

    #print(f"✅ Found {len(formatted_emails)} emails")

    return formatted_emails


# ========================
# Test
# ========================
if __name__ == "__main__":

    print("Testing Google integration...\n")

    events = get_upcoming_events()

    print("\n📅 Events:")
    for event in events:
        print(f"  - {event['title']} at {event['start']}")

    emails = get_recent_emails(max_results=5)

    print("\n📧 Emails:")
    for email in emails:
        print(f"  - {email['subject']} from {email['from']}")