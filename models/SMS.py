# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client

class TwilioSMS:
    def sms(self):
        # Find your Account SID and Auth Token at twilio.com/console
        # and set the environment variables. See http://twil.io/secure
        account_sid = os.environ['TWILIO_ACCOUNT_SID']
        auth_token = os.environ['TWILIO_AUTH_TOKEN']
        client = Client(account_sid, auth_token)

        message = client.messages \
                        .create(
                            body="Emergency message: A fall is detected in your house! Name: Johnny. Date of Birth: April 4, 1950. Location: 5612 LW, Eindhoven.",
                            from_='+18053015048',
                            to='+31613519355'
                        )

        print(message.sid)