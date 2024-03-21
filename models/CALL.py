# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client

class TwilioCall:
    def call(self):
        # Find your Account SID and Auth Token at twilio.com/console
        # and set the environment variables. See http://twil.io/secure
        account_sid = os.environ['TWILIO_ACCOUNT_SID']
        auth_token = os.environ['TWILIO_AUTH_TOKEN']
        client = Client(account_sid, auth_token)

        call = client.calls.create(
                                url='https://www.dropbox.com/scl/fi/h8gf6jriqdd5cd0pxz418/Voice_call.xml?',
                                to='+31613519355',
                                from_='+18053015048'
                            )

        print(call.sid)