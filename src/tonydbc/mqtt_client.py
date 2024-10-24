import os
import sys
import json
import datetime
from paho.mqtt import client as mqtt


class MQTTClient:
    """An MQTT client context manager"""

    def __init__(
        self,
        host,
        user,
        password,
        subscribed_topics,
        client_id="MQTTClient_name",
        port=1883,
    ):
        self.MQTT_host = host
        self.MQTT_user = user
        self.MQTT_password = password
        self.MQTT_port = int(port)
        self.MQTT_subscribed_topics = subscribed_topics
        self.MQTTClient_id = client_id

        assert len(self.MQTT_subscribed_topics) > 0

    def __enter__(self):
        print(f"MQTT client_id {self.MQTTClient_id}: __enter__")
        print(f"Subscribed topics: {self.MQTT_subscribed_topics}")

        # Create MQTT connection
        # NOTE: clean_session=False means it will be a "durable client";
        #       that is, messages will be saved if the client disconnects
        self.__mqtt_client = mqtt.Client(
            client_id=self.MQTTClient_id, clean_session=False, reconnect_on_failure=True
        )

        # Attach callback functions
        self.__mqtt_client.on_connect = self.on_connect
        self.__mqtt_client.on_message = self.on_message
        self.__mqtt_client.on_disconnect = self.on_disconnect
        self.__mqtt_client.username_pw_set(
            self.MQTT_user,
            password=self.MQTT_password,
        )

        # Connect to MQTT broker
        self.__mqtt_client.connect(self.MQTT_host, port=self.MQTT_port, keepalive=30)

        return self

    def __exit__(self, exit_type, value, traceback):
        print(f"EXIT DIPS_MQTT_Logger {self.MQTTClient_id}")
        self.__mqtt_client.disconnect()

    def loop_forever(self):
        """Start a blocking form of the network loop.
        Automatically handles reconnect.

        Note: as a blocking function, this function
        will NOT return until the client calls disconnect().
        """
        self.__mqtt_client.loop_forever()

    def publish(self, topic, message):
        if not self.__mqtt_client.is_connected():
            print("MQTT client not connected. Attempting to reconnect.")
            try:
                self.__mqtt_client.reconnect()
            except Exception as e:
                print(f"Failed to reconnect: {e}")
                return  # Exit the method if reconnection fails

        # Publish the message
        res = self.__mqtt_client.publish(topic, message)
        if res.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"MQTT Publish failed with rc={res.rc}. Attempting to reconnect.")
            try:
                self.__mqtt_client.reconnect()
                # Try publishing again
                res = self.__mqtt_client.publish(topic, message)
                if res.rc != mqtt.MQTT_ERR_SUCCESS:
                    print(f"MQTT Publish failed again after reconnecting. rc={res.rc}")
                else:
                    print("MQTT Publish succeeded after reconnecting.")
            except Exception as e:
                print(f"MQTT Reconnect failed: {e}")
        else:
            print("MQTT Publish succeeded.")

    def on_connect(self, client, userdata, flags, rc):
        # The callback for when the client receives a
        # CONNACK ("Connection Acknowledge") response from the server.
        msg = f"{str(self.now_utc)} | {self.MQTTClient_id} | INFO | Connecting to MQTT server: "
        if rc == 0:
            print(msg + "success")

            # Subscribe to topics AFTER a successful connection (or reconnection)
            for topic in self.MQTT_subscribed_topics:
                self.__mqtt_client.subscribe(topic)
        else:
            raise AssertionError(msg + "failed")

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print(f"Unexpected disconnection. rc={rc}. Attempting to reconnect.")
            try:
                client.reconnect()
            except Exception as e:
                print(f"Reconnect failed: {e}")

    def on_message(self, client, userdata, message):
        """Receive MQTT message

        This method should be overridden by a child class

        The message parameter is an instance of the MQTTMessage class,
        which contains the fields:
            topic, payload, qos, retain, mid, properties, and timestamp.
            ("qos" is quality of service, 0, 1, or 2)
            (topic will be like "7904023020504888772/status" aka "{uid}/status")
        """
        message_payload = json.loads(message.payload)

        res_message = {
            "timestamp": self.now_utc_string,
            "topic": topic,
            "message": message_payload,
        }

        print(
            f"{self.now_utc_string} | {self.MQTTClient_id} received message with topic {message.topic}: {res_message}"
        )

        if message_payload is None:
            print(f"{self.now_utc_string} | ERROR | No message received??")

    @property
    def now_utc(self):
        return datetime.datetime.now(datetime.timezone.utc)

    @property
    def now_utc_string(self):
        return self.now_utc.strftime("%Y-%m-%dT%H:%M:%S%z")
