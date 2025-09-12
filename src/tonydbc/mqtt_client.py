import datetime
import json
import uuid
from typing import Any

from paho.mqtt import client as mqtt

MAX_RECONNECT_ATTEMPTS = 10

RECONNECT_ON_FAILURE = True


class MQTTClient:
    """An MQTT client context manager"""

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        subscribed_topics: list[str],
        client_id: str = "MQTTClient_name",
        port: int = 1883,
    ) -> None:
        self.MQTT_host = host
        self.MQTT_user = user
        self.MQTT_password = password
        self.MQTT_port = int(port)
        self.MQTT_subscribed_topics = subscribed_topics
        self.MQTTClient_id = f"{str(client_id)}_{str(uuid.uuid4())[:4]}"

        assert len(self.MQTT_subscribed_topics) > 0

    def __enter__(self) -> "MQTTClient":
        print(f"MQTT client_id {self.MQTTClient_id}: __enter__")
        print(f"Subscribed topics: {self.MQTT_subscribed_topics}")

        # Create MQTT connection
        # NOTE: clean_session=False means it will be a "durable client";
        #       that is, messages will be saved if the client disconnects
        self.__mqtt_client = mqtt.Client(
            client_id=self.MQTTClient_id,
            clean_session=False,
            reconnect_on_failure=RECONNECT_ON_FAILURE,
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

    def __exit__(
        self, exit_type: type | None, value: BaseException | None, traceback: Any | None
    ) -> None:
        print(f"EXIT DIPS_MQTT_Logger {self.MQTTClient_id}")
        self.__mqtt_client.disconnect()

    def loop_forever(self) -> None:
        """Start a blocking form of the network loop.
        Automatically handles reconnect.

        Note: as a blocking function, this function
        will NOT return until the client calls disconnect().
        """
        self.__mqtt_client.loop_forever()

    def publish(self, topic: str, message: str, verbose: bool = False) -> None:
        # Publish the message
        res = self.__mqtt_client.publish(topic, message)
        if res.rc == mqtt.MQTT_ERR_SUCCESS:
            if verbose:
                print(f"MQTT publish on topic {topic} succeeded.")
        else:
            raise IOError(f"MQTT publish on topic {topic} failed with rc={res.rc}.")

    def on_connect(
        self, client: mqtt.Client, userdata: Any, flags: dict, rc: int
    ) -> None:
        # The callback for when the client receives a
        # CONNACK ("Connection Acknowledge") response from the server.
        msg = f"{str(self.now_utc)} | {self.MQTTClient_id} | INFO | Connected to MQTT server: "
        if rc != mqtt.MQTT_ERR_SUCCESS:
            raise IOError(msg + f"failed with rc={rc}")

        print(msg + "success")

        # Subscribe to topics AFTER a successful connection (or reconnection)
        for topic in self.MQTT_subscribed_topics:
            self.__mqtt_client.subscribe(topic)

    def on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        assert client == self.__mqtt_client
        if rc == mqtt.MQTT_ERR_SUCCESS:
            print("Disconnection (expected) of with rc = mqtt.MQTT_ERR_SUCCESS.")
        if rc != 0:
            if not RECONNECT_ON_FAILURE:
                raise IOError(
                    f"Unexpected disconnection rc={rc}.  Not reconnecting manually since RECONNECT_ON_FAILURE=True"
                )
            else:
                print(
                    f"Unexpected disconnection rc={rc}.  Hopefully reconnection will occur automatically."
                )

    def on_message(
        self, client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage
    ) -> None:
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
            "topic": message.topic,
            "message": message_payload,
        }

        print(
            f"{self.now_utc_string} | {self.MQTTClient_id} received message with topic {message.topic}: {res_message}"
        )

        if message_payload is None:
            print(f"{self.now_utc_string} | ERROR | No message received??")

    @property
    def now_utc(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc)

    @property
    def now_utc_string(self) -> str:
        return self.now_utc.strftime("%Y-%m-%dT%H:%M:%S%z")
