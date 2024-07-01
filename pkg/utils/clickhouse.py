import pandas as pd
import clickhouse_connect


class ClickhouseConnector:
    def __init__(
        self,
        user,
        password,
        host='ec2-user@ec2-13-213-34-174.ap-southeast-1.compute.amazonaws.com',
        port=8123,
        database='fyp_rentinsg',
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None

    def connect(self):
        self.connection = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            database=self.database,
            username=self.user,
            password=self.password
        )

    def execute_query(self, query):
        return self.connection.query(query)

    def close(self):
        if self.connection:
            self.connection.close()


if __name__ == "__main__":
    clickhouse = ClickhouseConnector(user="fyp_user", password="fyp2024")
    clickhouse.connect()
    query = "SELECT count() FROM system.tables"
    result = clickhouse.execute_query(query)
    print(result)
    clickhouse.close()
