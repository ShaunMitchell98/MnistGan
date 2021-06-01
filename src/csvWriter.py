import csv
import os


class CsvWriter:

    def __init__(self, settings, fileName):
        self.fileName = os.path.join(settings.Output, fileName)

    def write_headers(self, headers):
        with open(self.fileName, 'w+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def write_object(self, objectInput):

        values = []

        for attr, value in objectInput.__dict__.items():
            values.append(value)

        self.write_array(values)

    def write_array(self, arrayInput):

        with open(self.fileName, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(arrayInput)
