import csv


class CsvWriter:

    def __init__(self, fileName):
        self.fileName = fileName

    def write_headers(self, headers):
        with open(self.fileName, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def write_stats(self, stats):
        with open(self.fileName, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(stats)
