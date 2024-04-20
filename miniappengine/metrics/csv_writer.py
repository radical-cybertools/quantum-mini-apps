import csv


class MetricsFileWriter:
    def __init__(self, output_file, header):
        self.output_file = output_file
        self.header = header
        self.fh = open(self.output_file, "w", newline='')
        self.writer = csv.writer(self.fh)
        self.write(self.header)

    def write(self, metrics):
        self.writer.writerow(metrics)

    def close(self):
        self.fh.close()

    def __del__(self):
        self.fh.close()
