import csv
import os


class MetricsFileWriter:
    def __init__(self, output_file, header):
        self.output_file = output_file
        self.header = header
        file_exists = os.path.isfile(self.output_file)
        
        self.fh = open(self.output_file, "a", newline='')
        self.writer = csv.writer(self.fh)
        
        # Only write the header if the file is new
        if not file_exists:
            self.write(self.header)

    def write(self, metrics):
        self.writer.writerow(metrics)
        self.fh.flush()

    def close(self):
        self.fh.close()

    def __del__(self):
        self.fh.close()
