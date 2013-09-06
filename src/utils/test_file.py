import csv
import numpy as np

class TestFile:
  # create a new instance of the test file from the given path
  def __init__(self, path, contain_header=True):
    self.path = path
    self.contain_header = contain_header
    self.headers = []
    self.data = []

  # reading the csv file from the given path. 
  # the file can contain header
  def Read(self):
    header_handled = False

    with open(self.path, "r") as csvfile:
      rows = csv.reader(csvfile, delimiter=",")
      for row in rows:
        if self.contain_header and not header_handled:
          self.headers = list(row);
          header_handled = True
          continue

        r = [float(x) for x in row[:]]
        self.data.append(r)
