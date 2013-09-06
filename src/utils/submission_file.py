import csv

class SubmissionFile:
  # create a new instance of the submission file to the given path
  def __init__(self, path, contents, headers=None, index=True):
    self.path = path
    self.contents = contents
    self.headers = headers
    self.index = index

  # create and write the file
  def Write(self):
    with open(self.path, "wb") as csvfile:
      writer = csv.writer(csvfile)
      if self.headers != None:
        writer.writerow(self.headers)
      
      i = 1
      for row in self.contents:
        if self.index:
          writer.writerow((i, row))
          i += 1
        else:
          writer.writerow(row)
