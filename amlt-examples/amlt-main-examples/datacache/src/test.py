import os
import argparse


parser = argparse.ArgumentParser()

train = "/tmp/data"

print("Mounted path: {}".format(train))
dirs = os.listdir(train)

print("Files under dir:")
for file in dirs:
    print(file)
