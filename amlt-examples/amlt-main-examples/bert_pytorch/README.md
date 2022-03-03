# Example for distributed finetuning of BERT on AMLK8s

The code and yaml files in this folder show how to fine-tune BERT on AMLK8s.
The example is specific to AMLK8s because checkpoints are saved to fast local network storage (NFS), which is faster than writing to blob storage (particularly if many jobs write concurrently).
See https://aka.ms/amulet for a detailed walkthrough.

##  Data source
This example uses [THUCNews](http://thuctc.thunlp.org/) dataset, which is a publicly available Chinese news corpus launched by the Natural Language Processing Laboratory of Tsinghua University.


## Pre-processing steps
1. Sample 180,000, 10,000, and 10,000 articles from the original corpus for train, dev, and test datasets.
1. Leave articles' title only and remove their contents.
1. Create class.txt for listing all the categories. 
1. Aggregate all the sampled titles and the id of their categories to three files, train.txt, dev.txt, and test.txt.