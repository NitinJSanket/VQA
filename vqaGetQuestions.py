# coding: utf-8

from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os
import pprint as pprint

dataDir='../../VQA'
taskType='OpenEnded'
dataType='mscoco' # 'mscoco' for real and 'abstract_v002' for abstract
dataSubType='train2014'
annFile='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
quesFile='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)

# initialize VQA api for QA annotations
vqa=VQA(annFile, quesFile)

# Directly get the questions and write them to a file
NumQues = 1000
Questions = [str(vqa.qqa[a]['question']) for a in range (1,NumQues+1)]


with open("Output.txt", "w") as text_file:
	text_file.write("\n".join(Questions))
