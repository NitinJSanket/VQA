Visit http://www.visualqa.org/ for the base code.
Follow the instructions and setup the environment. (https://github.com/VT-vision-lab/VQA)
Note that we doo not need the images for this code and we can skip downloading the images.

Download the Code from this git and place it in the folder ./PythonHelperTools
For convenience I have placed the Questions.txt and the mat files with the labels.

Run the vqaGetQuestions.py (Same dependencies as VQA demo code), this will generate a Questions.txt.

Then we can run the PredictCommonObjectQuestion.m and uncomment the lines to label the questions. This has already been done so one can just run the code to see the output.

External Resources Used:
(1) VQA Codebase: https://github.com/VT-vision-lab/VQA
(2) Porter Stemmer: http://tartarus.org/martin/PorterStemmer/matlab.txt
(3) Liblinear: https://www.csie.ntu.edu.tw/~cjlin/liblinear/
 (for faster SVM)
(4) LibSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/