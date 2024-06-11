c# CV_Final_Global_Motion_Compensation
## png to yuv
### 1. pngtoyuv.py use ```python pngtovideo.py -p ./solution -o ./output.yuv```
### 2. png_test.py use ```python python png_test.py -y output.yuv -o ./solution_new ```
png_test.py會產生129個圖片回來存在./solution_new，你可以再用eval.py去測試psnr是否相同。

## file description (codes that you should download in Common_code directory)
### 1. GMC.py - main func.
### 2. Utils.py - some function to construct the structure.
### 3. Models.py - the motion model.
## run step
### ```python GMC.py```
