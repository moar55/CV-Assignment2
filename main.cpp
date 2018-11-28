#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
#include "vector"
using namespace cv;

double getVariance(Mat &inputImg, int y , int x) {
    int lowerBoundY = y -1 >=  0? y -1 : y;
    int upperBoundY = y + 1 < inputImg.rows? y + 1 : y;
    int lowerBoundX = x - 1 >= 0? x -1  :  x;
    int upperBoundX = x + 1 < inputImg.cols? x + 1 : x;
    Mat tempMat;
    int length = (upperBoundY - lowerBoundY) + 1;
    int width = (upperBoundX - lowerBoundX) + 1;
    inputImg(Rect(lowerBoundX, lowerBoundY, width, length)).copyTo(tempMat);
    Scalar mean, stddev;
    meanStdDev(tempMat, mean, stddev);
    return stddev.val[0] * stddev.val[0];
}

float calculateNoise(Mat inputImg) {
    float numOfNoisy = 0.0;
    for (int i = 0; i < inputImg.rows; i ++) {
        for (int j = 0; j < inputImg.cols; j++) {
            if ( getVariance(inputImg, i, j)  > 450) {
                numOfNoisy++;
            }
        }
    }
    return  ( numOfNoisy / (inputImg.rows * inputImg.cols) ) * 100;
}

float calculateEdginess(Mat inputImg) {
    float numOfBlurry = 0.0;
    float sum = 0;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    int kernelSize = 3, scale = 1, delta = 0;

    Sobel( inputImg, grad_x, CV_8U, 1, 0, kernelSize, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    Sobel( inputImg, grad_y, CV_8U, 0, 1, kernelSize, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    Mat grad;

    add(abs_grad_x, abs_grad_y, grad);

    for (int i = 0; i < grad.rows; i ++) {
        for (int j = 0; j < grad.cols; j++) {
            sum += (int)(grad.at<uchar>(i,j));
            if ( (int)(grad.at<uchar>(i,j))  < 20) {
                numOfBlurry++;
            }
        }
    }
    return ( numOfBlurry/ (inputImg.rows * inputImg.cols) ) * 100;
}

tuple<bool,float,Mat> removeBlurIfPresent(Mat const& inputImg) {
    float blurPercentage = calculateEdginess(inputImg);
    if (blurPercentage > 68) {
        Mat smoothened;
        GaussianBlur(inputImg, smoothened,Size(23, 23), 5,5);
        Mat edges = inputImg - smoothened;
        Mat sharpened;
        addWeighted(inputImg, 1, edges, 0.7, 0,sharpened);
        return make_tuple(true, blurPercentage, sharpened);
    } else {
        return make_tuple(false, blurPercentage, inputImg);
    }
}

tuple<bool,float,Mat> removeNoiseIfPresent(Mat const& inputImg) {
    Mat smoothened;
    float noisePercentage = calculateNoise(inputImg);
    if (noisePercentage > 50) {
        medianBlur(inputImg, smoothened,5);
        return make_tuple(true, noisePercentage, smoothened);
    }
    return make_tuple(false,noisePercentage,inputImg);
}

tuple<bool,float,Mat> removeCollapsnessIfPresent(Mat const& inputImg) {
    double min, max;

    minMaxLoc(inputImg, &min,&max);
    double spreadnessPercentage = (((max - min) + 1) / 256) * 100;

    if (spreadnessPercentage < 60) {
        double scalingFactor = 256 / ((max - min) + 1);
        double a = 0, c = min;
        Mat uncollapsed = ((inputImg - c) * (scalingFactor) + a)  * 1.15;
        return make_tuple(true, (100 - spreadnessPercentage), uncollapsed);
    }
    return make_tuple(false,(100 - spreadnessPercentage),inputImg);
}

void processImage(Mat const& inputImg, char imageId) {
    tuple<bool,float,Mat> output = removeNoiseIfPresent(inputImg);
    if (get<0>(output)) {
        cout << "Noisy: " << get<1>(output)<<"%"<<endl;
    } else {
        cout<<"No noise!"<< " " << "Noise %: " <<get<1>(output)<<"%"<<endl;
    }

    Mat newImg = get<2>(output);
    tuple<bool,float,Mat> output2= removeBlurIfPresent(newImg);
    if (get<0>(output2)) {
        cout << "Blurry: " << get<1>(output2)<<"%"<<endl;
    } else {
        cout<<"No Blur! "<<"Blur %: "<< get<1>(output2)<<"%"<<endl;
    }

    newImg = get<2>(output2);
    tuple<bool,float,Mat> output3= removeCollapsnessIfPresent(newImg);
    if (get<0>(output3)) {
        cout << "Collapsed: " << get<1>(output3)<<"%"<<endl;
    } else {
        cout<<"No Collapsness! "<<"Collapsness %: "<< get<1>(output3)<<"%"<<endl;
    }
    string fileName = "fixed";
    fileName += imageId;
    fileName += ".jpg";

    imwrite(fileName, get<2>(output3));
    cout << "-----------------------------------------------------------------------------" <<endl;
}

int main() {
    vector<string> files{"1-normal", "2-noisy", "3-blurry", "4-collapsed", "5-blurry_noisy", "6-collapsed_noisy", "7-collapsed_blurry", "8-collapsed_blurry_noisy"};
    auto openAndProcess = [](string name) {cout << name <<endl;processImage(imread(name + ".jpg", 0), name[0]);};
    for_each(files.begin(), files.end(), openAndProcess);
}