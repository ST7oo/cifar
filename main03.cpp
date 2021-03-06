

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/ml.hpp>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>

using namespace cv;
using namespace std;

//Ptr<AKAZE> akaze = AKAZE::create();
Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(300);
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
//FlannBasedMatcher matcher;
Ptr<DescriptorExtractor> extractor = xfeatures2d::SurfDescriptorExtractor::create();
//---dictionary size=number of cluster's centroids
int dictionarySize = 600;
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;
BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
BOWImgDescriptorExtractor bowDE(extractor, matcher);
Mat featuresUnclustered;


void read_batch(string filename, vector<Mat> &vec, Mat &label){
    ifstream file (filename.c_str(), ios::binary);
    if (file.is_open())
    {
        int number_of_images = 10000;
        int n_rows = 32;
        int n_cols = 32;
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char tplabel = 0;
            file.read((char*) &tplabel, sizeof(tplabel));
            vector<Mat> channels;
            Mat fin_img = Mat::zeros(n_rows, n_cols, CV_8UC3);
            for(int ch = 0; ch < 3; ++ch){
                Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
                for(int r = 0; r < n_rows; ++r){
                    for(int c = 0; c < n_cols; ++c){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        tp.at<uchar>(r, c) = (int) temp;
                    }
                }
                channels.push_back(tp);
            }
            merge(channels, fin_img);
            vec.push_back(fin_img);
            label.at<float>(i, 0) = (float)tplabel;
        }
    }
}

void preprocess(vector<Mat> &vec){
    for(int i=0; i<vec.size(); i++){
        vector<KeyPoint> keypoints;
        Mat features;
        detector->detectAndCompute(vec[i], noArray(), keypoints, features);
        featuresUnclustered.push_back(features);
    }
}

// reads data and stores in trainX, trainY, testX, testY
void read_CIFAR10(string path, Mat &trainX, Mat &testX, Mat &trainY, Mat &testY){

    // variables
    int num_batches = 1;
    vector< vector<Mat> > batches(num_batches, vector<Mat>());
    vector<Mat> labels(num_batches);
    vector<thread> threads(num_batches);
    vector<Mat> mts(num_batches);
    vector<thread> threads2(num_batches);
    string filename;

    cout << "Reading batches\n";

    for(int i = 0; i < num_batches; i++) {
        if (i == 5) {
            filename = path + "test_batch.bin";
        } else {
            filename = path + "data_batch_" + to_string(i+1) + ".bin";
        }
        labels[i] = Mat::zeros(10000, 1, CV_32F);
        threads[i] = thread(read_batch, filename, ref(batches[i]), ref(labels[i]));
    }
    for(int i = 0; i < num_batches; i++) {
        threads[i].join();
    }
//    namedWindow("img0",WINDOW_NORMAL);
//    namedWindow("img1",WINDOW_NORMAL);
//    namedWindow("img2",WINDOW_NORMAL);
//    imshow("img0",batches[0][0]);
//    imshow("img1",batches[0][1]);
//    imshow("img2",batches[0][2]);
//    waitKey();

    cout << "Processing\n";

//    for(int i = 0; i < num_batches; i++) {
//        preprocess(ref(batches[i]));
//    }
//    preprocess(ref(batches[0]));
    cout<<batches[0].size()<<endl;
    for(int i=0; i<batches[0].size(); i++){
        vector<KeyPoint> keypoints;
        Mat features;
        Mat gray;
        cvtColor(batches[0][i], gray, CV_RGB2GRAY);
        detector->detectAndCompute(gray, noArray(), keypoints, features);
        featuresUnclustered.push_back(features);
//        if(i<5) {
//            cout<<keypoints.size()<<endl;
//        }
    }
    cout<<"Clustering "<<featuresUnclustered.size()<<" features"<<endl;
    Mat dictionary = bowTrainer.cluster(featuresUnclustered);
    //store the vocabulary
//    FileStorage fs("dictionary.yml", FileStorage::WRITE);
//    fs << "vocabulary" << dictionary;
//    fs.release();
    // read vocabulary
//    Mat dictionary;
//    FileStorage fs("dictionary.yml", FileStorage::READ);
//    fs["vocabulary"] >> dictionary;
//    fs.release();

    bowDE.setVocabulary(dictionary);

    cout<<"extracting histograms in the form of BOW for each image "<<endl;
	Mat trainingData(0, dictionarySize, CV_32FC1);
	int k=0;
	vector<KeyPoint> keypoint1;
	Mat bowDescriptor1;

	for(int i=0; i<batches[0].size(); i++){
        Mat gray;
        cvtColor(batches[0][i], gray, CV_RGB2GRAY);
        detector->detect(gray, keypoint1);
        if(i<10){
            cout<<keypoint1.size()<<endl;
            /*for(int j=0; j<keypoint1.size(); j++){
                cout<<(string)keypoint1[j]<<endl;
            }*/
        }
		bowDE.compute(gray, keypoint1, bowDescriptor1);
        trainingData.push_back(bowDescriptor1);
//        if(i<5) {
//            cout<<bowDescriptor1.size()<<endl;
//            cout<<bowDescriptor1<<endl;
//        }
	}
	cout<<trainingData.size()<<endl;
//	cout<<trainingData(Rect(0,0,64,5))<<endl;

//    cout<<labels[0].size()<<endl;
//    Y.convertTo(Y,CV_32S);
    /*Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::RBF);
    svm->setGamma(0.5);
    svm->setC(50);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingData, ml::ROW_SAMPLE, labels[0]);*/

    /*for(int i = 0; i < 15; i++) {
        Mat res;
        Mat sample = X.row(i);
        model->predict(sample, res);
        cout << Y.at<float>(i) << " : " << res.at<float>(0) << endl;
    }*/


    /*cout << "Finishing\n";

    for(int i = 0; i < num_batches-1; i++) {
        Rect roi = cv::Rect(0, mts[i].rows * i, trainX.cols, mts[i].rows);
        Mat subView = trainX(roi);
        mts[i].copyTo(subView);
        roi = cv::Rect(0, labels[i].rows * i, 1, labels[i].rows);
        subView = trainY(roi);
        labels[i].copyTo(subView);
    }
    mts[num_batches-1].copyTo(testX);
    labels[num_batches-1].copyTo(testY);*/
}


Ptr<ml::SVM> trainSVM(Mat X, Mat Y) {
//    Ptr<ml::SVM> svm = ml::StatModel::load<ml::SVM>("svm.yml");
    Y.convertTo(Y,CV_32S);
    Ptr<ml::SVM> svm = ml::SVM::create();
//    svm->setType(ml::SVM::C_SVC);
//    svm->setKernel(ml::SVM::RBF);
//    svm->setGamma(10);
//    svm->setC(15);
//    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(X, ml::ROW_SAMPLE, Y);
//    svm->save("svm.yml");
    return svm;
}

// evaluates the prediction
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
    assert(predicted.rows == actual.rows);
    int t = 0;
    int f = 0;
    for(int i = 0; i < actual.rows; i++) {
        float p = predicted.at<float>(i,0);
        float a = actual.at<float>(i,0);
        if(p == a) {
            t++;
        } else {
            f++;
        }
    }
    return (t * 1.0) / (t + f);
}

void predictSVM(Ptr<ml::SVM> model, Mat X, Mat Y) {
    for(int i = 0; i < 15; i++) {
        Mat res;
        Mat sample = X.row(i);
        model->predict(sample, res);
        cout << Y.at<float>(i) << " : " << res.at<float>(0) << endl;
    }

    Mat predicted(Y.rows, 1, CV_32F);
    model->predict(X, predicted);
    cout << "Accuracy = " << evaluate(predicted, Y) << endl;
}

// calculates the elapsed time
string elapsed_time(std::chrono::steady_clock::time_point start_time, std::chrono::steady_clock::time_point end_time) {
    float elapsed = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1000000.0;
    if (elapsed > 60) {
        elapsed /= 60;
        return to_string(elapsed) + " min";
    } else {
        return to_string(elapsed) + " sec";
    }
}

int main()
{
    cout << "CIFAR10\n\n";

    // Variables
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    Mat trainX = Mat::zeros(50000, 256, CV_32F);
    Mat testX = Mat::zeros(10000, 256, CV_32F);
    Mat trainY = Mat::zeros(50000, 1, CV_32F);
    Mat testY = Mat::zeros(10000, 1, CV_32F);
    string path_data = "../cifar-10-batches-bin/";


    // Reading
    cout << "\nStart reading:\n";

    start_time = std::chrono::steady_clock::now();
    read_CIFAR10(path_data, trainX, testX, trainY, testY);
    end_time = std::chrono::steady_clock::now();

    cout << "Reading completed in " << elapsed_time(start_time, end_time) << endl;

/*
    // Training
    cout << "\nStart training:\n";

    start_time = std::chrono::steady_clock::now();

//    ann = trainANN(trainX, trainY);
    Ptr<ml::SVM> svm = trainSVM(trainX, trainY);

    end_time = std::chrono::steady_clock::now();

    cout << "Training completed in " << elapsed_time(start_time, end_time) << endl;


    // Predicting
    cout << "\nStart predicting:\n";

    start_time = std::chrono::steady_clock::now();

    predictSVM(svm, testX, testY);

    end_time = std::chrono::steady_clock::now();

    cout << "Predicting completed in " << elapsed_time(start_time, end_time) << endl;
*/

    waitKey();
    return 0;
}

