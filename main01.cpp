

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml.hpp>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>

using namespace cv;
using namespace std;

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

void preprocess(vector<Mat> &vec, Mat &res){
    int height = vec[0].rows;
    int width = vec[0].cols;
    for(int i=0; i<vec.size(); i++){
        Mat img(height, width, CV_32F);
        Mat gray(height, width, CV_8UC1);
        cvtColor(vec[i], gray, CV_RGB2GRAY);
        gray.convertTo(img, CV_32F);
        Mat ptmat = img.reshape(0, height * width);
        //Rect roi = cv::Rect(0, i, ptmat.rows, ptmat.cols);
        //Mat subView = res(roi);
        //ptmat.copyTo(subView);
        //ptmat.copyTo(res(roi));
        for(int j=0; j<ptmat.cols; j++){
            res.at<float>(j,i) = ptmat.at<float>(j,0);
        }
    }
//    divide(res, 255.0, res);
}

// reads data and stores in trainX, trainY, testX, testY
void read_CIFAR10(string path, Mat &trainX, Mat &testX, Mat &trainY, Mat &testY){

    // variables
    int num_batches = 6;
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


    cout << "Processing\n";

    for(int i = 0; i < num_batches; i++) {
        mts[i] = Mat::zeros(batches[i].size(), batches[i][0].rows * batches[i][0].cols, CV_32F);
        threads2[i] = thread(preprocess, ref(batches[i]), ref(mts[i]));
    }
    for(int i = 0; i < num_batches; i++) {
        threads2[i].join();
    }


    cout << "Finishing\n";

    for(int i = 0; i < num_batches-1; i++) {
        Rect roi = cv::Rect(0, mts[i].rows * i, trainX.cols, mts[i].rows);
        Mat subView = trainX(roi);
        mts[i].copyTo(subView);
        roi = cv::Rect(0, labels[i].rows * i, 1, labels[i].rows);
        subView = trainY(roi);
        labels[i].copyTo(subView);
    }
    mts[num_batches-1].copyTo(testX);
    labels[num_batches-1].copyTo(testY);
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

    // variables
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    Mat trainX = Mat::zeros(50000, 1024, CV_32F);
    Mat testX = Mat::zeros(10000, 1024, CV_32F);
    Mat trainY = Mat::zeros(50000, 1, CV_32F);
    Mat testY = Mat::zeros(10000, 1, CV_32F);
    string path_data = "../cifar-10-batches-bin/";


    cout << "\nStart reading:\n";

    start_time = std::chrono::steady_clock::now();
    read_CIFAR10(path_data, trainX, testX, trainY, testY);
    end_time = std::chrono::steady_clock::now();

    cout << "Reading completed in " << elapsed_time(start_time, end_time) << endl;


    cout << "\nStart training:\n";

    start_time = std::chrono::steady_clock::now();
    // ANN
//    Ptr<ml::ANN_MLP> nn = ml::ANN_MLP::create();
//    nn->setTrainMethod(ml::ANN_MLP::BACKPROP);
//    nn->setBackpropMomentumScale(0.1);
//    nn->setBackpropWeightScale(0.1);
//    nn->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)100, 1e-6));

//    CvTermCriteria criteria;
//    criteria.max_iter = 100;
//    criteria.epsilon = 0.00001f;
//    criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
//    nn->setTermCriteria(criteria);

//    Mat layers = Mat(2, 1, CV_32SC1);
//    layers.row(0) = Scalar(1024);
//    layers.row(1) = Scalar(1);
//    nn->setLayerSizes(layers);
//    nn->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
//    nn->train(trainX, ml::ROW_SAMPLE, trainY);

    // KNN
    /*Ptr<ml::KNearest> knn = ml::KNearest::create();
    knn->setIsClassifier(true);
    knn->setAlgorithmType(ml::KNearest::BRUTE_FORCE);  //BRUTE_FORCE KD_TREE COMPRESSED
    knn->train(trainX, ml::ROW_SAMPLE, trainY);*/

    // SVM
//    cout << trainX(Rect(0,0,1024,10)) << endl;
//    Ptr<ml::SVM> svm = ml::StatModel::load<ml::SVM>("svm.dat");
    Mat trainY2;
    trainY.convertTo(trainY2,CV_32S);
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::RBF);
    svm->setGamma(10);
    svm->setC(15);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    svm->train(trainX, ml::ROW_SAMPLE, trainY2);
    svm->save("svm.dat");

    end_time = std::chrono::steady_clock::now();

    cout << "Training completed in " << elapsed_time(start_time, end_time) << endl;


    cout << "\nStart predicting:\n";

    start_time = std::chrono::steady_clock::now();

    // ANN
    //Mat predicted(testY.rows, 1, CV_32F);
    /*for(int i = 0; i < 10; i++) {
        Mat res;
        Mat sample = testX.row(i);
        nn->predict(sample, res);
        cout << testY.at<float>(i) << " : " << res.at<float>(0) << endl;
        //predicted.at<float>(i,0) = response.at<float>(0,0);
    }*/
    //Mat predicted(testY.rows, 1, CV_32F);
    //nn->predict(testX, predicted);

    // SVM
    for(int i = 0; i < 15; i++) {
        Mat res;
        Mat sample = testX.row(i);
        svm->predict(sample, res);
        cout << res << endl;
        cout << testY.at<float>(i) << " : " << res.at<float>(0) << endl;
    }
    Mat predicted(testY.rows, 1, CV_32F);
    svm->predict(testX, predicted);

    // KNN
	//Mat predicted(testY.rows, 1, CV_32F);
	//knn->findNearest(testX, 7, predicted);
	/*for (int i=0; i<10; i++)
    {
        Mat res;
        knn->findNearest(testX.row(i), 5, noArray(), res);
        float e = testY.at<float>(i);
        float p = res.at<float>(0);
        cerr << e << " : " << p << endl;
    }*/

    /*int c = 0;
    for(int i=0; i<predicted.rows; i++){
        if(predicted.ATD(i,0) == 0.0){
            c++;
        } else {
            //printf("non-zero: %f\n", predicted.ATD(i,0));
        }
    }
    cout << c << endl;*/

	cout << "Accuracy = " << evaluate(predicted, testY) << endl;

    end_time = std::chrono::steady_clock::now();

    cout << "Predicting completed in " << elapsed_time(start_time, end_time) << endl;


    waitKey();
    return 0;
}
