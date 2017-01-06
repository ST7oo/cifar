

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml.hpp>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>

#define ATD at<double>

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
            label.ATD(i, 0) = (double)tplabel;
        }
    }
}

void concatenateMat(vector<Mat> &vec, Mat &res){
    int height = vec[0].rows;
    int width = vec[0].cols;
    for(int i=0; i<vec.size(); i++){
        Mat img(height, width, CV_64FC1);
        Mat gray(height, width, CV_8UC1);
        cvtColor(vec[i], gray, CV_RGB2GRAY);
        gray.convertTo(img, CV_64FC1);
        Mat ptmat = img.reshape(0, height * width);
        Rect roi = cv::Rect(0, i, ptmat.rows, ptmat.cols);
        Mat subView = res(roi);
        ptmat.copyTo(subView);
    }
    divide(res, 255.0, res);
}

void read_CIFAR10(string path, Mat &trainX, Mat &testX, Mat &trainY, Mat &testY){

    cout << "Reading batches\n";

    int num_batches = 6;
    vector< vector<Mat> > batches(num_batches, vector<Mat>());
    vector<Mat> labels(num_batches);
    vector<thread> threads(num_batches);
    string filename;

    for(int i = 0; i < num_batches; i++) {
        if (i == 5) {
            filename = path + "test_batch.bin";
        } else {
            filename = path + "data_batch_" + to_string(i+1) + ".bin";
        }
        labels[i] = Mat::zeros(10000, 1, CV_64FC1);
        threads[i] = thread(read_batch, filename, ref(batches[i]), ref(labels[i]));
    }

    for(int i = 0; i < num_batches; i++) {
        threads[i].join();
    }
    cv::Size sb = batches[0][0].size();
    cv::Size sl = labels[0].size();


    cout << "Processing\n";

    vector<Mat> mts(num_batches);
    vector<thread> threads2(num_batches);

    for(int i = 0; i < num_batches; i++) {
        mts[i] = Mat::zeros(batches[i].size(), batches[i][0].rows * batches[i][0].cols, CV_64FC1);
        threads2[i] = thread(concatenateMat, ref(batches[i]), ref(mts[i]));
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

/*void train() {
    // Train the ANN
    //! [init]
    Ptr< ANN_MLP >  nn = ANN_MLP::create();
    nn->setTrainMethod(cv::ml::ANN_MLP::BACKPROP);
    nn->setBackpropMomentumScale(0.1);
    nn->setBackpropWeightScale(0.1);
    nn->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)100000, 1e-6));

    //setting the NN layer size
    cv :: Mat layers = cv :: Mat (4 , 1 , CV_32SC1 );
    layers . row (0) = cv :: Scalar (4) ;
    layers . row (1) = cv :: Scalar (4) ;
    layers . row (2) = cv :: Scalar (4) ;
    layers . row (3) = cv :: Scalar (1) ;
    nn->setLayerSizes(layers);
    nn->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);

    nn->train(trainingData, ROW_SAMPLE, trainingClasses);

    Mat predicted(testingClasses.rows, 1, CV_32F);
    nn->predict(testingData, predicted);

    cout<<"predict: " << endl << predicted << endl;

	//cout << "Accuracy_{MLP} = " << evaluate(predicted, testClasses) << endl;
}*/

float evaluate(cv::Mat& predicted, cv::Mat& actual) {
    assert(predicted.rows == actual.rows);
    int t = 0;
    int f = 0;
    for(int i = 0; i < actual.rows; i++) {
        float p = predicted.ATD(i,0);
        float a = actual.ATD(i,0);
        if(p == a) {
            t++;
        } else {
            f++;
        }
    }
    return (t * 1.0) / (t + f);
}

int main()
{
    cout << "CIFAR10\n\n";
    Mat trainX, testX;
    Mat trainY, testY;
    trainX = Mat::zeros(50000, 1024, CV_32FC1);
    testX = Mat::zeros(10000, 1024, CV_32FC1);
    trainY = Mat::zeros(50000, 1, CV_32FC1);
    testY = Mat::zeros(10000, 1, CV_32FC1);
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    float elapsed_time;

    cout << "\nStart reading:\n";
    start_time = std::chrono::steady_clock::now();
    string path_data = "../cifar-10-batches-bin/";
    read_CIFAR10(path_data, trainX, testX, trainY, testY);
    end_time = std::chrono::steady_clock::now();
    elapsed_time = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1000000.0;
    printf("Reading completed in %f seconds.\n", elapsed_time);

    cout << "\nStart training:\n";
    start_time = std::chrono::steady_clock::now();
    // Train the ANN
    Ptr<ml::ANN_MLP> nn = ml::ANN_MLP::create();
    nn->setTrainMethod(ml::ANN_MLP::BACKPROP);
    nn->setBackpropMomentumScale(0.1);
    nn->setBackpropWeightScale(0.1);
    nn->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)100000, 1e-6));

    //setting the NN layer size
    Mat layers = Mat(3, 1, CV_32SC1);
    layers.row(0) = Scalar(1024);
    layers.row(1) = Scalar(8);
    layers.row(2) = Scalar(1);
    nn->setLayerSizes(layers);
    nn->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);

    nn->train(trainX, ml::ROW_SAMPLE, trainY);
    end_time = std::chrono::steady_clock::now();
    elapsed_time = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1000000.0;
    printf("Training completed in %f seconds.\n", elapsed_time);

    cout << "\nStart predicting:\n";
    start_time = std::chrono::steady_clock::now();
    cv::Mat response(1, 1, CV_32FC1);
    cv::Mat predicted(testY.rows, 1, CV_32F);
    for(int i = 0; i < testX.rows; i++) {
        cv::Mat response(1, 1, CV_32FC1);
        cv::Mat sample = testX.row(i);
        nn->predict(sample, response);
        if(i<2){
            for(int j=0; j<10; j++){
                printf("%f,",sample.ATD(j,0));
            }
            cout << response << endl;
        }
        predicted.at<float>(i,0) = response.at<float>(0,0);

    }
    //Mat predicted(testY.rows, 1, CV_32F);
    //nn->predict(testX, predicted);
    int c = 0;
    for(int i=0; i<predicted.rows; i++){
        if(predicted.ATD(i,0) == 0.0){
            c++;
        }
    }
    cout << c << endl;
	cout << "Accuracy_{MLP} = " << evaluate(predicted, testY) << endl;
    end_time = std::chrono::steady_clock::now();
    elapsed_time = (std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1000000.0;
    printf("Predicting completed in %f seconds.\n", elapsed_time);

    waitKey();
    return 0;
}



//#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

//using namespace cv; // all the new OpenCV API is put into "cv" namespace. Export its content
//using namespace std;

/*
void help()
{
	cout <<
	"\nThis program shows how to use OpenCV.\n"
	"It shows reading of image and visualize it on the screen\n"
	"Call:\n"
	"./helloWorld [image-name Default: lena.jpg]\n" << endl;
}


int main( int argc, char** argv )
{
	help();
    const char* imagename = argc > 1 ? argv[1] : "lena.jpg";

    Mat img = imread(imagename); // this function is in accordance MATLAB-style function
    if(img.empty())
    {
        fcout << stderr, "Can not load image %s\n", imagename);
        return -1;
    }
    if( !img.data ) // check if the image has been loaded properly
        return -1;

    string str ="image loaded: ";
    str += imagename;
    imshow(str, img); //function to show an image

    waitKey(); //waiting until the user press a key
    return 0;
    // all the memory will automatically be released!!
}
*/
