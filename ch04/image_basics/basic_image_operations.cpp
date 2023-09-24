// basic_image_operations.cpp
// 2023 SEP 23
// Tershire


#include <iostream>

#include <opencv2/opencv.hpp>


int main(int argc, char **argv)
{
    // image //////////////////////////////////////////////////////////////////
    cv::Mat image;

    // load image /////////////////////////////////////////////////////////////
    image = cv::imread(argv[1]);

    // check loading
    if (image.data == nullptr)
    {
        std::cerr << "file" << argv[1] << " does not exist." << std::endl;
        return 0;
    }

    // basic info.
    std::cout << "w: " << image.cols << " == " << image.size().width  << std::endl
              << "h: " << image.rows << " == " << image.size().height << std::endl;
    std::cout << "#channels: " << image.channels() << std::endl;

    // type
    std::cout << "image_type: " << image.type() << std::endl;
    assert(image.type() == CV_8UC1 || image.type() == CV_8UC3);

    // 
    unsigned char* row_ptr;
    unsigned char* pixel_ptr;
    unsigned char gray_level;
    for (size_t y = 0; y < image.rows; ++y)
    {
        row_ptr = image.ptr<unsigned char>(y);

        for (size_t x = 0; x < image.cols; ++x)
        {
            pixel_ptr = &row_ptr[x * image.channels()];

            for (int c = 0; c < image.channels(); ++c)
            {
                gray_level = pixel_ptr[c];
            }
        }
    }

    // print (B, G, R) of the last pixel
    std::cout << "last pixel's (B, G, R) is ";
    std::cout << "(" << int(pixel_ptr[0]) << ", " 
                     << int(pixel_ptr[1]) << ", " 
                     << int(pixel_ptr[2]) << ")" << std::endl;

    cv::imshow("image", image);
    cv::waitKey(0);

    // cv::Mat copy ///////////////////////////////////////////////////////////
    // <M1> operator = --------------------------------------------------------
    // operator = yields reference, not data copy
    cv::Mat image_other = image;

    // changing image_other will change image too
    image_other(cv::Rect(0, 0, 100, 100)).setTo(0);

    cv::imshow("image", image);
    cv::waitKey(0);

    // <M2> clone -------------------------------------------------------------
    // .clone() copies the data
    cv::Mat image_clone = image.clone();

    // changing image_clone will not change image
    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);

    cv::imshow("image", image);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}