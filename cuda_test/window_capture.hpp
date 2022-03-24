#pragma once

#include <Windows.h>
#include <opencv2/opencv.hpp>

struct WindowCapturer {
    HDC hwindowDC_, hwindowCompatibleDC_;
    int height_, width_, srcheight_, srcwidth_;

    HBITMAP hbwindow_;
    BITMAPINFOHEADER  bi_;

    HWND hwnd_;

    void set_target(HWND hwnd) {
        hwnd_ = hwnd;

        hwindowDC_ = GetDC(hwnd_);
        hwindowCompatibleDC_ = CreateCompatibleDC(hwindowDC_);
        SetStretchBltMode(hwindowCompatibleDC_, COLORONCOLOR);

        RECT windowsize;    // get the height and width of the screen
        GetClientRect(hwnd_, &windowsize);

        srcheight_ = windowsize.bottom;
        srcwidth_ = windowsize.right;
        height_ = windowsize.bottom;  //change this to whatever size you want to resize to
        width_ = windowsize.right;

        // create a bitmap
        hbwindow_ = CreateCompatibleBitmap(hwindowDC_, width_, height_);
        bi_.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
        bi_.biWidth = width_;
        bi_.biHeight = -height_;  //this is the line that makes it draw upside down or not
        bi_.biPlanes = 1;
        bi_.biBitCount = 24;
        bi_.biCompression = BI_RGB;
        bi_.biSizeImage = 0;
        bi_.biXPelsPerMeter = 0;
        bi_.biYPelsPerMeter = 0;
        bi_.biClrUsed = 0;
        bi_.biClrImportant = 0;

        // use the previously created device context with the bitmap
        SelectObject(hwindowCompatibleDC_, hbwindow_);
    }

    void capture(cv::Mat& out) {
        // copy from the window device context to the bitmap device context
        if (out.rows != height_ || out.cols != width_ || out.type() != CV_8UC3)
            return;
        StretchBlt(hwindowCompatibleDC_, 0, 0, width_, height_, hwindowDC_, 0, 0, srcwidth_, srcheight_, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
        GetDIBits(hwindowCompatibleDC_, hbwindow_, 0, height_, out.data, (BITMAPINFO*)&bi_, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow

    }

    ~WindowCapturer() {
        DeleteObject(hbwindow_); 
        DeleteDC(hwindowCompatibleDC_); 
        ReleaseDC(hwnd_, hwindowDC_);
    }

};
