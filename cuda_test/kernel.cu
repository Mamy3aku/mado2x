#if 1
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <SDL.h>

#include <cudnn.h>
#include "waifu2x.cuh"

#include <opencv2/opencv.hpp>

#include <vector>
#include <algorithm>
#include <chrono>

#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>

#include <Windows.h>
#include <WinUser.h>

using namespace std;

class Timer {
    using clock = chrono::high_resolution_clock;
    clock::time_point startTime;

    template <class T>
    auto elapsed() const {
        return chrono::duration_cast<T>(clock::now() - startTime).count();
    }

public:
    Timer() { restart(); }

    void restart() { startTime = clock::now(); }

    auto milli() const { return elapsed<chrono::milliseconds>(); }

    auto micro() const { return elapsed<chrono::microseconds>(); }
};

__global__ void leakyRelu_(float* vec, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        vec[i] = 0.1f * fminf(vec[i], 0.f) + fmaxf(vec[i], 0.f);
}

HWND GetWindowHandleByPID(const DWORD targetPID)
{
    HWND hWnd = GetTopWindow(NULL);
    do {
        if (GetWindowLong(hWnd, GWLP_HWNDPARENT) != 0 || !IsWindowVisible(hWnd)) {
            continue;
        }
        DWORD getPID;
        GetWindowThreadProcessId( hWnd, &getPID);
        if (targetPID == getPID) {
            return hWnd;
        }
    } while((hWnd = GetNextWindow( hWnd, GW_HWNDNEXT)) != NULL);
     
    return NULL;
}

cv::Mat hwnd2mat(HWND hwnd){
    HDC hwindowDC,hwindowCompatibleDC;

    int height,width,srcheight,srcwidth;
    HBITMAP hbwindow;
    cv::Mat src;
    BITMAPINFOHEADER  bi;

    hwindowDC=GetDC(hwnd);
    hwindowCompatibleDC=CreateCompatibleDC(hwindowDC);
    SetStretchBltMode(hwindowCompatibleDC,COLORONCOLOR);  

    RECT windowsize;    // get the height and width of the screen
    GetClientRect(hwnd, &windowsize);

    srcheight = windowsize.bottom;
    srcwidth = windowsize.right;
    height = windowsize.bottom;  //change this to whatever size you want to resize to
    width = windowsize.right;

    src.create(height,width,CV_8UC4);

    // create a bitmap
    hbwindow = CreateCompatibleBitmap( hwindowDC, width, height);
    bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
    bi.biWidth = width;    
    bi.biHeight = -height;  //this is the line that makes it draw upside down or not
    bi.biPlanes = 1;    
    bi.biBitCount = 32;    
    bi.biCompression = BI_RGB;    
    bi.biSizeImage = 0;  
    bi.biXPelsPerMeter = 0;    
    bi.biYPelsPerMeter = 0;    
    bi.biClrUsed = 0;    
    bi.biClrImportant = 0;

    // use the previously created device context with the bitmap
    SelectObject(hwindowCompatibleDC, hbwindow);
    // copy from the window device context to the bitmap device context
    StretchBlt( hwindowCompatibleDC, 0,0, width, height, hwindowDC, 0, 0,srcwidth,srcheight, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
    GetDIBits(hwindowCompatibleDC,hbwindow,0,height,src.data,(BITMAPINFO *)&bi,DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow

    // avoid memory leak
    DeleteObject (hbwindow); DeleteDC(hwindowCompatibleDC); ReleaseDC(hwnd, hwindowDC);

    return src;
}

void waifu2xDiff(
    cv::Mat const& original, cv::Mat const& original_waifu2xed, 
    cv::Mat const& target, cv::Mat& out,
    int const block_h, int const block_w
) {
    std::vector<std::vector<bool>> map(original.rows, std::vector<bool>(original.cols, false));
    for (int y = 0; y < original.rows; y++) {
        auto original_row_ptr = original.ptr(y);
        auto target_row_ptr = target.ptr(y);
        for (int x = 0; x < original.cols; x++) {
            for (int c = 0; c < 3; c++) {
                int const index = 3 * x + c;
                if (original_row_ptr[index] != target_row_ptr[index])
                    map[y][x] = true;
            }
        }
    }

    int nDiffPixels = 0;
    for (int y = 0; y < map.size(); y++)
        nDiffPixels += std::count(map[y].begin(), map[y].end(), true);

    out = original_waifu2xed.clone();
    while (nDiffPixels > 0) {


    }
}

template<class Numeric>
Numeric clamp(Numeric a, Numeric l, Numeric u) {
    return std::max(l, std::min(a, u));
}

int main(int argc, char* argv[])
{
    try {
        int version = -1;
        cudaRuntimeGetVersion(&version);
        mypv(version);

        mypv(cudnnGetVersion());

        int deviceCount = -1;
        cudaGetDeviceCount(&deviceCount);
       
        std::cerr << argv[2] << std::endl;
        HWND hwnd = GetWindowHandleByPID(std::stoi(argv[2]));

        cudnnHandle_t cudnn_handle = nullptr;
        cudnnCheck(cudnnCreate(&cudnn_handle));

        CNNModel cn(argv[1]);
        cv::Mat image = hwnd2mat(hwnd);

        cv::Mat copied;
        cv::copyMakeBorder(image, copied, 7, 7, 7, 7, cv::BORDER_REPLICATE);

        size_t workspace_size = 0;
        size_t image_size = 3 * sizeof(float) * copied.rows * copied.cols;

        cudnnTensorDescriptor_t tensor0 = nullptr, tensor1 = nullptr;

        cudnnCheck(cudnnCreateTensorDescriptor(&tensor0));
        cudnnCheck(cudnnCreateTensorDescriptor(&tensor1));
        cudnnCheck(cudnnSetTensor4dDescriptor(tensor0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, copied.rows, copied.cols));

        int h = copied.rows, w = copied.cols;
        for (int i = 0; i < cn.layers.size(); i++) {
            cn.layers[i].diff(h, w);
            cudnnCheck(cudnnSetTensor4dDescriptor(tensor1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, cn.layers[i].nOutputPlane_, h, w));
            workspace_size = std::max(workspace_size, cn.layers[i].algorithm_workspace(cudnn_handle, tensor0, tensor1));
            image_size = std::max(image_size, sizeof(float) * cn.layers[i].nOutputPlane_ * copied.rows * copied.cols);

            std::swap(tensor0, tensor1);
        }

        auto workspace = cuda_memory_allocate(workspace_size);

        auto image0 = cuda_memory_allocate(image_size);
        auto image1 = cuda_memory_allocate(image_size);
        mypv(image.cols);
        mypv(image.rows);

        SDL_Init(SDL_INIT_EVERYTHING);
        SDL_Window* win;
        SDL_Renderer* renderer;
        int ret = SDL_CreateWindowAndRenderer(image.cols * 2, image.rows * 2, SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI, &win, &renderer);

        float ddpi, vdpi, hdpi;
        SDL_GetDisplayDPI(SDL_GetWindowDisplayIndex(win), &ddpi, &hdpi, &vdpi);
        std::cerr << ddpi << ' ' << hdpi << ' ' << vdpi << std::endl;

        SDL_GetRendererOutputSize(renderer, &w, &h);
        std::cerr << w << ' ' << h << std::endl;

        SDL_DisplayMode dm;
        SDL_GetDisplayMode(0, 0, &dm);
        std::cerr << dm.w << "x" << dm.h << "@" << dm.refresh_rate << std::endl;

        auto texture = SDL_CreateTexture(renderer, SDL_PixelFormatEnum::SDL_PIXELFORMAT_ARGB8888,
            SDL_TextureAccess::SDL_TEXTUREACCESS_STREAMING, image.cols * 2, image.rows * 2);

        SDL_Event ev;
        bool running = true;
        while (running) {
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_QUIT)
                    running = false;
            }
            image = hwnd2mat(hwnd);
            array<double, 4> mean = {};
            for (int y = 0; y < image.rows; y++) {
                uchar* row_ptr = image.ptr(y);
                for (int x = 0; x < image.cols; x++) {
                    for (int c = 0; c < 4; c++) {
                        mean[c] += row_ptr[4 * x + c];
                    }
                    row_ptr[4 * x + 3] = 255;
                }
            }

            cv::copyMakeBorder(image, copied, 7, 7, 7, 7, cv::BORDER_REPLICATE);

            std::vector<float> image_float(3 * copied.rows * copied.cols);
            for (int y = 0; y < copied.rows; y++) {
                uchar const* row_ptr = copied.ptr(y);
                for (int x = 0; x < copied.cols; x++) {
                    size_t const channel_size = copied.rows * copied.cols;
                    image_float[0 * channel_size + y * copied.cols + x] = row_ptr[4 * x + 2] / 255.f;
                    image_float[1 * channel_size + y * copied.cols + x] = row_ptr[4 * x + 1] / 255.f;
                    image_float[2 * channel_size + y * copied.cols + x] = row_ptr[4 * x + 0] / 255.f;
                }
            }

            Timer t;
            int act_time = 0;
            cudaCheck(cudaMemcpy(image0.get(), image_float.data(), 
                sizeof(float) * image_float.size(), cudaMemcpyKind::cudaMemcpyHostToDevice));

            h = copied.rows, w = copied.cols;

            int conv_time = 0;
            Timer ttt;
            cudnnCheck(cudnnSetTensor4dDescriptor(tensor0, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, copied.rows, copied.cols));
            for (int i = 0; i < cn.layers.size(); i++) {
                cn.layers[i].diff(h, w);
                cudnnCheck(cudnnSetTensor4dDescriptor(tensor1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, cn.layers[i].nOutputPlane_, h, w));

                Timer timer_iter;
                cn.layers[i].execute(cudnn_handle, tensor0, image0.get(), tensor1, image1.get(), workspace.get(), workspace_size);

                if (cn.layers[i].direction_ == CNNModel::Layer::Direction::Forward) {
                    Timer tt;
                    int const size = cn.layers[i].nOutputPlane_ * h * w;
                    int const blockSize = 2;
                    dim3 dimBlock(blockSize);
                    dim3 dimGrid(std::ceil(size / (float)(blockSize)));
                    leakyRelu_ << <dimGrid, dimBlock >> > ((float*)(image1.get()), size);
                    act_time += tt.micro();
                }
                std::swap(tensor0, tensor1);
                std::swap(image0, image1);
                mypv(timer_iter.micro());
            }

            image_float.resize(4 * 3 * image.rows * image.cols * 5);
            cudaCheck(cudaMemcpy(image_float.data(), image0.get(), sizeof(float) * image_float.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
            conv_time += ttt.micro();


            uchar* pixels;
            int pitch;
            SDL_LockTexture(texture, nullptr, (void**)(&pixels), &pitch);
            double b_mean = 0.0;
            double r_mean = 0.0;
            double g_mean = 0.0;
            for (int y = 0; y < 2 * image.rows; y++) {
                uchar* row_ptr = pixels + pitch * y;
                size_t channel_size = 4 * image.rows * image.cols;
                for (int x = 0; x < 2 * image.cols; x++) {
                    uchar const r = clamp(int(image_float[0 * channel_size + y * 2 * image.cols + x] * 255.f + 0.5f), 0, 255);
                    uchar const g = clamp(int(image_float[1 * channel_size + y * 2 * image.cols + x] * 255.f + 0.5f), 0, 255);
                    uchar const b = clamp(int(image_float[2 * channel_size + y * 2 * image.cols + x] * 255.f + 0.5f), 0, 255);
                    b_mean += b;
                    r_mean += r;
                    g_mean += g;
                    row_ptr[4 * x + 0] = b;
                    row_ptr[4 * x + 1] = g;
                    row_ptr[4 * x + 2] = r;
                    row_ptr[4 * x + 3] = 255;
                }
            }
            SDL_UnlockTexture(texture);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);
            SDL_RenderPresent(renderer);

            cv::Mat output_image(image.rows * 2, image.cols * 2, CV_8UC3);
            for (int y = 0; y < output_image.rows; y++) {
                uchar* row_ptr = output_image.ptr(y);
                size_t channel_size = output_image.rows * output_image.cols;
                for (int x = 0; x < output_image.cols; x++) {
                    row_ptr[3 * x + 0] = clamp(int(image_float[2 * channel_size + y * output_image.cols + x] * 255.f + 0.5f), 0, 255);
                    row_ptr[3 * x + 1] = clamp(int(image_float[1 * channel_size + y * output_image.cols + x] * 255.f + 0.5f), 0, 255);
                    row_ptr[3 * x + 2] = clamp(int(image_float[0 * channel_size + y * output_image.cols + x] * 255.f + 0.5f), 0, 255);
                }
            }


            mypv(t.micro());
            mypv(conv_time);
        }
    }
    catch (std::exception const& exc) {
        std::cerr << exc.what() << std::endl;

    }

    return 0;

}
#endif
