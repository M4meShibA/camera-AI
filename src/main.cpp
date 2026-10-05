#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>

// Windows Socket (Winsock2) 用ヘッダー
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

// ==========================================
// Winsock 通信用関数
// ==========================================

SOCKET connectToPythonServer(const std::string& ip, int port) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return INVALID_SOCKET;
    }

    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        WSACleanup();
        return INVALID_SOCKET;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &serverAddr.sin_addr);

    if (connect(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        closesocket(sock);
        WSACleanup();
        return INVALID_SOCKET;
    }

    std::cout << "[OK] Connected to Python Server (" << ip << ":" << port << ")\n";
    return sock;
}

void sendDetectionData(SOCKET& sock, int objectCount, double inferenceTime, const std::vector<std::string>& detectedClasses) {
    if (sock == INVALID_SOCKET) return;

    std::string classListJson = "[";
    for (size_t i = 0; i < detectedClasses.size(); ++i) {
        classListJson += "\"" + detectedClasses[i] + "\"";
        if (i < detectedClasses.size() - 1) classListJson += ",";
    }
    classListJson += "]";

    std::string jsonPayload = "{\"count\":" + std::to_string(objectCount) +
                              ",\"inference_ms\":" + std::to_string(inferenceTime) +
                              ",\"classes\":" + classListJson + "}\n";

    int bytesSent = send(sock, jsonPayload.c_str(), static_cast<int>(jsonPayload.length()), 0);
    
    if (bytesSent == SOCKET_ERROR) {
        closesocket(sock);
        WSACleanup();
        sock = INVALID_SOCKET;
    }
}

int main()
{
    std::cout << "====================================\n";
    std::cout << "       CAMERA AI - YOLO11n\n";
    std::cout << "====================================\n";

    // ==========================================
    // COCO クラス名
    // ==========================================

    const std::vector<std::string> classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    // ==========================================
    // 設定
    // ==========================================

    const std::string modelPath = "models/yolo11n.onnx";

    constexpr int inputWidth = 640;
    constexpr int inputHeight = 640;

    constexpr float confidenceThreshold = 0.25f;
    constexpr float nmsThreshold = 0.45f;

    // TCP通信設定
    const std::string serverIp = "127.0.0.1";
    const int serverPort = 5001;
    SOCKET pySocket = INVALID_SOCKET;

    pySocket = connectToPythonServer(serverIp, serverPort);

    // ==========================================
    // YOLOモデル読み込み
    // ==========================================

    cv::dnn::Net net =
        cv::dnn::readNetFromONNX(modelPath);

    if (net.empty())
    {
        std::cerr << "ERROR: Failed to load YOLO model.\n";
        return 1;
    }

    std::cout << "[OK] YOLO model loaded.\n";

    net.setPreferableBackend(
        cv::dnn::DNN_BACKEND_OPENCV
    );

    net.setPreferableTarget(
        cv::dnn::DNN_TARGET_CPU
    );

    // ==========================================
    // カメラ
    // ==========================================

    cv::VideoCapture camera(0);

    if (!camera.isOpened())
    {
        std::cerr << "ERROR: Camera could not be opened.\n";
        return 1;
    }

    std::cout << "[OK] Camera opened.\n";
    std::cout << "Press Q or ESC to exit.\n";

    cv::Mat frame;

    // ==========================================
    // FPS計算
    // ==========================================

    auto fpsStart =
        std::chrono::steady_clock::now();

    auto lastSendTime =
        std::chrono::steady_clock::now();

    int frameCount = 0;
    double fps = 0.0;

    // ==========================================
    // メインループ
    // ==========================================

    while (true)
    {
        camera >> frame;

        if (frame.empty())
        {
            std::cerr
                << "ERROR: Failed to capture frame.\n";

            break;
        }

        // Pythonサーバーと未接続の場合、3秒おきに再接続を試行
        if (pySocket == INVALID_SOCKET) {
            auto nowCheck = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(nowCheck - lastSendTime).count() > 3.0) {
                pySocket = connectToPythonServer(serverIp, serverPort);
                lastSendTime = nowCheck;
            }
        }

        // ======================================
        // YOLO推論
        // ======================================

        auto inferenceStart =
            std::chrono::steady_clock::now();

        cv::Mat blob =
            cv::dnn::blobFromImage(
                frame,
                1.0 / 255.0,
                cv::Size(
                    inputWidth,
                    inputHeight
                ),
                cv::Scalar(),
                true,
                false
            );

        net.setInput(blob);

        cv::Mat output =
            net.forward();

        auto inferenceEnd =
            std::chrono::steady_clock::now();

        double inferenceTime =
            std::chrono::duration<double, std::milli>(
                inferenceEnd - inferenceStart
            ).count();

        // ======================================
        // YOLO出力解析
        // ======================================

        cv::Mat detections(
            output.size[1],
            output.size[2],
            CV_32F,
            output.ptr<float>()
        );

        cv::Mat transposed;

        cv::transpose(
            detections,
            transposed
        );

        // ======================================
        // 検出結果格納
        // ======================================

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        float xFactor =
            static_cast<float>(frame.cols) /
            static_cast<float>(inputWidth);

        float yFactor =
            static_cast<float>(frame.rows) /
            static_cast<float>(inputHeight);

        // ======================================
        // 8400候補を解析
        // ======================================

        for (int i = 0;
             i < transposed.rows;
             ++i)
        {
            float centerX =
                transposed.at<float>(i, 0);

            float centerY =
                transposed.at<float>(i, 1);

            float width =
                transposed.at<float>(i, 2);

            float height =
                transposed.at<float>(i, 3);

            cv::Mat scores =
                transposed
                    .row(i)
                    .colRange(4, 84);

            cv::Point classIdPoint;
            double confidence;

            cv::minMaxLoc(
                scores,
                nullptr,
                &confidence,
                nullptr,
                &classIdPoint
            );

            if (confidence <
                confidenceThreshold)
            {
                continue;
            }

            int classId =
                classIdPoint.x;

            if (classId < 0 ||
                classId >=
                    static_cast<int>(
                        classNames.size()
                    ))
            {
                continue;
            }

            int left =
                static_cast<int>(
                    (centerX -
                     width / 2.0f)
                    * xFactor
                );

            int top =
                static_cast<int>(
                    (centerY -
                     height / 2.0f)
                    * yFactor
                );

            int boxWidth =
                static_cast<int>(
                    width * xFactor
                );

            int boxHeight =
                static_cast<int>(
                    height * yFactor
                );

            if (boxWidth <= 0 ||
                boxHeight <= 0)
            {
                continue;
            }

            cv::Rect box(
                left,
                top,
                boxWidth,
                boxHeight
            );

            // 画面内に制限
            box &= cv::Rect(
                0,
                0,
                frame.cols,
                frame.rows
            );

            if (box.width <= 0 ||
                box.height <= 0)
            {
                continue;
            }

            boxes.push_back(box);

            confidences.push_back(
                static_cast<float>(
                    confidence
                )
            );

            classIds.push_back(
                classId
            );
        }

        // ======================================
        // NMS
        // ======================================

        std::vector<int> indices;

        cv::dnn::NMSBoxes(
            boxes,
            confidences,
            confidenceThreshold,
            nmsThreshold,
            indices
        );

        // 検出されたクラス名のリストを収集
        std::vector<std::string> detectedClasses;
        for (int index : indices) {
            detectedClasses.push_back(classNames[classIds[index]]);
        }

        // ======================================
        // TCPデータ送信 (0.5秒間隔で実行)
        // ======================================

        auto nowTime = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(nowTime - lastSendTime).count() >= 0.5) {
            sendDetectionData(pySocket, static_cast<int>(indices.size()), inferenceTime, detectedClasses);
            lastSendTime = nowTime;
        }

        // ======================================
        // 表示用画像
        // ======================================

        cv::Mat displayFrame =
            frame.clone();

        // ======================================
        // Bounding Box + ラベル
        // ======================================

        for (int index : indices)
        {
            if (index < 0 ||
                index >=
                    static_cast<int>(
                        boxes.size()
                    ))
            {
                continue;
            }

            cv::Rect box =
                boxes[index];

            int classId =
                classIds[index];

            float confidence =
                confidences[index];

            std::string label =
                classNames[classId] +
                " " +
                cv::format(
                    "%.2f",
                    confidence
                );

            // Bounding Box
            cv::rectangle(
                displayFrame,
                box,
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                2
            );

            // ラベル
            cv::putText(
                displayFrame,
                label,
                cv::Point(
                    box.x,
                    std::max(
                        20,
                        box.y - 5
                    )
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                2
            );
        }

        // ======================================
        // FPS更新
        // ======================================

        frameCount++;

        auto now =
            std::chrono::steady_clock::now();

        double elapsed =
            std::chrono::duration<double>(
                now - fpsStart
            ).count();

        if (elapsed >= 1.0)
        {
            fps =
                static_cast<double>(
                    frameCount
                ) / elapsed;

            frameCount = 0;

            fpsStart = now;
        }

        // ======================================
        // 情報表示
        // ======================================

        std::string fpsText =
            "FPS: " +
            cv::format(
                "%.1f",
                fps
            );

        std::string inferenceText =
            "Inference: " +
            cv::format(
                "%.1f ms",
                inferenceTime
            );

        std::string objectsText =
            "Objects: " +
            std::to_string(
                indices.size()
            );

        cv::putText(
            displayFrame,
            fpsText,
            cv::Point(
                20,
                35
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(
                0,
                255,
                0
            ),
            2
        );

        cv::putText(
            displayFrame,
            inferenceText,
            cv::Point(
                20,
                70
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(
                0,
                255,
                0
            ),
            2
        );

        cv::putText(
            displayFrame,
            objectsText,
            cv::Point(
                20,
                105
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(
                0,
                255,
                0
            ),
            2
        );

        // TCP接続ステータス表示
        std::string sockStatus = (pySocket != INVALID_SOCKET) ? "TCP: Connected" : "TCP: Disconnected";
        cv::Scalar sockColor = (pySocket != INVALID_SOCKET) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(
            displayFrame,
            sockStatus,
            cv::Point(
                20,
                140
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            sockColor,
            2
        );

        // ======================================
        // 画面表示
        // ======================================

        cv::imshow(
            "Camera AI - YOLO11n",
            displayFrame
        );

        // ======================================
        // キー入力
        // ======================================

        int key =
            cv::waitKey(1);

        if (key == 'q' ||
            key == 'Q' ||
            key == 27)
        {
            break;
        }
    }

    // ==========================================
    // 終了処理
    // ==========================================

    if (pySocket != INVALID_SOCKET) {
        closesocket(pySocket);
        WSACleanup();
    }

    camera.release();

    cv::destroyAllWindows();

    std::cout
        << "CAMERA AI END\n";

    return 0;
}

// g++ src/main.cpp -o CameraAI.exe $(pkg-config --cflags --libs opencv5) -lws2_32
// ./CameraAI.exe