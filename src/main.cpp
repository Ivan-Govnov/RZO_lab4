#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// функция определения типа фигуры по контуру
string getShapeName(const vector<Point>& contour) {
    double peri = arcLength(contour, true);
    vector<Point> approx;
    // аппроксимация контура многоугольником
    approxPolyDP(contour, approx, 0.02 * peri, true);

    int vertices = approx.size();

    // треугольник
    if (vertices == 3)
        return "Triangle";

    // четырёхугольник
    if (vertices == 4) {
        Rect rect = boundingRect(approx);
        double aspect = static_cast<double>(rect.width) / rect.height;
        // Допуск для квадрата
        if (aspect >= 0.9 && aspect <= 1.1)
            return "Square";
        else
            return "Rectangle";
    }

    // окружность
    double area = contourArea(contour);
    if (peri > 0) {
        double circularity = 4 * M_PI * area / (peri * peri);
        if (circularity > 0.75)
            return "Circle";
    }

    // остальные фигуры
    return "Polygon";
}

int main() {
    setlocale(LC_ALL, "ru_RU.UTF-8");

    Mat img = imread("Lab4_image.png");
    if (img.empty()) {
        cerr << "Не удалось загрузить изображение!" << endl;
        return -1;
    }

    // преобразование
    Mat gray, blurred, edges, binary;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    
    // детектор границ Canny
    Canny(blurred, edges, 50, 150);
    
    // морфологическое закрытие для устранения разрывов
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(edges, binary, MORPH_CLOSE, kernel);

    // поиск контуров
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat result = img.clone();

    cout << "Найдено контуров (до фильтрации): " << contours.size() << endl;

    int shapeCount = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        // фильтр по минимальной площади
        if (area < 100.0)
            continue;

        // периметр
        double perimeter = arcLength(contours[i], true);
        
        // центр масс через моменты
        Moments m = moments(contours[i]);
        Point2f center(static_cast<float>(m.m10 / m.m00),
                       static_cast<float>(m.m01 / m.m00));

        // тип фигуры
        string shape = getShapeName(contours[i]);

        char label[100];
        snprintf(label, sizeof(label), "%s\nP=%.1f", shape.c_str(), perimeter);

        // контур
        drawContours(result, contours, static_cast<int>(i), Scalar(0, 255, 0), 2);
        // центр масс
        circle(result, center, 5, Scalar(0, 0, 255), -1);
        circle(result, center, 5, Scalar(255, 255, 255), 2);
        
        int font = FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
        int baseline = 0;
        Size textSize = getTextSize(label, font, fontScale, thickness, &baseline);
        Point textOrg(center.x - textSize.width / 2, center.y - textSize.height / 2 - 5);
        putText(result, label, textOrg, font, fontScale, Scalar(0, 0, 0), thickness);

        cout << "Фигура " << ++shapeCount << ": " << shape
             << " | Периметр = " << perimeter
             << " | Центр = (" << center.x << ", " << center.y << ")" << endl;
    }

    cout << "Всего распознано фигур: " << shapeCount << endl;

    imwrite("result.png", result);
    imwrite("edges.png", edges);
    
    double scale = 0.7;
    Mat dispOrig, dispEdges, dispRes;
    resize(img, dispOrig, Size(), scale, scale);
    resize(edges, dispEdges, Size(), scale, scale);
    resize(result, dispRes, Size(), scale, scale);
    
    imshow("Original", dispOrig);
    imshow("Edges (Canny)", dispEdges);
    imshow("Result", dispRes);
    waitKey(0);

    return 0;
}