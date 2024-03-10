#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

/**
 * @brief The PolygonDraw class is responsible for drawing polygons on an image and saving the coordinates of the drawn points.
 */
class PolygonDraw {
public:
    /**
     * @brief Constructs a PolygonDraw object.
     * 
     * @param imagePath The path to the image file.
     * @param scale The scale factor for displaying the image.
     * @param saveFilePath The path to the file where the coordinates of the drawn points will be saved.
     */
    PolygonDraw(const string& imagePath, float scale, const string& saveFilePath) : imagePath(imagePath), scale(scale), saveFilePath(saveFilePath) {
        image = imread(imagePath);
        if (image.empty()) {
            cout << "Could not read the image: " << imagePath << endl;
            throw runtime_error("Image not found");
        }
        imageCopy = image.clone();
    }


    /**
     * @brief Runs the polygon drawing application.
     */
    void run() {
        namedWindow(windowName, WINDOW_AUTOSIZE);
        setMouseCallback(windowName, onMouse, this);

        while (true) {
            displayImage(); // Ensure the image is displayed before checking for key presses

            char key = waitKey(1);
            if (key == 27) // ESC to exit
                break;
            if (key == 'c') // 'c' to clear
                clearPoints();
            if (key == 's') // 's' to draw and save
                drawAndSave();
        }
    }

private:
    Mat                 image;        /**< The original image. */
    Mat                 imageCopy;    /**< A copy of the image for drawing purposes. */
    vector<Point>       points;       /**< The points of the polygon. */
    string              imagePath;    /**< The path to the image file. */
    float               scale;        /**< The scale factor for displaying the image. */
    string              saveFilePath; /**< The path to the file where the coordinates of the drawn points will be saved. */

    static const string windowName;   /**< The name of the window for displaying the image. */


    /**
     * @brief Mouse event handler for handling mouse clicks on the image window.
     * 
     * @param event The type of mouse event.
     * @param x The x-coordinate of the mouse click.
     * @param y The y-coordinate of the mouse click.
     * @param flags Additional flags.
     * @param userdata User data.
     */
    static void onMouse(int event, int x, int y, int flags, void* userdata) {
        PolygonDraw* editor = reinterpret_cast<PolygonDraw*>(userdata);
        editor->handleMouseEvent(event, x, y, flags);
    }


    /**
     * @brief Handles mouse events.
     * 
     * @param event The type of mouse event.
     * @param x The x-coordinate of the mouse click.
     * @param y The y-coordinate of the mouse click.
     * @param flags Additional flags.
     */
    void handleMouseEvent(int event, int x, int y, int flags) {
        int adjustedX = static_cast<int>(x / scale);
        int adjustedY = static_cast<int>(y / scale);

        if (event == EVENT_LBUTTONDOWN) {
            cout << "Left button clicked - position (" << adjustedX << ", " << adjustedY << ")" << endl;
            points.push_back(Point(adjustedX, adjustedY));
        } else if (event == EVENT_RBUTTONDOWN) {
            cout << "Right button clicked - position (" << adjustedX << ", " << adjustedY << ")" << endl;
        } else if (event == EVENT_MBUTTONDOWN) {
            cout << "Middle button clicked - position (" << adjustedX << ", " << adjustedY << ")" << endl;
        }
    }


    /**
     * @brief Clears the points of the polygon.
     */
    void clearPoints() {
        cout << "Clear" << endl;
        points.clear();
        imageCopy = image.clone();
    }


    /**
     * @brief Draws the polygon on the image and saves the coordinates of the drawn points to a file.
     */
    void drawAndSave() {
        cout << "Draw" << endl;
        if (!points.empty()) {
            polylines(imageCopy, points, true, Scalar(0, 255, 0), 2);

            ofstream myfile(saveFilePath); // 使用保存点的文件路径
            for (const auto& point : points) {
                myfile << (point.x / (float)image.cols) << "," << (point.y / (float)image.rows) << endl;
            }
            myfile.close();
        }
    }


    /**
     * @brief Displays the image with the drawn polygon and the coordinates of the points.
     */
    void displayImage() {
        Mat displayImage = imageCopy.clone();
        for (const auto& point : points) {
            circle(displayImage, point, 5, Scalar(0, 0, 255), -1);
            putText(displayImage, "(" + to_string(point.x) + "," + to_string(point.y) + ")", point, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        }

        Mat resizedImage;
        resize(displayImage, resizedImage, Size(image.cols * scale, image.rows * scale)); // Resize for display purposes
        imshow(windowName, resizedImage);
    }
};

const string PolygonDraw::windowName = "Polygon Draw";

int main(int argc, char** argv) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " [image_path] [scale] [save_file_path]" << endl;
        return -1;
    }

    try {
        string imagePath = argv[1];
        float scale = stof(argv[2]);
        string saveFilePath = argv[3];

        PolygonDraw editor(imagePath, scale, saveFilePath);
        editor.run();

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}
