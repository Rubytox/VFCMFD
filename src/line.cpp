#include "../include/line.hpp"

#include <utility>

using namespace std;
using namespace cv;
using namespace defals;

Line::Line(InterestPoint  point1, InterestPoint point2) : _point1(move(point1)), _point2(move(point2))
{
    computeCartes();
    computePolar();

    _origineOrtho = Point2f(0, 0);
    _pointOrtho = ortho();
}

Line::Line(const Line& other) : Line(other._point1, other._point2)
{
    computeCartes();
    computePolar();

    _origineOrtho = Point2f(0, 0);
    _pointOrtho = ortho();
}

double Line::distanceOrigine() const
{
    return abs(_c) / sqrt(_a * _a + _b * _b);
}

void Line::draw(Mat& canvas, const Scalar& color, int thickness) const
{
    circle(canvas, _point1.pt, 2*thickness, Scalar(0xFF, 0xFF, 0), thickness);
    circle(canvas, _point2.pt, 2*thickness, Scalar(0, 0xFF, 0xFF), thickness);
    line(canvas, _point1.pt, _point2.pt, color, thickness);
}

void Line::prolonger(Mat& canvas, Scalar color, int thickness) const
{
    int maxX = canvas.cols;
    int maxY = canvas.rows;

    Point2f top_left = Point2f(0, 0);
    Point2f top_right = Point2f(maxX, 0);
    Point2f bottom_left = Point2f(0, maxY);
    Point2f bottom_right = Point2f(maxX, maxY);

    vector<Line> borders = { Line(top_left, top_right),
                             Line(top_right, bottom_right),
                             Line(bottom_right, bottom_left),
                             Line(bottom_left, top_left) };

    vector<Point2f*> intersections;
    for (const auto& border : borders) {
        Point2f *inter = intersect(border);
        if (inter != nullptr) {
            if (inter->x >= 0 && inter->x <= maxX &&
                inter->y >= 0 && inter->y <= maxY)
                intersections.push_back(inter);
        }
    }

    for (const auto& point : intersections) {
        double distance1 = norm(*point - _point1.pt);
        double distance2 = norm(*point - _point2.pt);

        Point2f choice = distance1 < distance2 ? _point1.pt : _point2.pt;
        line(canvas, *point, choice, color, thickness);
        delete point;
    }

}

Line Line::getOrtho() const
{
    return Line(_origineOrtho, _pointOrtho);
}

Point2f Line::ortho() const
{
    /* Point2f other(_rho * cos(_theta), _rho * sin(_theta)); */
    double nrm = _a * _a + _b * _b;
    double x = -_a * _c / nrm;
    double y =  -_b * _c / nrm;

    return Point2f(x, y);
}

void Line::computeCartes()
{
    _a = _point1.pt.y - _point2.pt.y;
    _b = _point2.pt.x - _point1.pt.x;
    /*
     * _ax + by + _c = 0
     * <==>
     * _c = -_ax - by
     * or _c = cste don_c _c = -_a_point1.pt.x - b_point1.pt.y
     */
    _c = -_a * _point1.pt.x - _b * _point1.pt.y;
}

double Line::length() const
{
    return norm(_point1.pt - _point2.pt);
}

void Line::computePolar()
{
    double nrm = _a * _a + _b * _b;
    double xN = -_a * _c / nrm;
    double yN = -_b * _c / nrm;

    double rho = sqrt(xN*xN + yN*yN);
    double theta = atan2(yN, xN);
    
    _rho = rho;
    _theta = theta;
}

void Line::printPolar() const
{
    cout << _rho << "," << _theta << endl;
}

void Line::printPolar3D() const
{
    cout << _rho << "," << _theta << "," << length() << endl;
}

void Line::printCartes() const
{
    cout << "((" << _point1.pt.x << ", " << _point1.pt.y << "), (";
    cout << _point2.pt.x << ", " << _point2.pt.y << "))" << endl;
}

string Line::toString() const
{
    return to_string(_rho) + "," + to_string(_theta) + "," + to_string(length());
}

string Line::toStringCartes() const
{
    string str = "((" + to_string(_point1.pt.x) + ", " + to_string(_point1.pt.y) + "), (";
    str += to_string(_point2.pt.x) + ", " + to_string(_point2.pt.y) + "))";

    return str;
}

Vec2f Line::director() const
{
    return Vec2f(-_b, _a);
}

Point2f* Line::intersect(const Line& other) const
{
    /*
     * On vérifie si les droites sont parallèles ou confondues.
     */

    Vec2f d1 = director();
    Vec2f d2 = other.director();
    if (d1[0] * d2[1] == d1[1] * d2[0])
        return nullptr;
    
    Vec2f v1(_point2.pt - _point1.pt);
    double a1 = v1[1];
    double b1 = -v1[0];
    double c1 = a1 * _point1.pt.x + b1 * _point1.pt.y;

    Vec2f v2(other._point2.pt - other._point1.pt);
    double a2 = v2[1];
    double b2 = -v2[0];
    double c2 = a2 * other._point1.pt.x + b2 * other._point1.pt.y;

    double x = (b2 * c1 - b1 * c2) / (a1 * b2 - a2 * b1);
    double y = (a2 * c1 - a1 * c2) / (a2 * b1 - a1 * b2);

    return new Point2f(x, y);
}

bool Line::equals(const Line& other) const
{
    return _a == other._a &&
           _b == other._b &&
           _c == other._c;
}

void Line::translate(double dx, double dy)
{
    Point2f translationVector(dx, dy);
    _point1.pt += translationVector;
    _point2.pt += translationVector;

    computeCartes();
    computePolar();
}

bool Line::proche(const Line &other, int rhoTreshold, double thetaThreshold) const {
    return abs(_rho - other._rho) <= rhoTreshold &&
           abs(_theta - other._theta) <= thetaThreshold;
}

void Line::invert() {
    swap(_point1, _point2);
    computeCartes();
    computePolar();
}

Mat fullPlane(const Mat& src, std::vector<Line>& lines, int thickness)
{
    int length = src.cols;
    int height = src.rows;

    Scalar white_bg(0xFF, 0xFF, 0xFF);
    Mat dst(height * 2, length * 2, CV_8UC3, white_bg);

    /*
     * Maintenant on copie l'image dans le quadrant
     * inférieur droit
     */
    src.copyTo(dst(Rect(length, height, src.cols, src.rows)));

    /*
     * On trace ensuite les axes du repère
     */
    Point2f top(length, 0);
    Point2f bottom(length, height *  2);
    Point2f left(0, height);
    Point2f right(length * 2, height);

    line(dst, top, bottom, Scalar(0, 0, 0), thickness);
    line(dst, left, right, Scalar(0, 0, 0), thickness);
    
    for (auto& droite : lines) {
        droite.translate(length, height);
    }

    return dst;
}

const InterestPoint& Line::getPoint1() const {
    return _point1;
}

const InterestPoint& Line::getPoint2() const {
    return _point2;
}

double Line::pente() const {
    if (_b != 0)
        return -_a / _b;
    else
        return nan("inf");
}

double Line::getTheta() const {
    return _theta;
}

string Line::toStringDBSCAN() const {
    return to_string(_point1.pt.x) + "," + to_string(_point1.pt.y) + "," + to_string(_theta) + "," + to_string(length());
}

