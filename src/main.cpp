#include <iostream>

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/file.hpp>

#include <opencv2/core/utility.hpp>

#include "../include/copyMoveDetector.hpp"
#include "../include/DetectorOptions.hpp"

using namespace std;
using namespace cv;

struct DetectorOptions;

void init_logger(int level, const string& logfile = "") {
    if (!logfile.empty())
        boost::log::add_file_log(logfile);

    auto max_level = boost::log::trivial::fatal;
    auto expected_level = max_level - level;
    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= expected_level
    );
}

int main(int argc, char *argv[])
{
    const string keys =
            "{help h usage ? |      | print this message   }"
            "{@image         |<none>| The path to the image to analyze   }"
            "{mask           |      | The binary mask of the falsification }"
            "{debug d        |0     | Level of debug messages (0 to 5) }"
            "{log l          |<none>| The path to the log file }"
            "{hessian        |0     | Keypoints detection Hessian threshold }"
            "{angle          |4     | Fast g2NN algorithm threshold on angle value }"
            "{norm           |1.2   | Fast g2NN algorithm threshold on norm value }"
            "{length         |50    | Minimum length of line segments }"
            "{minPts         |10    | DBSCAN minimal number of other segments in cluster }"
            "{epsilon        |0.5   | DBSCAN size of ball considered around each segment }"
            "{wx             |0.25  | DBSCAN weight on x parameter }"
            "{wy             |0.25  | DBSCAN weight on y parameter }"
            "{wtheta         |0.25  | DBSCAN weight on theta parameter }"
            "{PSNR p         |150   | PSNR threshold for mask expansion }"
            "{keypoints kp k |      | Copies the picture and draws keypoints }"
            "{matches m      |      | Copies the picture and draws matches }"
            "{clusters c     |      | Copies the picture and draws clusters }"
            "{hulls hu       |      | Copies the picture and draws convex hulls }"
            "{expansion e    |      | Exports pictures representing step by step expansion }"
            "{before_expansion be    |      | Computes binary mask before expansion }"
            "{jobs j         |      | Number of simultaneous jobs }"
    ;

    CommandLineParser parser(argc, argv, keys);


    parser.about("copyMoveCheck is a program that checks whether an image has been tampered by one or multipe copy-move forgeries.");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    auto image = parser.get<string>("@image");
    auto mask = parser.get<string>("mask");
    auto level = parser.get<int>("debug");

    string logfile;
    if (parser.has("log")) {
        logfile = parser.get<string>("log");
    }

    auto hessian = parser.get<int>("hessian");
    auto angle = parser.get<double>("angle");
    auto norm = parser.get<double>("norm");
    auto length = parser.get<double>("length");
    auto minPts = parser.get<int>("minPts");
    auto epsilon = parser.get<double>("epsilon");
    auto wx = parser.get<double>("wx");
    auto wy = parser.get<double>("wy");
    auto wtheta = parser.get<double>("wtheta");
    auto PSNR = parser.get<double>("PSNR");

    auto kp = parser.has("keypoints");
    auto matches = parser.has("matches");
    auto clusters = parser.has("clusters");
    auto hulls = parser.has("hulls");

    auto expansion = parser.has("expansion");
    auto before_expansion = parser.has("before_expansion");

    int jobs = 1;
    if (parser.has("jobs")) {
        jobs = parser.get<int>("jobs");
    }

    if (!parser.check()) {
        parser.printMessage();
        parser.printErrors();
        return -1;
    }

    init_logger(level, logfile);

    int lastIndex = image.find_last_of('.');
    string rawName = image.substr(0, lastIndex);
    string extension = image.substr(lastIndex, string::npos);

    BOOST_LOG_TRIVIAL(debug) << "Filename: " << rawName << "." << extension;

    DetectorOptions options = {image,
                               rawName,
                               extension,
                               mask,
                               hessian,
                               angle,
                               norm,
                               length,
                               minPts,
                               epsilon,
                               wx,
                               wy,
                               wtheta,
                               PSNR,
                               kp,
                               matches,
                               clusters,
                               hulls,
                               expansion,
                               before_expansion,
                               jobs};

    defals::copyMoveDetector detector(options);
    detector.detect();
    //detector.printInfo();
    detector.show();

    return 0;
}
