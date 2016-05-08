#ifndef HOST_QRTEST3_H
#define HOST_QRTEST3_H

namespace host {
namespace qrtest3 {
const char* matrix = "[4,2,2,4,4,2,4,2,4,4,2,2,4,4,4,2,2,4,4,4,4,4,4,4,4]";

const char* qref =
    "[0.60302, -0.59216, -0.36314,  0.39223,  0.00000,\
        0.30151,  0.78954, -0.36314,  0.39223,  0.00000,\
        0.30151,  0.06580,  0.60523,  0.19612, -0.70711,\
        0.30151,  0.06580,  0.60523,  0.19612,  0.70711,\
        0.60302,  0.13159, -0.06052, -0.78446,  0.00000]";

const char* rref =
    "[6.63325, 6.03023, 6.63325, 8.44232, 8.44232,\
        0.00000, 2.76340, 1.44749, 1.84226, 1.84226,\
        0.00000, 0.00000, 3.14718, 1.69464, 1.69464,\
        0.00000, 0.00000, 0.00000, 1.56893, 1.56893,\
        0.00000, 0.00000, 0.00000, 0.00000, 0.00000,]";
};
};

#endif  // QRTEST3
