
#ifndef OGLA_GA_PROCESS_H
#define	OGLA_GA_PROCESS_H

#include "GATypes.h"

namespace ga {

    class GAProcess {
    protected:
        GAData* gaData;
    public:
        void setGAData(GAData* gaData);
        virtual int exeucte() = 0;
    };
}


#endif	/* POPULATIONINFO_H */

