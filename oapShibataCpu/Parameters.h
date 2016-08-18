
#ifndef OAP_PARAMETERS_SHIBATA_CPU_H
#define	OAP_PARAMETERS_SHIBATA_CPU_H

#include "DebugLogs.h"
#include "Matrix.h"

namespace shibataCpu {

    class Parameters {
    public:

        template <typename T>class Parameter {
        public:

            Parameter() : isAvailable(false) {
            }

            Parameter & operator=(const Parameter & orig) {
                this->isAvailable = orig.isAvailable;
                this->value = orig.value;
                return *this;
            }

            void set(T value) {
                isAvailable = true;
                this->value = value;
            }
            T value;
            bool isAvailable;
        };

        Parameters();
        virtual ~Parameters();
        Parameters(const Parameters& orig);

        bool isThreadsCount() const;
        void setThreadsCount(int count);
        int getThreadsCount() const;

        bool isSerieLimit() const;
        void setSerieLimit(int limit);
        int getSerieLimit() const;

        bool isQuantumsCount() const;
        void setQuantumsCount(int qunatumsCount);
        int getQunatumsCount() const;

        bool isSpinsCount() const;
        void setSpinsCount(int spinsCount);
        int getSpinsCount() const;

        bool isTrotterNumber() const;
        void setTrotterNumber(int spinsCount);
        int getTrotterNumber() const;

        bool areAllAvailable() const;
    private:
        Parameter<int> threadsCount;
        Parameter<int> serieLimit;
        Parameter<int> quantumsCount;
        Parameter<int> spinsCount;
        Parameter<int> trotterNumber;
        
    };
}
#endif	/* PARAMETERS_H */

