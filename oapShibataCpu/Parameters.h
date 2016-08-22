/*
 * Copyright 2016 Marcin Matula
 *
 * This file is part of Oap.
 *
 * Oap is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Oap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Oap.  If not, see <http://www.gnu.org/licenses/>.
 */




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
