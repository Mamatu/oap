#ifndef STACKANALYZER_H
#define STACKANALYZER_H

#include <vector>
#include <stdio.h>
#include "Math.h"

namespace stack {
typedef unsigned long long int llu;

void getInstructionsPointers(std::vector<uintt>& rips,
                             std::vector<uintt>& rbps);
}

#endif  // MEMORYANALYZER_H
