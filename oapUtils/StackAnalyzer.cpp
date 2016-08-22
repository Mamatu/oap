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



#include "StackAnalyzer.h"

namespace stack {
inline llu* getPrevRBPAddress(llu* rbp) {
  llu value_rbp = *rbp;
  return reinterpret_cast<llu*>(value_rbp);
}

inline llu* getPRBPA(llu* rbp) { return getPrevRBPAddress(rbp); }

void getInstructionsPointers(std::vector<uintt>& rips,
                             std::vector<uintt>& rbps) {
  register llu* RBP asm("rbp");
  register llu* RSP asm("rsp");
  llu* tempRBP = RBP;
  do {
    fprintf(stderr, "pointer = %x \n", *tempRBP);
    fprintf(stderr, "pointer = %x \n", *(tempRBP + 1));
    rbps.push_back(*tempRBP);
    rips.push_back(*(tempRBP + 1));
    tempRBP = getPRBPA(tempRBP);
  } while (tempRBP != NULL);
}
}
